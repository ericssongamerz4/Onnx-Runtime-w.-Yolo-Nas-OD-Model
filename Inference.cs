using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;

namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    public class Inference
    {

        public static (string? sortedDigits, List<(RectangleF box, string label, float score)>? detections)
            ExtractDigits(string modelPath, DenseTensor<byte> inputTensor, InferenceSession session)
        {
            try
            {
                string inputName = session.InputMetadata.Keys.First();

                var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
                };

                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

                float[] boxes = results.First(r => r.Name.Contains("pred_boxes")).AsEnumerable<float>().ToArray();
                long[] classes = results.First(r => r.Name.Contains("pred_classes")).AsEnumerable<long>().ToArray();
                float[] scores = results.First(r => r.Name.Contains("pred_scores")).AsEnumerable<float>().ToArray();
                var numPreds = results.First(r => r.Name.Contains("num_predictions")).AsEnumerable<long>().First();

                var digits = new List<(float xCenter, string digit)>();
                var detections = new List<(RectangleF box, string label, float score)>();

                for (int i = 0; i < numPreds; i++)
                {
                    long clsId = classes[i];
                    float score = scores[i];

#if DEBUG
                    Console.WriteLine($"[ALL DETECTIONS] class={clsId}, score={score:F2}");//

#endif
                    // Class 1–10 map to digits 0–9; index 0 is reserved label "digits"
                    // Filter out predictions with low confidence and class IDs that aren't numbers 0-9
                    if (clsId >= 1 && clsId <= 10 && score >= Config.confidenceThreshold)
                    {
                        float x1 = boxes[i * 4];
                        float y1 = boxes[i * 4 + 1];
                        float x2 = boxes[i * 4 + 2];
                        float y2 = boxes[i * 4 + 3];

                        float xCenter = (x1 + x2) / 2.0f;
                        // Convert class ID to digit label, assuming class IDs 1-10 correspond to digits 0-9 and there is a safeguard for unknown classes
                        string digit = Config.Labels.ElementAtOrDefault((int)clsId) ?? "unknown";

                        digits.Add((xCenter, digit));

                        var rect = new RectangleF(x1, y1, x2 - x1, y2 - y1);
                        detections.Add((rect, digit, score));
                    }
                }

                // Apply class-agnostic NMS
                var filteredDetections = Postprocessing.ApplyClassAgnosticNMS(detections);

                var orderedDigits = filteredDetections.Select(d => ((d.box.Left + d.box.Right) / 2f, d.label))
                    .OrderBy(d => d.Item1)
                    .Select(d => d.label)
                    .ToArray();

                string sortedDigits = string.Join("", orderedDigits);

                return (sortedDigits, filteredDetections);
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Error during inference: {ex.Message}");
                Console.ResetColor();
                return (null, null);
            }
        }

        public static void WarmUpSession(InferenceSession session)
        {
            string inputName = session.InputMetadata.Keys.First();

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, Config.inputTensor)
            };

            // Run inference once to warm up
            using var _ = session.Run(inputs);
        }


    }
}

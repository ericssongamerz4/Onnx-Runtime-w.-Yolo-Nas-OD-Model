using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;
namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    class Program
    {
        public static string ExtractDigits(string modelPath, string imagePath)
        {
            using var session = new InferenceSession(modelPath);

            // Load and resize image using ImageSharp
            using Image<Rgb24> image = Image.Load<Rgb24>(imagePath);
            image.Mutate(ctx => ctx.Resize(640, 640));

            // Prepare uint8[1,3,640,640] tensor
            var inputTensor = new DenseTensor<byte>(new[] { 1, 3, 640, 640 });

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < 640; y++)
                {
                    var pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < 640; x++)
                    {
                        Rgb24 pixel = pixelRow[x];
                        inputTensor[0, 0, y, x] = pixel.R;
                        inputTensor[0, 1, y, x] = pixel.G;
                        inputTensor[0, 2, y, x] = pixel.B;
                    }
                }
            });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("onnx::Cast_0", inputTensor)
            };

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(inputs);

            var boxes = results.First(r => r.Name.Contains("pred_boxes")).AsEnumerable<float>().ToArray();
            var classes = results.First(r => r.Name.Contains("pred_classes")).AsEnumerable<long>().ToArray();
            var scores = results.First(r => r.Name.Contains("pred_scores")).AsEnumerable<float>().ToArray();
            var numPreds = results.First(r => r.Name.Contains("num_predictions")).AsEnumerable<long>().First();


            var digits = new List<(float xCenter, string digit)>();
            var detections = new List<(RectangleF box, string label, float score)>();

            for (int i = 0; i < numPreds; i++)
            {
                long clsId = classes[i];
                float score = scores[i];

                if (clsId >= 1 && clsId <= 10 && score > 0.5f)
                {
                    float x1 = boxes[i * 4];
                    float y1 = boxes[i * 4 + 1];
                    float x2 = boxes[i * 4 + 2];
                    float y2 = boxes[i * 4 + 3];

                    float xCenter = (x1 + x2) / 2.0f;
                    string digit = LabelMap.Labels[clsId];

                    digits.Add((xCenter, digit));

                    var rect = new RectangleF(x1, y1, x2 - x1, y2 - y1);
                    detections.Add((rect, digit, score));
                }
            }

            var orderedDigits = digits.OrderBy(d => d.xCenter).Select(d => d.digit).ToArray();
            string sortedDigits = string.Join("", orderedDigits);
            Guid guid = Guid.NewGuid();

            // Save image with bounding boxes
            DrawDetections(image, string.Concat(@"C:\Users\hp\Desktop\test results\", sortedDigits.ToString(), "_", guid.ToString().Substring(0, 5), ".jpeg"), detections);

            return sortedDigits;

        }
        public static void DrawDetections(Image<Rgb24> image, string outputPath, List<(RectangleF box, string label, float score)> detections)
        {
            // Clone the image to avoid modifying the original
            using var outputImage = image.CloneAs<Rgba32>();
            var font = SystemFonts.CreateFont("Arial", 16);

            foreach (var det in detections)
            {
                // Find the index of the label in Labels array
                int labelIndex = Array.IndexOf(LabelMap.Labels, det.label);

                // Default to Red if label not found (or use another fallback)
                Color color = labelIndex >= 0 && labelIndex < LabelMap.LabelColors.Length
                    ? LabelMap.LabelColors[labelIndex]
                    : Color.Red;

                var pen = Pens.Solid(color, 3);

                outputImage.Mutate(x =>
                {
                    x.Draw(pen, det.box);
                    x.DrawText($"{det.label} {det.score:F2}", font, color,
                        new PointF(det.box.X, det.box.Y - 20));
                });
            }

            outputImage.Save(outputPath);
        }

        static void Main()
        {
            Stopwatch stopwatch = new();
            int totalTime = 0;
            foreach (string imagePath in LabelMap.Images)
            {            
                //Start stopwatch to measure inference time
                stopwatch.Start();
                if (!File.Exists(imagePath))
                {
                    Console.WriteLine($"Image not found: {imagePath}");
                    continue;
                }
                string result = ExtractDigits(LabelMap.ModelPath, imagePath);
                Console.WriteLine($"\nDetected Water Meter Reading: {result}");
                stopwatch.Stop();
                Console.WriteLine($"Inference Time: {stopwatch.ElapsedMilliseconds} ms");
                totalTime += (int)stopwatch.ElapsedMilliseconds;
                stopwatch.Reset();
            }

            Console.WriteLine($"\nTotal Inference Time for all images: {totalTime} ms"); 
            Console.WriteLine("Press any key to exit...");
        }
    }
}

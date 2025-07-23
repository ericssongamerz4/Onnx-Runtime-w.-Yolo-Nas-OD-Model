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
        public static string ExtractDigits(string modelPath, string imagePath, int imageNumber)
        {
            try
            {
                #region Validation
                // Check if the model file exists
                if (!File.Exists(modelPath))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Model file not found: {modelPath}");
                    Console.ResetColor();
                    return "";
                }
                // Check if the image file exists
                if (!File.Exists(imagePath))
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"Image file not found: {imagePath}");
                    Console.ResetColor();
                    return "";
                }
                #endregion

                // Create an inference session with the ONNX model
                using var session = new InferenceSession(modelPath);

                // Load the image using ImageSharp
                using Image<Rgb24> imageRaw = Image.Load<Rgb24>(imagePath);

                // Resize the image to 640x640 with padding to maintain aspect ratio and avoid distortion
                using Image<Rgb24> image = ResizeWithPadding(imageRaw);

                // Prepare uint8[1,3,640,640] tensor
                var inputTensor = new DenseTensor<byte>(new[] { 1, 3, 640, 640 });

                // Fill the tensor with pixel data from the image
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

                //Get the input name from the session metadata
                // The input name is usually the first key in the session's input metadata
                string inputName = session.InputMetadata.Keys.First();
                var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };

                // Run the inference session
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

                    Console.WriteLine($"[ALL DETECTIONS] class={clsId}, score={score:F2}");//

                    // Class 1–10 map to digits 0–9; index 0 is reserved label "digits"
                    // Filter out predictions with low confidence and class IDs that aren't numbers 0-9
                    if (clsId >= 1 && clsId <= 10 && score >= LabelMap.confidenceThreshold)
                    {
                        float x1 = boxes[i * 4];
                        float y1 = boxes[i * 4 + 1];
                        float x2 = boxes[i * 4 + 2];
                        float y2 = boxes[i * 4 + 3];

                        float xCenter = (x1 + x2) / 2.0f;
                        // Convert class ID to digit label, assuming class IDs 1-10 correspond to digits 0-9 and there is a safeguard for unknown classes
                        string digit = LabelMap.Labels.ElementAtOrDefault((int)clsId) ?? "unknown";

                        digits.Add((xCenter, digit));

                        var rect = new RectangleF(x1, y1, x2 - x1, y2 - y1);
                        detections.Add((rect, digit, score));
                    }
                }

                var orderedDigits = digits.OrderBy(d => d.xCenter).Select(d => d.digit).ToArray();
                string sortedDigits = string.Join("", orderedDigits);

                // Save image with bounding boxes
                DrawDetections(image, string.Concat(LabelMap.OutputFolder, "\\", imageNumber, "_", sortedDigits.ToString(), "_", Guid.NewGuid().ToString().Substring(0, 7), ".jpeg"), detections);

                return sortedDigits;
            }
            #region Error Handling
            catch (OnnxRuntimeException ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ONNX ERROR] Inference failed on {Path.GetFileName(imagePath)}: {ex.Message}");
                Console.ResetColor();
            }
            catch (IOException ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[I/O ERROR] File issue with {Path.GetFileName(imagePath)}: {ex.Message}");
                Console.ResetColor();
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] Unexpected issue on {Path.GetFileName(imagePath)}: {ex.Message}");
                Console.ResetColor();
            } 
            #endregion
            return string.Empty;
        }
        public static void DrawDetections(Image<Rgb24> image, string outputPath, List<(RectangleF box, string label, float score)> detections)
        {
            // Ensure the output directory exists and create it if it doesn't
            if (!Directory.Exists(LabelMap.OutputFolder))
            {
                Directory.CreateDirectory(LabelMap.OutputFolder);
            }

            // Check if there are any detections to draw if not return early
            if (detections.Count == 0)
            {
                Console.ForegroundColor = ConsoleColor.Yellow;
                Console.WriteLine("\nNo detections found.");                
                Console.ResetColor();
                return;
            }

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

                image.Mutate(x =>
                {
                    x.Draw(pen, det.box);
                    x.DrawText($"{det.label} {det.score:F2}", font, color,
                        new PointF(det.box.X, det.box.Y - 20));
                });
            }

            image.SaveAsJpeg(outputPath);
        }
        public static Image<Rgb24> ResizeWithPadding(Image<Rgb24> image, int targetWidth = 640, int targetHeight = 640, Rgb24 paddingColor = default)
        {
            if (paddingColor.Equals(default))
            {
                paddingColor = new Rgb24(114, 114, 114);//Gray color
            }

            float ratio = Math.Min((float)targetWidth / image.Width, (float)targetHeight / image.Height);
            int newWidth = (int)(image.Width * ratio);
            int newHeight = (int)(image.Height * ratio);

            image.Mutate(x => x.Resize(newWidth, newHeight));

            // Create a new padded image
            var paddedImage = new Image<Rgb24>(targetWidth, targetHeight, paddingColor);

            // Calculate offsets for centering
            int xOffset = (targetWidth - newWidth) / 2;
            int yOffset = (targetHeight - newHeight) / 2;

            paddedImage.Mutate(ctx => ctx.DrawImage(image, new Point(xOffset, yOffset), 1f));
            return paddedImage;
        }
        public static void Main()
        {
            Stopwatch stopwatch = new();
            int totalTime = 0, i = 0;        
            Console.WriteLine("Starting Water Meter Reading Detection...");

            // Replace with your actual image folder path
            string imageFolderPath = LabelMap.InputFolder; // Or provide a direct string path

            // Filter for JPEG and PNG files (you can modify this as needed)
            var imagePaths = Directory.GetFiles(imageFolderPath, "*.*")
                                      .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                                                  f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                                                  f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                                      .ToArray();

            foreach (string imagePath in imagePaths)
                //foreach (string imagePath in LabelMap.Images)
            { 
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"\nProcessing {i + 1} of {imagePaths.Length} images: {Path.GetFileName(imagePath)}");

                //Console.WriteLine($"\nProcessing {i + 1} of {LabelMap.Images.Length} images: {Path.GetFileName(imagePath)}");
                Console.ResetColor();

                //Start stopwatch to measure inference time
                stopwatch.Start();

                string result = ExtractDigits(LabelMap.ModelPath, imagePath, i+1);
                Console.WriteLine($"\nDetected Water Meter Reading: {result}");
                stopwatch.Stop();
                Console.WriteLine($"Inference Time: {stopwatch.ElapsedMilliseconds} ms");
                totalTime += (int)stopwatch.ElapsedMilliseconds;
                stopwatch.Reset();                
                i++;

            }

            Console.WriteLine($"\nTotal Inference Time for all images: {totalTime} ms");
            Console.WriteLine("Press any key to exit...");
        }
    }
}

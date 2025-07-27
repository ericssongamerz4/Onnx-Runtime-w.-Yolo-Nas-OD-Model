using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Diagnostics;
using System.Threading.Tasks;

namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    class Program
    {
        private static readonly Stopwatch stopwatch = new Stopwatch();

        public static string ExecuteModel(string modelPath, string imagePath, int imageNumber, InferenceSession session)
        {
            // Check if the image file exists
            if (!File.Exists(imagePath))
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Image file not found: {imagePath}");
                Console.ResetColor();
                return string.Empty;
            }

            try
            {
                // Load the image using ImageSharp
                using Image<Rgb24> imageRaw = Image.Load<Rgb24>(imagePath);

                // Resize the image to 640x640 with padding to maintain aspect ratio and avoid distortion
                using Image<Rgb24> image = Preprocessing.ResizeWithPadding(imageRaw);

                // Create a tensor with shape [1, 3, 640, 640]

                // Fill the tensor with pixel data from the resized image
                Preprocessing.PrepareInputTensor(image, Config.inputTensor);

                var (sortedDigits, detections, rawDetections) = Inference.ExtractDigits(modelPath, Config.inputTensor, session);

                // Check if there are any detections to draw if not return early
                if (string.IsNullOrEmpty(sortedDigits) || detections is null)
                {
                    Console.ForegroundColor = ConsoleColor.Yellow;
                    Console.WriteLine("\nNo detections found.");
                    Console.ResetColor();
                    return string.Empty;
                }

                // Save image with bounding boxes
                Postprocessing.DrawDetections(image, string.Concat(Config.OutputFolder, "\\", imageNumber, "_", sortedDigits.ToString(), "_", Guid.NewGuid().ToString().Substring(0, 7), ".jpeg"), detections);

                Logger.LogResult(Path.GetFileName(imagePath), sortedDigits, ((int)stopwatch.ElapsedMilliseconds), rawDetections);

                return sortedDigits;
            }
            catch (Exception ex)
            {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"[ERROR] {ex.Message}");
                Logger.LogError(Path.GetFileName(imagePath), ex.Message);
                Console.ResetColor();
            }

            return string.Empty;
        }

        static void Main(string[] args)
        {
            // Ensure the output directory exists and create it if it doesn't
            if (!Directory.Exists(Config.OutputFolder))
            {
                Directory.CreateDirectory(Config.OutputFolder);
            }

            int totalTime = 0, i = 0;
            Console.WriteLine("Starting Water Meter Reading Detection...");

            // Replace with your actual image folder path
            string imageFolderPath = Config.InputFolder; // Or provide a direct string path

            // Filter for JPEG and PNG files (you can modify this as needed)
            var imagePaths = Directory.GetFiles(imageFolderPath, "*.*")
                                      .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                                                  f.EndsWith(".jpeg", StringComparison.OrdinalIgnoreCase) ||
                                                  f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
                                      .ToArray();

            string modelPath = Config.ModelPath;

            // Check if the model file exists
            if (!File.Exists(modelPath)) {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"Model file not found: {modelPath}");
                Logger.LogError("Model loading", $"Model file not found: {modelPath}");
                Console.ResetColor();
            }

            // Load the ONNX model and run inference session
            using var session = new InferenceSession(modelPath);
            Inference.WarmUpSession(session);

            foreach (string imagePath in imagePaths)
            {
                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"\nProcessing {i + 1} of {imagePaths.Length} images: {Path.GetFileName(imagePath)}");
                Console.ResetColor();

                //Start stopwatch to measure inference time
                stopwatch.Start();

                string result = ExecuteModel(Config.ModelPath, imagePath, i + 1, session);
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

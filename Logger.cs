using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    public static class Logger
    {
        private static readonly string LogFilePath = Path.Combine(Config.OutputFolder, $"{DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss")}_inference_log_{Guid.NewGuid()}.txt");

        public static void LogResult(string imageName, string result, int inferenceTimeMs, List<(long clsId, float score)>? rawDetections)
        {
            var logEntry = new StringBuilder();
            logEntry.AppendLine($"Image: {imageName}");
            if (rawDetections == null || !rawDetections.Any())
            {
                logEntry.AppendLine("No detections found.");
            }
            else
            {
                foreach (var (clsId, score) in rawDetections)
                {
                    string label = Config.Labels.ElementAtOrDefault((int)clsId) ?? "unknown";
                    logEntry.AppendLine($"Class ID: {clsId}, Label: {label}, Score: {score:F2}");
                }
            }
            logEntry.AppendLine($"Result: {result}");
            logEntry.AppendLine($"Inference Time: {inferenceTimeMs} ms");
            logEntry.AppendLine($"Timestamp: {DateTime.Now}");
            logEntry.AppendLine(new string('-', 50));

            File.AppendAllText(LogFilePath, logEntry.ToString());
        }

        public static void LogError(string imageName, string errorMessage)
        {
            var logEntry = new StringBuilder();
            logEntry.AppendLine($"[ERROR] Image: {imageName}");
            logEntry.AppendLine($"Message: {errorMessage}");
            logEntry.AppendLine($"Timestamp: {DateTime.Now}");
            logEntry.AppendLine(new string('-', 50));

            File.AppendAllText(LogFilePath, logEntry.ToString());
        }
    }
}

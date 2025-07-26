using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    public class ModelPostProcessing
    {
        public static void DrawDetections(Image<Rgb24> image, string outputPath, List<(RectangleF box, string label, float score)> detections)
        {
            var font = SystemFonts.Families.FirstOrDefault().CreateFont(16);

            foreach (var det in detections)
            {
                // Find the index of the label in Labels array
                int labelIndex = Array.IndexOf(LabelMap.Labels, det.label);

                // Default to Red if label not found (or use another fallback)
                Color color = labelIndex >= 0 && labelIndex < LabelMap.LabelColors.Length
                    ? LabelMap.LabelColors[labelIndex]
                    : Color.Red;

                var pen = Pens.Solid(color, 4);

                image.Mutate(x =>
                {
                    x.Draw(pen, det.box);
                    x.DrawText($"{det.label} {det.score:F2}", font, color,
                        new PointF(det.box.X, det.box.Y - 20));
                });
            }

            image.SaveAsJpeg(outputPath);
        }


        public static float ComputeIoU(RectangleF boxA, RectangleF boxB)
        {
            float xA = Math.Max(boxA.Left, boxB.Left);
            float yA = Math.Max(boxA.Top, boxB.Top);
            float xB = Math.Min(boxA.Right, boxB.Right);
            float yB = Math.Min(boxA.Bottom, boxB.Bottom);

            float interW = Math.Max(0, xB - xA);
            float interH = Math.Max(0, yB - yA);
            float intersection = interW * interH;

            float areaA = boxA.Width * boxA.Height;
            float areaB = boxB.Width * boxB.Height;

            return intersection / (areaA + areaB - intersection);
        }

        public static List<(RectangleF box, string label, float score)> ApplyClassAgnosticNMS(
            List<(RectangleF box, string label, float score)> detections,
            float iouThreshold = 0.5f)
        {
            // 1. Sort all detections by descending score
            var sorted = detections
                .OrderByDescending(d => d.score)
                .ToList();

            var results = new List<(RectangleF, string, float)>();

            // 2. Iterate: pick highest-score box, remove any with IoU ≥ threshold
            while (sorted.Count > 0)
            {
                var current = sorted[0];
                results.Add(current);
                sorted.RemoveAt(0);

                sorted = sorted
                    .Where(d => ComputeIoU(current.box, d.box) < iouThreshold)
                    .ToList();
            }

            return results;
        }
    }
}

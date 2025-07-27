using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    public class Postprocessing
    {
        public static void DrawDetections(Image<Rgb24> image, string outputPath, List<(RectangleF box, string label, float score)> detections)
        {
            Font font = SystemFonts.Families.FirstOrDefault().CreateFont(16);

            foreach (var (box, label, score) in detections)
            {
                Color color = Config.LabelColorsDict.TryGetValue(label, out var c) ? c : Color.Red;

                var pen = Pens.Solid(color, 4);

                image.Mutate(x =>
                {
                    x.Draw(pen, box);
                    x.DrawText($"{label} {score:F2}", font, color,
                        new PointF(box.X, box.Y - 20));
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
            List<(RectangleF box, string label, float score)> detections)
        {
            var sorted = detections
                .OrderByDescending(d => d.score)
                .ToList();

            var results = new List<(RectangleF, string, float)>();

            while (sorted.Count > 0)
            {
                var current = sorted[0];
                results.Add(current);
                sorted.RemoveAt(0);

                sorted = sorted
                    .Where(d => ComputeIoU(current.box, d.box) < Config.iouThreshold)
                    .ToList();
            }

            return results;
        }
    }
}

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    public class Preprocessing
    {
        public static Image<Rgb24> ResizeWithPadding(Image<Rgb24> image)
        {
            int targetWidth = Config.imagewidth;
            int targetHeight = Config.imageHeight;

            float ratio = Math.Min((float)targetWidth / image.Width, (float)targetHeight / image.Height);
            int newWidth = (int)(image.Width * ratio);
            int newHeight = (int)(image.Height * ratio);

            image.Mutate(x => x.Resize(newWidth, newHeight));

            // Create a new padded image
            var paddedImage = new Image<Rgb24>(targetWidth, targetHeight, Config.paddingColor);

            // Calculate offsets for centering
            int xOffset = (targetWidth - newWidth) / 2;
            int yOffset = (targetHeight - newHeight) / 2;

            paddedImage.Mutate(ctx => ctx.DrawImage(image, new Point(xOffset, yOffset), 1f));
            return paddedImage;
        }

        public static DenseTensor<byte> PrepareInputTensor(Image<Rgb24> image, DenseTensor<byte> inputTensor)
        {
            // Fill the tensor with pixel data from the image
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < Config.imageHeight; y++)
                {
                    var pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < Config.imagewidth; x++)
                    {
                        Rgb24 pixel = pixelRow[x];
                        inputTensor[0, 0, y, x] = pixel.R;
                        inputTensor[0, 1, y, x] = pixel.G;
                        inputTensor[0, 2, y, x] = pixel.B;
                    }
                }
            });
            return inputTensor;
        }
    }
}

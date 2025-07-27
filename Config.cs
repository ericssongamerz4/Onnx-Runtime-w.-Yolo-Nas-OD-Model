using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    public class Config
    {
        public static readonly Dictionary<string, Color> LabelColorsDict = new()
        {
            { "digits", Color.Gray },
            { "0", Color.Green },
            { "1", Color.Blue },
            { "2", Color.Yellow },
            { "3", Color.Red },
            { "4", Color.Cyan },
            { "5", Color.Gold },
            { "6", Color.Indigo },
            { "7", Color.Orange },
            { "8", Color.Purple },
            { "9", Color.Fuchsia },
            { "border_water_meter_number", Color.Gray }
        };

        public static readonly string[] Labels =
        [
            "digits",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "border_water_meter_number"
        ];

        public static readonly string InputFolder = @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025";
        public static readonly string OutputFolder = @"C:\Users\hp\Desktop\test results\yolo-nas-v3-2234";


        public static readonly string ModelPath = @"C:\Users\hp\Downloads\yolo_nas_s_3.onnx";

        public const float confidenceThreshold = 0.5f;
        public const float iouThreshold = 0.5f;

        public static readonly DenseTensor<byte> inputTensor = new([1, 3, 640, 640]);

        public static readonly Rgb24 paddingColor = new(114, 114, 114);//Gray color

        public const int imageHeight = 640;
        public const int imagewidth = 640;

    }
}
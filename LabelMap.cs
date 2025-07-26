using SixLabors.ImageSharp;

namespace Onnx_Runtime_w._Yolo_Nas_OD_Model
{
    public class LabelMap
    {
        // The labels corresponding to the classes in the model
        public static readonly string[] Labels =
        {
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
        };

        // The colors associated with each label for visualization
        public static readonly Color[] LabelColors =
        {
            Color.Gray,    
            Color.Green,   
            Color.Blue,    
            Color.Yellow,  
            Color.Red, 
            Color.Cyan,    
            Color.Gold,   
            Color.Indigo,   
            Color.Orange,  
            Color.Purple,   
            Color.Fuchsia,     
            Color.Gray      
        };

        // The paths to the images to be processed
        public static readonly string[] Images =
            {
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\401762prueba.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\401762pruebav3.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\Flujometro 1.JPEG",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\Flujometro 2.jpg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\Flujometro 3.jpg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-8B95B137-AC5F-49B4-9A64-D352216C4C7E.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-8EF7BEB7-685D-4514-B72E-CDA7C39A5E44.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-17B04FF7-55BA-42B1-B221-ED15BBC7957E.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-855C8CD3-1D60-4657-AF91-C804CE511424.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-2484C759-3BF3-42C5-8080-592ABB24D4BC.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-C8E4155D-6F29-432A-B2EC-0F3CBAE52463.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-D418F4FF-0780-4F42-B382-DDD17F3B1C48.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-D7413884-DBB8-4950-A5FF-90954AE725DD.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-E6C02B02-D8C9-4FED-8667-4DAE34E593F5.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\original-EECBA4FD-0C6E-4DDC-8A4B-04D2FA496F30.jpeg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\thumbnail_image001.jpg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\thumbnail_image002.jpg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\069657.jpg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\Flujometro 2 copy.jpg",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\Flujometro 1 zoom.JPEG",
               @"C:\Users\hp\Documents\Proyecto Huella Hidrica\OneDrive_1_6-14-2025\Flujometro 12323 zoom.JPEG" // Test image with an invalid path


            };

        public static string InputFolder = @"C:\Users\hp\Pictures\water-meter-detection.v1i.yolov11\test\images";


        // The path to the ONNX model file 
        //public static readonly string ModelPath = @"C:\Users\hp\Downloads\yolo-nas-onnx\yolo_nas_s_2.onnx";
        //public static readonly string ModelPath = @"C:\Users\hp\Downloads\yolo_nas_s_2.5.onnx";
        public static readonly string ModelPath = @"C:\Users\hp\Downloads\yolo_nas_s_3.onnx";


        // The path to the output folder where results will be saved
        public static readonly string OutputFolder = @"C:\Users\hp\Desktop\test results\yolo-nas-v3-2234";

        // Confidence threshold for filtering predictions
        public static float confidenceThreshold = 0.5f;
    }
}
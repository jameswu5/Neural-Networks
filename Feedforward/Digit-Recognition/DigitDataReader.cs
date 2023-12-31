using System;
using System.Collections.Generic;

namespace NeuralNetworks.Feedforward {

    // credit to stackoverflow.com/questions/49407772/reading-mnist-database
    public static class DigitDataReader {
        
        public const string TrainImages = "Feedforward/Digit-Recognition/Digit-Data/train-images.idx3-ubyte";
        public const string TrainLabels = "Feedforward/Digit-Recognition/Digit-Data/train-labels.idx1-ubyte";
        public const string TestImages = "Feedforward/Digit-Recognition/Digit-Data/test-images.idx3-ubyte";
        public const string TestLabels = "Feedforward/Digit-Recognition/Digit-Data/test-labels.idx1-ubyte";

        public static List<Image> ReadTrainingData()
        {
            return Read(TrainImages, TrainLabels);
        }

        public static List<Image> ReadTestData()
        {
            return Read(TestImages, TestLabels);
        }

        private static List<Image> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            List<Image> imageList = new List<Image>();


            for (int i = 0; i < numberOfImages; i++)
            {
                byte[] bytes = images.ReadBytes(width * height);
                byte[,] arr = new byte[height, width];

                arr.ForEach((j,k) => arr[j, k] = bytes[j * height + k]);

                imageList.Add(new Image(labels.ReadByte(), arr, bytes));
            }

            return imageList;
        }

    }

    public static class Extensions {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(int));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
    
}
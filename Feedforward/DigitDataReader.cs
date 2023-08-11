using System;
using System.Collections.Generic;

namespace NeuralNetworks.Feedforward {

    // credit to stackoverflow.com/questions/49407772/reading-mnist-database
    public static class DigitDataReader {
        
        public const string TrainImages = "Feedforward/Digit-Data/train-images.idx3-ubyte";
        public const string TrainLabels = "Feedforward/Digit-Data/train-labels.idx1-ubyte";
        public const string TestImages = "Feedforward/Digit-Data/test-images.idx3-ubyte";
        public const string TestLabels = "Feedforward/Digit-Data/test-labels.idx1-ubyte";

        public static IEnumerable<Image> ReadTrainingData()
        {
            foreach (var item in Read(TrainImages, TrainLabels))
            {
                yield return item;
            }
        }

        public static IEnumerable<Image> ReadTestData()
        {
            foreach (var item in Read(TestImages, TestLabels))
            {
                yield return item;
            }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new byte[height, width];

                arr.ForEach((j,k) => arr[j, k] = bytes[j * height + k]);

                yield return new Image()
                {
                    Data = arr,
                    Label = labels.ReadByte()
                };
            }
        }

    }

    public class Image
    {
        public byte Label { get; set; }
        public byte[,] Data { get; set; }

        public void Display() {
            for (int i = 0; i < Data.GetLength(0); i++) {
                for (int j = 0; j < Data.GetLength(1); j++) {
                    string d = Data[i, j].ToString();
                    switch (d.Length) {
                        case 1:
                            Console.Write("  ");
                            Console.Write(d);
                            Console.Write("  ");
                            break;
                        case 2:
                            Console.Write(" ");
                            Console.Write(d);
                            Console.Write("  ");
                            break;
                        case 3:
                            Console.Write(" ");
                            Console.Write(d);
                            Console.Write(" ");
                            break;
                        default:
                            break;
                    }
                }
                Console.WriteLine();
            }
        }
    }

    public static class Extensions {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
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
using System;
using System.Collections.Generic;

namespace NeuralNetworks.Feedforward {

    public class Image
    {
        public int label;
        public byte[,] data;
        public double[] dataArray;

        public Image(byte label, byte[,] data, byte[] dataArray) {
            this.label = label;
            this.data = data;
            this.dataArray = GetImageArray(dataArray);
        }

        public void Display() {
            for (int i = 0; i < data.GetLength(0); i++) {
                for (int j = 0; j < data.GetLength(1); j++) {
                    string d = data[i, j].ToString();
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

        public static double[] GetImageArray(byte[] array) {

            double[] res = new double[array.Length];
            for (int i = 0; i < array.Length; i++) {
                res[i] = array[i];
            }
            return res;
        }

    }

    
}
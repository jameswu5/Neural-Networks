using System;
using System.Collections.Generic;
using System.Linq;

namespace RecurrentNeuralnetwork {

    public class Program {
        public static void Main(string[] args) {
            double[,] matrix1 = {
                {0, 4},
                {1, 5},
                {2, 6}
            };
            double[,] matrix2 = {
                {1, 2, 3, 4},
                {5, 6, 7, 8}
            };

            Matrix.Display(Matrix.MatrixMultiply(matrix1, matrix2));

            double[] vector1 = {3, 5, 6};
            double[] vector2 = {1, 6.5, 2};
            Console.WriteLine(Matrix.DotProduct(vector1, vector2));

            Matrix.Display(Matrix.Transpose(matrix1));
        }
    }
}
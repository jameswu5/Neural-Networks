using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks {

    public class Program {

        public static void Main() {
            // Recurrent.Program.TrainLanguages(10);
            // TestUtility();
            // Feedforward.Program.TestForwardPropagation();
            // Feedforward.Program.TestImages();

            // Feedforward.DigitRecognition.TrainDefault();
            // Feedforward.DigitRecognition.TestOneImage();
            Feedforward.DigitRecognition.TestDefault();
            // Feedforward.DigitRecognition.TrainOneBatch();
            // Feedforward.DigitRecognition.TrainIndividual(5);
        }
    
        public static void TestUtility() {
            // everything seems to work properly
            double[][] matrix0 = new[] {
                new double[] {1, 2, 3},
                new double[] {4, 5, 6},
                new double[] {7, 8, 9},
                new double[] {9, 10, 11}
            };
            double[,] matrix1 = {
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9},
                {9, 10, 11}
            };
            double[,] matrix2 = {
                {10, 11, 12},
                {13, 14, 15},
                {16, 17, 17},
                {18, 19, 20}
            };
            // Matrix.Display(matrix0[^1]);
            // Matrix.Display(Matrix.Add(matrix1, matrix2));
            
            double[] vector1 = {1, 4, 7};
            double[] vector2 = {0.5, 1, 1.5};
            double[] vector3 = {0.9, 1, 1.1};
            // Matrix.Display(Matrix.Add(vector1, vector2));
            // Matrix.Display(Matrix.Add(vector1, vector2, vector3));
            // Matrix.Display(Matrix.Add(matrix1, 5));
            // Matrix.Display(Matrix.Add(vector1, 1.2));
            // Matrix.Display(Matrix.ScalarMultiply(matrix1, 2));
            // Matrix.Display(Matrix.ScalarMultiply(vector1, 1.5));

            double[,] matrix3 = {
                {10, 11, 12, 20, 21},
                {13, 14, 15, 4, 1},
                {16, 17, 17, 9, 2},
            };

            // Matrix.Display(Matrix.MatrixMultiply(matrix1, matrix3));
            // Matrix.Display(Matrix.MatrixMultiply(matrix1, vector2));
            double[] vector4 = {-2, 1, 6, -1.5};

            // Matrix.Display(Matrix.MatrixMultiply(vector1, vector4));
            // Matrix.Display(Matrix.Transpose(matrix3));

            // Matrix.Display(Activation.Sigmoid(vector4));
            // Matrix.Display(Derivative.Sigmoid(vector4));

            // Matrix.Display(Matrix.Concatenate(vector1, vector4));

            double[,] matrix4 = {
                {1, 4}, {2, 5}, {3, 6}
            };
            double[] vector5 = {1, 2, 3};
            // Matrix.Display(Matrix.MatrixMultiply(vector5, matrix4));

            // Console.WriteLine(Loss.MeanSquaredError(vector1, vector2));


            List<double> vector6 = new List<double>() {1,2,3,4,5,6,7,8,9,10};

            foreach (double d in vector6.GetRange(3,4)) {
                Console.WriteLine(d);
            }
        }

    }
}
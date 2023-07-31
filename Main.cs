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

            double[,] matrix3 = {
                {0, 1},
                {2, 3},
                {4, 5}
            };

            // Matrix.Display(Matrix.MatrixMultiply(matrix1, matrix2));

            // Matrix.Display(Matrix.Subtract(matrix1, matrix3));
            // Console.WriteLine();
            // Matrix.Display(matrix1);


            double[] vector1 = {3, 5, 6};
            double[] vector2 = {1, 6.5, 2};
            double[] vector = {2, 3};

            // Matrix.Display(Matrix.MatrixMultiply(matrix3, vector));

            // Matrix.Display(Matrix.Transpose(matrix1));

            double[] soft = Activation.Softmax(vector1);
            double[] tanh = Activation.Tanh(vector1);

            // foreach (double s in tanh) {
            //     Console.WriteLine(s);
            // }


            double[] vector3 = {0.15, 0.23, 0.62};
            double[] vector4 = {0.9, 0.1, 0};
            // Console.WriteLine(Loss.CrossEntropy(vector3, vector4));

            double[] vector5 = {1, 2, 3, 4};

            Network nn = new Network(4, 3, 5, 0.1);
            Matrix.Display(nn.ForwardPropogate(vector5, vector4));

        }
    }
}


using System;
using System.Collections.Generic;
using System.Linq;


namespace RecurrentNeuralnetwork {
    public static class Matrix {

        // matrix1 + matrix2
        public static double[,] Add(double[,] matrix1, double[,] matrix2) {

            if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1)) {
                throw new ArgumentException("Matrix sizes are not compatible");
            }

            for (int i = 0; i < matrix2.GetLength(0); i++) {
                for (int j = 0; j < matrix2.GetLength(1); j++) {
                    matrix1[i,j] += matrix2[i,j];
                }
            }

            return matrix1;
        }

        // matrix1 - matrix2
        public static double[,] Subtract(double[,] matrix1, double[,] matrix2) {
            if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1)) {
                throw new ArgumentException("Matrix sizes are not compatible");
            }

            for (int i = 0; i < matrix2.GetLength(0); i++) {
                for (int j = 0; j < matrix2.GetLength(1); j++) {
                    matrix1[i,j] -= matrix2[i,j];
                }
            }

            return matrix1;
        }
    
        // a * matrix
        public static double[,] ScalarMultiply(double[,] matrix, double scalar) {
            for (int i = 0; i < matrix.GetLength(0); i++) {
                for (int j = 0; j < matrix.GetLength(1); j++) {
                    matrix[i,j] *= scalar;
                }
            }

            return matrix;
        }

        // matrix1 * matrix2 (standard O(n^3) algorithm, might be able to optimise)
        public static double[,] MatrixMultiply(double[,] matrix1, double[,] matrix2) {
            int rows1 = matrix1.GetLength(0);
            int cols1 = matrix1.GetLength(1);
            int rows2 = matrix2.GetLength(0);
            int cols2 = matrix2.GetLength(1);

            if (cols1 != rows2) {
                throw new ArgumentException("Matrix sizes are not compatible for multiplication");
            }

            double[,] result = new double[rows1, cols2];

            for (int i = 0; i < rows1; i++) {
                for (int j = 0; j < cols2; j++) {
                    for (int k = 0; k < cols1; k++) {
                        result[i,j] += matrix1[i,k] * matrix2[k,j];
                    }
                }
            }
            return result;
        }

        public static double DotProduct(double[] vector1, double[] vector2) {
            if (vector1.Length != vector2.Length) {
                throw new ArgumentException("Vector sizes are not the same");
            }

            double res = 0;

            for (int i = 0; i < vector1.Length; i++) {
                res += vector1[i] * vector2[i];
            }

            return res;
        }

        public static double[,] Transpose(double[,] matrix) {
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);

            double[,] res = new double[n,m];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    res[j,i] = matrix[i,j];
                }
            }
            return res;
        }

        public static void Display(double[,] matrix) {
            for (int i = 0; i < matrix.GetLength(0); i++) {
                for (int j = 0; j < matrix.GetLength(1); j++) {
                    Console.Write(matrix[i,j]);
                    Console.Write(" ");
                }
                Console.WriteLine();
            }
        }

    }
}
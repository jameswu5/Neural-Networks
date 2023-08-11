using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks {
    public static class Matrix {

        public static double[,] Add(double[,] matrix1, double[,] matrix2) {

            if (matrix1.GetLength(0) != matrix2.GetLength(0) || matrix1.GetLength(1) != matrix2.GetLength(1)) {
                throw new ArgumentException($"Matrix sizes [{matrix1.GetLength(0)},{matrix1.GetLength(1)}] and [{matrix2.GetLength(0)},{matrix2.GetLength(1)}] are not compatible");
            }
            
            int m = matrix1.GetLength(0);
            int n = matrix1.GetLength(1);

            double[,] result = new double[m, n];

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    result[i, j] = matrix1[i,j] + matrix2[i,j];
                }
            }

            return result;
        }

        public static double[] Add(double[] vector1, double[] vector2) {
            if (vector1.Length != vector2.Length) {
                throw new ArgumentException($"vector lengths [{vector1.Length}, {vector2.Length}] don't match");
            }

            double[] result = new double[vector1.Length];
            for (int i = 0; i < vector1.Length; i++) {
                result[i] = vector1[i] + vector2[i];
            }
            return result;
        }

        public static double[] Add(double[] vector1, double[] vector2, double[] vector3) {
            if (!(vector1.Length == vector2.Length && vector1.Length == vector3.Length)) {
                throw new ArgumentException("vector lengths don't match");
            }

            double[] result = new double[vector1.Length];
            for (int i = 0; i < vector1.Length; i++) {
                result[i] = vector1[i] + vector2[i] + vector3[i];
            }
            return result;
        }

        public static double[,] Add(double[,] matrix, double scalar) {
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);
            double[,] result = new double[m, n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    result[i, j] = matrix[i, j] + scalar;
                }
            }
            return result;
        }

        public static double[] Add(double[] vector, double scalar) {
            int n = vector.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = vector[i] + scalar;
            }
            return result;
        }
    
        // a * matrix
        public static double[,] ScalarMultiply(double[,] matrix, double scalar) {
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);

            double[,] result = new double[m,n];

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    result[i,j] = matrix[i,j] * scalar;
                }
            }

            return result;
        }

        public static double[] ScalarMultiply(double[] vector, double scalar) {
            int n = vector.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = vector[i] * scalar;
            }
            return result;
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

        public static double[] MatrixMultiply(double[,] matrix, double[] vector) {
            if (vector.Length != matrix.GetLength(1)) {
                throw new ArgumentException($"Matrix [{matrix.GetLength(1)}] and vector [{vector.Length}] size not compatible");
            }

            double[] result = new double[matrix.GetLength(0)];
            for (int i = 0; i < matrix.GetLength(0); i++) {
                for (int j = 0; j < vector.Length; j++) {
                    result[i] += matrix[i,j] * vector[j];
                }
            }

            return result;
        }

        public static double[] MultiplyVectorElementwise(double[] vector1, double[] vector2) {
            if (vector1.Length != vector2.Length) {
                throw new ArgumentException("Vector sizes don't match");
            }
            int n = vector1.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = vector1[i] * vector2[i];
            }
            return result;
        }

        // vector1 column vector, vector2 row vector, multiply to get matrix
        public static double[,] MatrixMultiply(double[] vector1, double[] vector2) {
            int m = vector1.Length;
            int n = vector2.Length;
            double[,] result = new double[m,n];

            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    result[i,j] = vector1[i] * vector2[j];
                }
            }

            return result;
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

        public static double[] Concatenate(double[] vector1, double[] vector2) {
            int m = vector1.Length;
            int n = vector2.Length;
            double[] result = new double[m+n];
            for (int i = 0; i < m; i++) {
                result[i] = vector1[i];
            }
            for (int i = 0; i < n; i++) {
                result[i + m] = vector2[i];
            }

            return result;
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
    
        public static void Display(double[][] matrix) {
            for (int i = 0; i < matrix.Length; i++) {
                for (int j = 0; j < matrix[i].Length; j++) {
                    Console.Write(matrix[i][j]);
                    Console.Write(" ");
                }
                Console.WriteLine();
            }
        }

        public static void Display(double[] vector) {
            foreach (double v in vector) {
                Console.Write($"{v} ");
            }
            Console.WriteLine();
        }
    
        public static double[,] InitialiseWeights(int m, int n) {
            double[,] weights = new double[m,n];

            Random rng = new Random();
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    double val = (rng.NextDouble() - 0.5) * 2 / Math.Sqrt(n);
                    weights[i,j] = val;
                }
            }

            return weights;
        }
    
        public static double[,] Clip(double[,] matrix, double lowerBound, double upperBound) {
            int m = matrix.GetLength(0);
            int n = matrix.GetLength(1);

            double[,] result = new double[m,n];
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    if (matrix[i,j] < lowerBound) {
                        result[i,j] = lowerBound;
                    } else if (matrix[i,j] > upperBound) {
                        result[i,j] = upperBound;
                    } else {
                        result[i,j] = matrix[i,j];
                    }
                }
            }
            return result;
        }

        public static double[] Clip(double[] vector, double lowerBound, double upperBound) {
            int n = vector.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                if (vector[i] < lowerBound) {
                    result[i] = lowerBound;
                } else if (vector[i] > upperBound) {
                    result[i] = upperBound;
                } else {
                    result[i] = vector[i];
                }
            }
            return result;
        }
    }

    public static class Activation {

        public static double[] Tanh(double[] vector) {
            int n = vector.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = Math.Tanh(vector[i]);
            }
            return result;
        }

        public static double[] Sigmoid(double[] vector) {
            int n = vector.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = 1 / (1 + Math.Exp(-vector[i]));
            }
            return result;
        }

        public static double[] Softmax(double[] vector) {
            int n = vector.Length;
            double maxValue = vector.Max();

            double[] result = new double[n];

            for (int i = 0; i < n; i++) {
                result[i] = Math.Exp(vector[i] - maxValue);
            }
            double sum = result.Sum();

            for (int i = 0; i < n; i++) {
                result[i] = result[i] / sum;
            }

            return result;
        }
    }

    public static class Derivative {
        public static double[] Tanh(double[] vector) {
            double[] temp = Matrix.Add(Matrix.MultiplyVectorElementwise(vector, vector), -1);
            return Matrix.ScalarMultiply(temp, -1);
        }

        public static double[] Sigmoid(double[] vector) {
            double[] temp = Matrix.Add(vector, -1);
            return Matrix.MultiplyVectorElementwise(vector, Matrix.ScalarMultiply(temp, -1));
        }
    }

    public static class Loss {
        public static double CrossEntropy(double[] vector, double[] target) {
            if (vector.Length != target.Length) {
                throw new ArgumentException("Vector sizes are not the same");
            }
            
            double result = 0;

            for (int i = 0; i < vector.Length; i++) {
                result -= target[i] * Math.Log2(vector[i]);
            }

            return result;
        }
    }

    public static class Utility {
        // Fisher-Yates shuffle
        public static void Shuffle<T>(IList<T> list)  
        {  
            Random rng = new Random();
            int n = list.Count;  
            while (n > 1) {  
                n--;
                int k = rng.Next(n + 1);
                (list[n], list[k]) = (list[k], list[n]);
            }
        }
    }

}
using System;
using System.Dynamic;

namespace NeuralNetworks {
    public class Activation {

        public enum Type { Tanh, Sigmoid, ReLu };
        public Type type;

        public Activation(Type type) {
            this.type = type;
        }

        public double[] GetActivation(double[] vector) {
            switch (type) {
                case Type.Tanh:
                    return Tanh(vector);
                case Type.Sigmoid:
                    return Sigmoid(vector);
                case Type.ReLu:
                    return ReLu(vector);
                default:
                    throw new Exception("Do not have activation type");
            }
        }

        public static double[] Tanh(double[] vector) {
            int n = vector.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = Math.Tanh(vector[i]);
            }
            return result;
        }

        public static double Sigmoid(double value) {
            return 1 / (1 + Math.Exp(-value));
        }

        public static double[] Sigmoid(double[] vector) {
            int n = vector.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = Sigmoid(vector[i]);
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

        public static double[] ReLu(double[] vector) {
            int n = vector.Length;
            double[] result = new double[n];
            for (int i = 0; i < n; i++) {
                result[i] = Math.Max(0, vector[i]);
            }
            return result;
        }
    }

    public class Derivative {
        public enum Type { Tanh, Sigmoid, ReLu };
        public Type type;

        public Derivative(Type type) {
            this.type = type;
        }

        public double[] GetDerivative(double[] vector) {
            switch (type) {
                case Type.Tanh:
                    return Tanh(vector);
                case Type.Sigmoid:
                    return Sigmoid(vector);
                case Type.ReLu:
                    return ReLu(vector);
                default:
                    throw new Exception("Do not have derivative type");
            }
        }

        public double GetDerivative(double value) {
            switch (type) {
                case Type.Tanh:
                    return Tanh(value);
                case Type.Sigmoid:
                    return Sigmoid(value);
                case Type.ReLu:
                    return ReLu(value);
                default:
                    throw new Exception("Do not have derivative type for single value");
            }
        }

        public static double[] Tanh(double[] vector) {
            double[] temp = Matrix.Add(Matrix.MultiplyVectorElementwise(vector, vector), -1);
            return Matrix.ScalarMultiply(temp, -1);
        }

        public static double Tanh(double value) {
            return 1 - value * value;
        }

        public static double[] Sigmoid(double[] vector) {
            double[] temp = Matrix.Add(vector, -1);
            return Matrix.MultiplyVectorElementwise(vector, Matrix.ScalarMultiply(temp, -1));
        }

        public static double Sigmoid(double value) {
            // assumes value has sigmoid already applied to it
            return value * (1 - value);
        }

        public static double MeanSquaredError(double observed, double expected) {
            return 2 * (observed - expected);
        }

        public static double[] ReLu(double[] vector) {
            double[] res = new double[vector.Length];
            for (int i = 0; i < vector.Length; i++) {
                if (vector[i] > 0) {
                    res[i] = 1;
                }
            }
            return res;
        }
        
        public static double ReLu(double value) {
            return value > 0 ? 1 : 0;
        }
    }

    public class Loss {
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

        public static double MeanSquaredError(double[] vector, double[] target) {
            if (vector.Length != target.Length) {
                throw new ArgumentException("Vector sizes are not the same");
            }

            double result = 0;

            for (int i = 0; i < vector.Length; i++) {
                result += (vector[i] - target[i]) * (vector[i] - target[i]);
            }

            return result / vector.Length;
            
        }
    }


}
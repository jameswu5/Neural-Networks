using System;
using System.Dynamic;

namespace NeuralNetworks {

    public interface IActivation {

        double[] Activate(double[] vector);
        double Activate(double value);

        double[] Derivative(double[] vector);
        double Derivative(double value);
    }

    public class Activation {

        public enum Type { Tanh, Sigmoid, ReLu, Softmax};

        public static IActivation GetActivationFromType(Type type) {
            switch (type) {
                case Type.Tanh:
                    return new Tanh();
                case Type.Sigmoid:
                    return new Sigmoid();
                case Type.ReLu:
                    return new ReLu();
                case Type.Softmax:
                    return new Softmax();
                default:
                    throw new Exception("Cannot identify type of activation");
            }
        }

        public class Tanh : IActivation {
            public double[] Activate(double[] vector) {
                int n = vector.Length;
                double[] result = new double[n];
                for (int i = 0; i < n; i++) {
                    result[i] = Math.Tanh(vector[i]);
                }
                return result;
            }

            public double Activate(double value) {
                return Math.Tanh(value);
            }

            public double[] Derivative(double[] vector) {
                double[] temp = Matrix.Add(Matrix.MultiplyVectorElementwise(vector, vector), -1);
                return Matrix.ScalarMultiply(temp, -1);
            }

            public double Derivative(double value) {
                return 1 - value * value;
            }
        }

        public class Sigmoid : IActivation {
            public double[] Activate(double[] vector) {
                int n = vector.Length;
                double[] result = new double[n];
                for (int i = 0; i < n; i++) {
                    result[i] = Activate(vector[i]);
                }
            return result;
            }

            public double Activate(double value) {
                return 1 / (1 + Math.Exp(-value));
            }

            public double[] Derivative(double[] vector) {
                double[] temp = Matrix.Add(vector, -1);
                return Matrix.MultiplyVectorElementwise(vector, Matrix.ScalarMultiply(temp, -1));
            }

            public double Derivative(double value) {
                return value * (1 - value);
            }
        }

        public class ReLu : IActivation {
            public double[] Activate(double[] vector) {
                int n = vector.Length;
                double[] result = new double[n];
                for (int i = 0; i < n; i++) {
                    result[i] = Math.Max(0, vector[i]);
                }
                return result;
            }

            public double Activate(double value) {
                return Math.Max(0, value);
            }

            public double[] Derivative(double[] vector) {
                double[] res = new double[vector.Length];
                for (int i = 0; i < vector.Length; i++) {
                    if (vector[i] > 0) {
                        res[i] = 1;
                    }
                }
                return res;
            }

            public double Derivative(double value) {
                return value > 0 ? 1 : 0;
            }
        }

        public class Softmax : IActivation {
            public double[] Activate(double[] vector) {
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

            // Need to implement
            public double Activate(double value) {
                throw new Exception("Not implemented Softmax.Activate(double value)");
            }

            public double[] Derivative(double[] vector) {
                throw new Exception("Not implemented Softmax.Derivative(double[] vector)");

            }

            public double Derivative(double value) {
                throw new Exception("Not implemented Softmax.Derivative(double value)");
            }
        }
    }
}
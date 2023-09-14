using System;

namespace NeuralNetworks {

    public interface ILoss {
        double LossFunction(double[] vector, double[] target);
        double LossDerivative(double value, double target);
    }

    public class Loss {

        public enum Type { MeanSquaredError, CrossEntropy };

        public static ILoss GetLossFromType(Type type) {
            switch (type) {
                case Type.MeanSquaredError:
                    return new MeanSquaredError();
                case Type.CrossEntropy:
                    return new CrossEntropy();
                default:
                    throw new Exception("Cannot identify type of loss");
            }
        }

        public class MeanSquaredError : ILoss {
            public double LossFunction(double[] vector, double[] target) {
                if (vector.Length != target.Length) {
                    throw new ArgumentException("Vector sizes are not the same");
                }

                double result = 0;

                for (int i = 0; i < vector.Length; i++) {
                    result += (vector[i] - target[i]) * (vector[i] - target[i]);
                }

                return result / vector.Length;
            }

            public double LossDerivative(double value, double target) {
                return 2 * (value - target);
            }
        }

        public class CrossEntropy : ILoss {
            public double LossFunction(double[] vector, double[] target) {
                if (vector.Length != target.Length) {
                    throw new ArgumentException("Vector sizes are not the same");
                }
                
                double result = 0;

                for (int i = 0; i < vector.Length; i++) {
                    result -= target[i] * Math.Log2(vector[i]);
                }

                return result;
            }

            public double LossDerivative(double value, double target) {
                if (value == 0 || value == 1) return 0;
                return (-value + target) / (value * (value - 1));
            }
        }
    }

}
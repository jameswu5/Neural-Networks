using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Feedforward {
    public class Feedforward {

        int[] layerSizes;
        int numberOfLayers;

        double[][,] weights;
        double[][] biases;
        double[][] layers;

        double learnRate = 1;

        public Feedforward(int[] layerSizes) {
            this.layerSizes = layerSizes;
            numberOfLayers = layerSizes.Length;

            weights = new double[numberOfLayers - 1][,];
            biases = new double[numberOfLayers - 1][];
            layers = new double[numberOfLayers][];

            // Initialise weights and biases
            for (int i = 1; i < numberOfLayers; i++) {
                weights[i - 1] = Matrix.InitialiseWeights(layerSizes[i-1], layerSizes[i]);
                biases[i - 1] = new double[layerSizes[i]];
            }
        }

        public double[] ForwardPropagate(double[] vector) {
            layers = new double[numberOfLayers][];
            layers[0] = vector;

            // assumes inputVector is already normalised
            for (int i = 0; i < numberOfLayers - 1; i++) {
                vector = Matrix.Add(Matrix.MatrixMultiply(vector, weights[i]), biases[i]);
                vector = Activation.Sigmoid(vector);
                layers[i + 1] = vector;
            }
            // apply softmax on the output vector
            vector = Activation.Softmax(vector);
            return vector;
        }

        public double[] GetOutputLayerNodeValues(double[] expectedOutput) {

            double[] nodeValues = new double[expectedOutput.Length];
            double[] outputLayer = layers[^1];

            for (int i = 0; i < expectedOutput.Length; i++) {
                double costDerivative = Derivative.MeanSquaredError(outputLayer[i], expectedOutput[i]);
                double activationDerivative = Derivative.Sigmoid(outputLayer[i]);
                nodeValues[i] = costDerivative * activationDerivative;
            }
            return nodeValues;
        }

        public static double[] GetHiddenLayerNodeValues(double[] hiddenLayer, double[] higherLayerNodeValues, double[,] weightMatrix) {
            double[] nodeValues = Matrix.MatrixMultiply(weightMatrix, higherLayerNodeValues);
            for (int i = 0; i < hiddenLayer.Length; i++) {
                double layerNode = hiddenLayer[i];
                double derivative = Derivative.Sigmoid(layerNode);
                nodeValues[i] = nodeValues[i] * derivative;
            }
            return nodeValues;
        }

        public double[][] GetNodeValues(double[] expectedOutput) {
            double[][] nodeValues = new double[numberOfLayers - 1][];
            nodeValues[0] = GetOutputLayerNodeValues(expectedOutput);

            for (int i = 0; i < numberOfLayers - 2; i++) {
                double[,] weightMatrix = weights[numberOfLayers - 2 - i];
                double[] layerNodeValues = GetHiddenLayerNodeValues(layers[numberOfLayers - 2 - i], nodeValues[i], weightMatrix);
                nodeValues[i + 1] = layerNodeValues;
            }

            Utility.Reverse(nodeValues);

            return nodeValues;
        }
    }
}
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Feedforward {
    public class Feedforward {

        public int[] layerSizes;
        public int numberOfLayers;

        public double[][,] weights;
        public double[][] biases;
        double[][] layers;

        const double learnRate = 0.05;

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

        public Feedforward(string path) {
            string[] networkData = File.ReadAllLines(path);
            string[] sizeData = networkData[0].TrimEnd(' ').Split(' ');
            layerSizes = Array.ConvertAll(sizeData, int.Parse);
            numberOfLayers = layerSizes.Length;

            weights = new double[numberOfLayers - 1][,];
            biases = new double[numberOfLayers - 1][];
            layers = new double[numberOfLayers][];

            int pointer = 1;

            for (int i = 0; i < numberOfLayers - 1; i++) {
                double[,] weightMatrix = new double[layerSizes[i], layerSizes[i+1]];

                for (int j = 0; j < layerSizes[i]; j++) {
                    string[] row = networkData[pointer + j].Split(' ');
                    for (int k = 0; k < layerSizes[i+1]; k++) {
                        weightMatrix[j, k] = double.Parse(row[k]);
                    }
                }
                weights[i] = weightMatrix;
                pointer += layerSizes[i];
            }

            for (int i = 0; i < numberOfLayers - 1; i++, pointer++) {
                double[] biasVector = new double[layerSizes[i+1]];
                string[] row = networkData[pointer].Split(' ');
                for (int j = 0; j < layerSizes[i+1]; j++) {
                    biasVector[j] = double.Parse(row[j]);
                }
                biases[i] = biasVector;
            }
        }

        public double[] ForwardPropagate(double[] vector) {
            layers = new double[numberOfLayers][];
            vector = Activation.Sigmoid(vector);
            layers[0] = vector;

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
                double derivative = Derivative.Sigmoid(hiddenLayer[i]);
                nodeValues[i] *= derivative;
            }
            return nodeValues;
        }

        public double[][] GetNodeValues(double[] expectedOutput) {
            double[][] nodeValues = new double[numberOfLayers - 1][];
            nodeValues[0] = GetOutputLayerNodeValues(expectedOutput);

            for (int i = 0; i < numberOfLayers - 2; i++) {
                double[,] weightMatrix = weights[numberOfLayers - 2 - i];
                nodeValues[i+1] = GetHiddenLayerNodeValues(layers[numberOfLayers - 2 - i], nodeValues[i], weightMatrix);
            }

            Utility.Reverse(nodeValues);

            return nodeValues;
        }

        public (double[][,] weightDerivatives, double[][] biasDerivatives) BackPropagate(double[] expectedOutput) {

            double[][] nodeValues = GetNodeValues(expectedOutput);

            double[][,] weightDerivatives = new double[numberOfLayers - 1][,];
            double[][] biasDerivatives = new double[numberOfLayers - 1][];

            for (int index = 0; index < numberOfLayers - 1; index++) {
                double[,] weightDerivative = new double[layerSizes[index], layerSizes[index + 1]];

                for (int inNodeIndex = 0; inNodeIndex < layerSizes[index]; inNodeIndex++) {
                    for (int outNodeIndex = 0; outNodeIndex < layerSizes[index + 1]; outNodeIndex++) {
                        weightDerivative[inNodeIndex, outNodeIndex] = layers[index][inNodeIndex] * nodeValues[index][outNodeIndex];
                    }
                }
                weightDerivatives[index] = weightDerivative;

                double[] biasDerivative = new double[layerSizes[index + 1]];
                for (int nodeIndex = 0; nodeIndex < layerSizes[index + 1]; nodeIndex++) {
                    biasDerivative[nodeIndex] = 1 * nodeValues[index][nodeIndex];
                }
                biasDerivatives[index] = biasDerivative;
            }

            return (weightDerivatives, biasDerivatives);
        }

        public void UpdateWeightsAndBiases(double[][,] weightDerivatives, double[][] biasDerivatives, double rate = learnRate) {
            for (int i = 0; i < numberOfLayers - 1; i++) {
                weights[i] = Matrix.Add(weights[i], Matrix.ScalarMultiply(weightDerivatives[i], -rate));
                biases[i] = Matrix.Add(biases[i], Matrix.ScalarMultiply(biasDerivatives[i], -rate));
            }
        }

        public double[] GetOneHotVector(int label) {
            double[] res = new double[layerSizes[^1]];
            res[label] = 1;
            return res;
        }

        public static int CheckIfCorrect(double[] outputVector, double[] expectedVector) {
            return Array.IndexOf(outputVector, outputVector.Max()) == Array.IndexOf(expectedVector, expectedVector.Max()) ? 1 : 0;
        }

        public void SaveNetwork(string path) {
            using (StreamWriter writer = new StreamWriter(path)) {
                foreach (int l in layerSizes) {
                    writer.Write($"{l} ");
                }
                writer.WriteLine();

                foreach (double[,] weightMatrix in weights) {
                    for (int i = 0; i < weightMatrix.GetLength(0); i++) {
                        for (int j = 0; j < weightMatrix.GetLength(1); j++) {
                            writer.Write(weightMatrix[i,j]);
                            writer.Write(" ");
                        }
                        writer.WriteLine();
                    }
                }

                foreach (double[] biasVector in biases) {
                    for (int i = 0; i < biasVector.Length; i++) {
                        writer.Write(biasVector[i]);
                        writer.Write(" ");
                    }
                    writer.WriteLine();
                }
            }
        }
    }
}
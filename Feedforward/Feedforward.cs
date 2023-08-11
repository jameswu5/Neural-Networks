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
    }
}
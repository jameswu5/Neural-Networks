using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;

namespace NeuralNetworks.Feedforward {
    public static class Program {
        public static void TestForwardPropagation() {

            int[] layerSizes = {4, 10, 5};
            double[] input = {0.2, 0.5, -0.6, 0};
            Feedforward network = new Feedforward(layerSizes);

            double[] output = network.ForwardPropagate(input);
            Matrix.Display(output);

            double[] expectedOutput = {0, 1, 0, 0, 0};

            (double[][,] dweights, double[][] dbiases) d = network.BackPropagate(expectedOutput);

            foreach (double[,] dweight in d.dweights) {
                Matrix.Display(dweight);
            }
            Matrix.Display(d.dbiases);
        }

        public static void TrainDefault() {
            List<Image> trainingSet = DigitDataReader.ReadTrainingData();
            TrainNetwork(trainingSet, 100, 10);
        }


        public static void TrainNetwork(List<Image> trainingSet, int batchSize, int epochs) {
            int[] layerSizes = {784, 50, 10};
            Feedforward network = new(layerSizes);

            for (int epoch = 1; epoch <= epochs; epoch++) {
                Console.WriteLine($"Epoch {epoch}");
                Utility.Shuffle(trainingSet);
                for (int i = 0; i < trainingSet.Count / batchSize; i++) {
                    List<Image> trainingBatch = trainingSet.GetRange(i * batchSize, batchSize);
                    Console.Write($"Epoch {epoch}, batch {i + 1}: ");
                    TrainBatch(trainingBatch, network);
                }
            }
        }

        public static void TrainBatch(List<Image> batch, Feedforward network) {

            double[][,] weightGradients = new double[network.numberOfLayers - 1][,];
            double[][] biasGradients = new double[network.numberOfLayers - 1][];

            for (int i = 0; i < network.numberOfLayers - 1; i++) {
                weightGradients[i] = new double[network.layerSizes[i], network.layerSizes[i + 1]];
                biasGradients[i] = new double[network.layerSizes[i+1]];
            }

            double totalCost = 0;
            double correct = 0;

            foreach (Image image in batch) {
                double[] outputVector = network.ForwardPropagate(image.dataArray);
                double[] expectedVector = network.GetOneHotVector(image.label);
                correct += Feedforward.CheckIfCorrect(outputVector, expectedVector);
                totalCost += Loss.MeanSquaredError(outputVector, expectedVector);

                (double[][,] weight, double[][] bias) derivatives = network.BackPropagate(expectedVector);
                for (int i = 0; i < network.numberOfLayers - 1; i++) {
                    weightGradients[i] = Matrix.Add(weightGradients[i], derivatives.weight[i]);
                    biasGradients[i] = Matrix.Add(biasGradients[i], derivatives.bias[i]);
                }

                for (int i = 0; i < network.numberOfLayers - 1; i++) {
                    weightGradients[i] = Matrix.ScalarMultiply(weightGradients[i], 1 / batch.Count);
                    biasGradients[i] = Matrix.ScalarMultiply(biasGradients[i], 1 / batch.Count);
                }

            }
            double averageCost = totalCost / batch.Count;
            Console.WriteLine($"{correct}/{batch.Count} correct, cost {averageCost}");
            network.UpdateWeightsAndBiases(weightGradients, biasGradients);
        }

        public static void TestImages() {
            List<Image> images = DigitDataReader.ReadTrainingData();
            foreach (Image image in images) {
                image.Display();
                
                break;
            }
        }
    }
}
using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Feedforward {
    public static class Program {

        public static void TrainDefault() {
            string path = "Feedforward/Saved-Networks/ver1.txt";
            int[] layerSizes = {784, 16, 16, 10};
            // Feedforward network = new Feedforward(layerSizes);
            Feedforward network = new Feedforward(path);
            List<Image> trainingSet = DigitDataReader.ReadTrainingData();
            TrainNetwork(network, trainingSet, 100, 1);
            network.SaveNetwork("Feedforward/Saved-Networks/ver1.txt");

        }

        public static void TrainNetwork(Feedforward network, List<Image> trainingSet, int batchSize, int epochs) {

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

            for (int b = 0; b < batch.Count; b++) { 
            // foreach (Image image in batch) {
                Image image = batch[b];
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

                // Testing by displaying
                // if (b == 1) {
                //     Matrix.Display(outputVector);
                // }

            }
            double averageCost = totalCost / batch.Count;
            Console.WriteLine($"{correct}/{batch.Count} correct, cost {averageCost}");
            network.UpdateWeightsAndBiases(weightGradients, biasGradients);
        }
    }
}
using System;
using System.Collections.Generic;
using System.Linq;

namespace RecurrentNeuralnetwork {
    public class Network {

        int vocabSize;
        int hiddenSize;
        int outputSize;

        double learnRate;

        double[,] weightsU;
        double[,] weightsV;
        double[,] weightsW;
        double[] biasB;
        double[] biasC;

        public Network(int vocabSize, int hiddenSize, int outputSize, double learnRate) {
            this.vocabSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            this.learnRate = learnRate;

            weightsU = Matrix.InitialiseWeights(hiddenSize, vocabSize);
            weightsV = Matrix.InitialiseWeights(outputSize, hiddenSize);
            weightsW = Matrix.InitialiseWeights(hiddenSize, hiddenSize);

            biasB = new double[hiddenSize];
            biasC = new double[outputSize];
        }

        // public double[][] ForwardPropogate(char[] input, double[] previousHiddenState) {
        public (double[][] inputStates, double[][] hiddenStates, double[][] outputStates) ForwardPropogate(char[] input, double[] previousHiddenState) {

            int n = input.Length;
            double[][] inputStates = new double[n][];
            double[][] hiddenStates = new double[n][];
            double[][] outputStates = new double[n][];

            for (int time = 0; time < n; time++) {
                inputStates[time] = new double[vocabSize];
                inputStates[time][input[time] - 'a'] = 1;

                double[] Wh;
                if (time == 0) {
                    Wh = Matrix.MatrixMultiply(weightsW, previousHiddenState);
                } else {
                    Wh = Matrix.MatrixMultiply(weightsW, hiddenStates[time - 1]);
                }
                double[] Ux = Matrix.MatrixMultiply(weightsU, inputStates[time]);
                double[] a = Matrix.Add(biasB, Wh, Ux);
                hiddenStates[time] = Activation.Tanh(a);

                double[] o = Matrix.Add(biasC, Matrix.MatrixMultiply(weightsV, hiddenStates[time]));
                outputStates[time] = Activation.Softmax(o);
            }

            return (inputStates, hiddenStates, outputStates);
        }
    }
}
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

        public double[] ForwardPropogate(double[] input, double[] previousHiddenState) {
            // We can remove the intermediate steps to reduce memory use, at the expense of readability
            double[] Wh = Matrix.MatrixMultiply(weightsW, previousHiddenState);
            double[] Ux = Matrix.MatrixMultiply(weightsU, input);
            double[] a = Matrix.Add(biasB, Wh, Ux);
            double[] h = Activation.Tanh(a);
            double[] o = Matrix.Add(biasC, Matrix.MatrixMultiply(weightsV, h));
            double[] yHat = Activation.Softmax(o);
            return yHat;
        }
    }
}
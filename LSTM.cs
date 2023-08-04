using System;
using System.Collections.Generic;
using System.Linq;

namespace RecurrentNeuralnetwork {

    public class LSTM {
        int inputSize;
        int hiddenSize;
        int outputSize;

        double learnRate = 0.02;

        // Weights and biases
        double[,] weightsForget;
        double[]  biasesForget;

        double[,] weightsInput;
        double[]  biasesInput;

        double[,] weightsCandidate;
        double[]  biasesCandidate;

        double[,] weightsOutput;
        double[]  biasesOutput;

        double[,] weightsFinal;
        double[]  biasesFinal;


        public LSTM(int inputSize, int hiddenSize, int outputSize) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            weightsForget = Matrix.InitialiseWeights(hiddenSize, inputSize);
            biasesForget = new double[hiddenSize];

            weightsInput = Matrix.InitialiseWeights(hiddenSize, inputSize);
            biasesInput = new double[hiddenSize];

            weightsCandidate = Matrix.InitialiseWeights(hiddenSize, inputSize);
            biasesCandidate = new double[hiddenSize];

            weightsOutput = Matrix.InitialiseWeights(hiddenSize, inputSize);
            biasesOutput = new double[hiddenSize];

            weightsFinal = Matrix.InitialiseWeights(outputSize, hiddenSize);
            biasesFinal = new double[outputSize];
        }
    
    }

}
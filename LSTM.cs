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

        int sequenceLength = 0;

        double[][] concatenatedInputs;
        double[][] hiddenStates;
        double[][] cellStates;
        double[][] candidateGates;
        double[][] outputGates;
        double[][] forgetGates;
        double[][] inputGates;
        double[][] outputs;


        public LSTM(int inputSize, int hiddenSize, int outputSize) {
            this.inputSize  = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            weightsForget    = Matrix.InitialiseWeights(hiddenSize, inputSize);
            biasesForget     = new double[hiddenSize];

            weightsInput     = Matrix.InitialiseWeights(hiddenSize, inputSize);
            biasesInput      = new double[hiddenSize];

            weightsCandidate = Matrix.InitialiseWeights(hiddenSize, inputSize);
            biasesCandidate  = new double[hiddenSize];

            weightsOutput    = Matrix.InitialiseWeights(hiddenSize, inputSize);
            biasesOutput     = new double[hiddenSize];

            weightsFinal     = Matrix.InitialiseWeights(outputSize, hiddenSize);
            biasesFinal      = new double[outputSize];

            // so that everything is initialsed as we exit the constructor
            ResetStates(1);
        }

        public void ResetStates(int sequenceLength) {
            concatenatedInputs = new double[sequenceLength][];
            hiddenStates = new double[sequenceLength][];
            cellStates = new double[sequenceLength][];
            candidateGates = new double[sequenceLength][];
            outputGates = new double[sequenceLength][];
            forgetGates = new double[sequenceLength][];
            inputGates = new double[sequenceLength][];
            outputs = new double[sequenceLength][];
        }

        public double[] ForwardPropagate(int[] inputs) {
            sequenceLength = inputs.Length;

            ResetStates(sequenceLength);

            double[] initialHiddenState = new double[hiddenSize];
            double[] initialCellState = new double[hiddenSize];

            for (int t = 0; t < sequenceLength; t++) {
                // create 1 hot vector
                double[] input = new double[inputSize - hiddenSize]; // inputSize - hiddenSize = vocabSize
                input[inputs[t]] = 1;

                double[] prevHiddenState = t == 0 ? initialHiddenState : hiddenStates[t-1];
                double[] prevCellState = t == 0 ? initialCellState : cellStates[t-1];

                concatenatedInputs[t] = Matrix.Concatenate(prevHiddenState, input);

                forgetGates[t]    = Activation.Sigmoid(Matrix.Add(Matrix.MatrixMultiply(weightsForget, concatenatedInputs[t]), biasesForget));
                inputGates[t]     = Activation.Sigmoid(Matrix.Add(Matrix.MatrixMultiply(weightsInput, concatenatedInputs[t]), biasesInput));
                candidateGates[t] = Activation.Tanh(Matrix.Add(Matrix.MatrixMultiply(weightsCandidate, concatenatedInputs[t]), biasesCandidate));
                outputGates[t]    = Activation.Sigmoid(Matrix.Add(Matrix.MatrixMultiply(weightsOutput, concatenatedInputs[t]), biasesOutput));

                cellStates[t] = Matrix.Add(Matrix.MultiplyVectorElementwise(forgetGates[t], prevCellState), Matrix.MultiplyVectorElementwise(inputGates[t], candidateGates[t]));
                hiddenStates[t] = Matrix.MultiplyVectorElementwise(outputGates[t], Activation.Tanh(cellStates[t]));

                outputs[t] = Matrix.Add(Matrix.MatrixMultiply(weightsFinal, hiddenStates[t]), biasesFinal);
            }

            return outputs[^1];
        }
    
    }

}
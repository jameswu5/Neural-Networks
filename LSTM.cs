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

        double[][] inputStates;
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

            weightsForget    = Matrix.InitialiseWeights(hiddenSize, inputSize + hiddenSize);
            biasesForget     = new double[hiddenSize];

            weightsInput     = Matrix.InitialiseWeights(hiddenSize, inputSize + hiddenSize);
            biasesInput      = new double[hiddenSize];

            weightsCandidate = Matrix.InitialiseWeights(hiddenSize, inputSize + hiddenSize);
            biasesCandidate  = new double[hiddenSize];

            weightsOutput    = Matrix.InitialiseWeights(hiddenSize, inputSize + hiddenSize);
            biasesOutput     = new double[hiddenSize];

            weightsFinal     = Matrix.InitialiseWeights(outputSize, hiddenSize);
            biasesFinal      = new double[outputSize];

            // so that everything is initialsed as we exit the constructor
            ResetStates(1);
        }

        public void ResetStates(int sequenceLength) {
            inputStates = new double[sequenceLength][];
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
                double[] input = new double[inputSize];
                input[inputs[t]] = 1;
                // inputStates[t] = input;

                double[] prevHiddenState = t == 0 ? initialHiddenState : hiddenStates[t-1];
                double[] prevCellState = t == 0 ? initialCellState : cellStates[t-1];

                double[] concatenatedInputs = Matrix.Concatenate(prevHiddenState, input);
                inputStates[t] = concatenatedInputs;

                forgetGates[t]    = Activation.Sigmoid(Matrix.Add(Matrix.MatrixMultiply(weightsForget, concatenatedInputs), biasesForget));
                inputGates[t]     = Activation.Sigmoid(Matrix.Add(Matrix.MatrixMultiply(weightsInput, concatenatedInputs), biasesInput));
                candidateGates[t] = Activation.Tanh(Matrix.Add(Matrix.MatrixMultiply(weightsCandidate, concatenatedInputs), biasesCandidate));
                outputGates[t]    = Activation.Sigmoid(Matrix.Add(Matrix.MatrixMultiply(weightsOutput, concatenatedInputs), biasesOutput));

                cellStates[t] = Matrix.Add(Matrix.MultiplyVectorElementwise(forgetGates[t], prevCellState), Matrix.MultiplyVectorElementwise(inputGates[t], candidateGates[t]));
                hiddenStates[t] = Matrix.MultiplyVectorElementwise(outputGates[t], Activation.Tanh(cellStates[t]));

                outputs[t] = Matrix.Add(Matrix.MatrixMultiply(weightsFinal, hiddenStates[t]), biasesFinal);
            }

            return Activation.Softmax(outputs[^1]);
        }

        public void BackPropagate(int target, bool train = true) {

            // Gradients
            double[,] d_weightsForget = new double[hiddenSize, inputSize + hiddenSize];
            double[] d_biasesForget = new double[hiddenSize];
            double[,] d_weightsInput = new double[hiddenSize, inputSize + hiddenSize];
            double[] d_biasesInput = new double[hiddenSize];
            double[,] d_weightsCandidate = new double[hiddenSize, inputSize + hiddenSize];
            double[] d_biasesCandidate = new double[hiddenSize];
            double[,] d_weightsOutput = new double[hiddenSize, inputSize + hiddenSize];
            double[] d_biasesOutput = new double[hiddenSize];
            double[,] d_weightsFinal = new double[outputSize, hiddenSize];
            double[] d_biasesFinal = new double[outputSize];

            // I think these are of hidden layer size
            double[] d_nextHidden = new double[hiddenSize];
            double[] d_nextCell = new double[hiddenSize];

            for (int t = sequenceLength - 1; t >= 0; t--) {

                double[] prevCellState = t == 0 ? new double[hiddenSize] : cellStates[t - 1];

                double[] error = Activation.Softmax(outputs[t]);
                error[target] -= 1;
                // multiply by -1 ?

                // Final gate
                d_weightsFinal = Matrix.Add(d_weightsFinal, Matrix.MatrixMultiply(error, hiddenStates[t]));
                d_biasesFinal = Matrix.Add(d_biasesFinal, error);

                // Hidden state error
                double[] d_hidden = Matrix.MatrixMultiply(Matrix.Transpose(d_weightsFinal), error);
                d_hidden = Matrix.Add(d_hidden, d_nextHidden);

                // Output gate
                double[] outputTemp = Matrix.MultiplyVectorElementwise(cellStates[t], d_hidden);
                outputTemp = Matrix.MultiplyVectorElementwise(outputTemp, Derivative.Sigmoid(outputGates[t]));
                outputTemp = Activation.Tanh(outputTemp);
                d_weightsOutput = Matrix.Add(d_weightsOutput, Matrix.MatrixMultiply(outputTemp, inputStates[t]));
                d_biasesOutput = Matrix.Add(d_biasesOutput, outputTemp);

                // Cell state error
                double[] d_cell = Derivative.Tanh(Activation.Tanh(cellStates[t]));
                d_cell = Matrix.MultiplyVectorElementwise(d_cell, outputGates[t]);
                d_cell = Matrix.MultiplyVectorElementwise(d_cell, d_hidden);
                d_cell = Matrix.Add(d_cell, d_nextCell);

                // Forget gate
                double[] forgetTemp = Matrix.MultiplyVectorElementwise(d_cell, prevCellState);
                forgetTemp = Matrix.MultiplyVectorElementwise(forgetTemp, Derivative.Sigmoid(forgetGates[t]));
                d_weightsForget = Matrix.Add(d_weightsForget, Matrix.MatrixMultiply(forgetTemp, inputStates[t]));
                d_biasesForget = Matrix.Add(d_biasesForget, forgetTemp);

                // Input gate
                double[] inputTemp = Matrix.MultiplyVectorElementwise(d_cell, candidateGates[t]);
                inputTemp = Matrix.MultiplyVectorElementwise(inputTemp, Derivative.Sigmoid(inputGates[t]));
                d_weightsInput = Matrix.Add(d_weightsInput, Matrix.MatrixMultiply(inputTemp, inputStates[t]));
                d_biasesInput = Matrix.Add(d_biasesInput, inputTemp);

                // Candidate gate
                double[] candidateTemp = Matrix.MultiplyVectorElementwise(d_cell, inputGates[t]);
                candidateTemp = Matrix.MultiplyVectorElementwise(candidateTemp, Derivative.Tanh(candidateGates[t]));
                d_weightsCandidate = Matrix.Add(d_weightsCandidate, Matrix.MatrixMultiply(candidateTemp, inputStates[t]));
                d_biasesCandidate = Matrix.Add(d_biasesCandidate, candidateTemp);

                // Cancatenated inputs
                double[] concatTemp = Matrix.MatrixMultiply(Matrix.Transpose(weightsForget), forgetTemp);
                concatTemp = Matrix.Add(concatTemp, Matrix.MatrixMultiply(Matrix.Transpose(weightsInput), inputTemp));
                concatTemp = Matrix.Add(concatTemp, Matrix.MatrixMultiply(Matrix.Transpose(weightsCandidate), candidateTemp));
                concatTemp = Matrix.Add(concatTemp, Matrix.MatrixMultiply(Matrix.Transpose(weightsOutput), outputTemp));

                // Hidden state and cell state
                d_nextHidden = concatTemp[..hiddenSize];
                d_nextCell = Matrix.MultiplyVectorElementwise(forgetGates[t], d_cell);
            }

            // clip the gradients to prevent exploding gradients
            d_weightsCandidate = Matrix.Clip(d_weightsCandidate, -1, 1);
            d_weightsFinal = Matrix.Clip(d_weightsFinal, -1, 1);
            d_weightsForget = Matrix.Clip(d_weightsForget, -1, 1);
            d_weightsInput = Matrix.Clip(d_weightsInput, -1, 1);
            d_weightsOutput = Matrix.Clip(d_weightsOutput, -1, 1);
            d_biasesCandidate = Matrix.Clip(d_biasesCandidate, -1, 1);
            d_biasesFinal = Matrix.Clip(d_biasesFinal, -1, 1);
            d_biasesForget = Matrix.Clip(d_biasesForget, -1, 1);
            d_biasesInput = Matrix.Clip(d_biasesInput, -1, 1);
            d_biasesOutput = Matrix.Clip(d_biasesOutput, -1, 1);

            if (train == true) {
                weightsForget = Matrix.Add(weightsForget, Matrix.ScalarMultiply(d_weightsForget, -learnRate));
                biasesForget = Matrix.Add(biasesForget, Matrix.ScalarMultiply(d_biasesForget, -learnRate));

                weightsInput = Matrix.Add(weightsInput, Matrix.ScalarMultiply(d_weightsInput, -learnRate));
                biasesInput = Matrix.Add(biasesInput, Matrix.ScalarMultiply(d_biasesInput, -learnRate));

                weightsCandidate = Matrix.Add(weightsCandidate, Matrix.ScalarMultiply(d_weightsCandidate, -learnRate));
                biasesCandidate = Matrix.Add(biasesCandidate, Matrix.ScalarMultiply(d_biasesCandidate, -learnRate));

                weightsOutput = Matrix.Add(weightsOutput, Matrix.ScalarMultiply(d_weightsOutput, -learnRate));
                biasesOutput = Matrix.Add(biasesOutput, Matrix.ScalarMultiply(d_biasesOutput, -learnRate));

                weightsFinal = Matrix.Add(weightsFinal, Matrix.ScalarMultiply(d_weightsFinal, -learnRate));
                biasesFinal = Matrix.Add(biasesFinal, Matrix.ScalarMultiply(d_biasesFinal, -learnRate));
            }
        }
    
    
        public void Train(int[] inputs, int label) {
            ForwardPropagate(inputs);
            BackPropagate(label);
        }
    
    
    }

}
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

        int sequenceLength;
        double[] previousHiddenState;

        public Network(int vocabSize, int hiddenSize, int outputSize, double learnRate) {
            this.vocabSize = vocabSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            this.learnRate = learnRate;

            weightsU = Matrix.InitialiseWeights(this.hiddenSize, this.vocabSize);
            weightsV = Matrix.InitialiseWeights(this.outputSize, this.hiddenSize);
            weightsW = Matrix.InitialiseWeights(this.hiddenSize, this.hiddenSize);

            biasB = new double[hiddenSize];
            biasC = new double[outputSize];

            sequenceLength = 0;
            previousHiddenState = new double[hiddenSize];
        }

        // This is for a char based recurrent neural network
        public (double[][] inputStates, double[][] hiddenStates, double[][] outputStates) ForwardPropagate(int[] input) {
            sequenceLength = input.Length;

            double[][] inputStates = new double[sequenceLength][];
            double[][] hiddenStates = new double[sequenceLength][];
            double[][] outputStates = new double[sequenceLength][];

            for (int time = 0; time < sequenceLength; time++) {
                inputStates[time] = new double[vocabSize];
                inputStates[time][input[time]] = 1;

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
    
    
        public (double[,] dU, double[,] dV, double[,] dW, double[] db, double[] dc) BackPropagate(double[][] inputStates, double[][] hiddenStates, double[][] outputStates, int target) {
            double[] dy = new double[outputSize];
            Array.Copy(outputStates[^1], dy, outputSize);
            dy[target]--;
            
            double[,] dU = new double[hiddenSize, vocabSize];
            double[,] dV = new double[outputSize, hiddenSize];
            double[,] dW = new double[hiddenSize, hiddenSize];
            double[] db = new double[hiddenSize];
            double[] dc = new double[outputSize];

            Array.Copy(dy, dc, outputSize); // dc is now done
            dV = Matrix.Add(dV, Matrix.MatrixMultiply(dy, hiddenStates[^1]));

            double[] dh = Matrix.MatrixMultiply(Matrix.Transpose(weightsV), dy);

            for (int t = sequenceLength - 1; t >= 0; t--) {
                double[,] val = Matrix.Add(Matrix.MatrixMultiply(hiddenStates[t], hiddenStates[t]), -1);
                val = Matrix.ScalarMultiply(val, -1);
                double[] temp = Matrix.MatrixMultiply(val, dh);

                db = Matrix.Add(db, temp);
                double[] prevHiddenState = t == 0 ? previousHiddenState : hiddenStates[t - 1];
                dW = Matrix.Add(dW, Matrix.MatrixMultiply(temp, prevHiddenState));
                dU = Matrix.Add(dU, Matrix.MatrixMultiply(temp, inputStates[t]));
                dh = Matrix.MatrixMultiply(weightsW, temp);
            }

            // Clip so that we don't have exploding gradients
            dU = Matrix.Clip(dU, -1, 1);
            dV = Matrix.Clip(dV, -1, 1);
            dW = Matrix.Clip(dW, -1, 1);
            db = Matrix.Clip(db, -1, 1);
            dc = Matrix.Clip(dc, -1, 1);

            return (dU, dV, dW, db, dc);
        }

        public void UpdateWeightsAndBiases(double[,] dU, double[,] dV, double[,] dW, double[] db, double[] dc) {
            weightsU = Matrix.Add(weightsU, Matrix.ScalarMultiply(dU, -learnRate));
            weightsV = Matrix.Add(weightsV, Matrix.ScalarMultiply(dV, -learnRate));
            weightsW = Matrix.Add(weightsW, Matrix.ScalarMultiply(dW, -learnRate));
            biasB = Matrix.Add(biasB, Matrix.ScalarMultiply(db, -learnRate));
            biasC = Matrix.Add(biasC, Matrix.ScalarMultiply(dc, -learnRate));
        }
    }
}
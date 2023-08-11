using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks.Recurrent {
    public class Vanilla {
        int inputSize;
        int hiddenSize;
        int outputSize;

        double learnRate = 0.01;

        double[,] weightsU = null!;
        double[,] weightsV = null!;
        double[,] weightsW = null!;
        double[] biasB = null!;
        double[] biasC = null!;

        int sequenceLength;
        double[] previousHiddenState;

        // creating a new network
        public Vanilla(int inputSize, int hiddenSize, int outputSize) {
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            this.outputSize = outputSize;

            weightsU = Matrix.InitialiseWeights(this.hiddenSize, this.inputSize);
            weightsV = Matrix.InitialiseWeights(this.outputSize, this.hiddenSize);
            weightsW = Matrix.InitialiseWeights(this.hiddenSize, this.hiddenSize);

            biasB = new double[hiddenSize];
            biasC = new double[outputSize];

            previousHiddenState = new double[hiddenSize];
        }

        // importing a network
        public Vanilla(string importFileName) {
            ImportNetwork(importFileName);
            previousHiddenState = new double[hiddenSize];
        }

        public (double[][] inputStates, double[][] hiddenStates, double[][] outputStates) ForwardPropagate(int[] input) {
            sequenceLength = input.Length;

            double[][] inputStates = new double[sequenceLength][];
            double[][] hiddenStates = new double[sequenceLength][];
            double[][] outputStates = new double[sequenceLength][];

            for (int time = 0; time < sequenceLength; time++) {
                // create 1 hot vector
                inputStates[time] = new double[inputSize];
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
            
            double[,] dU = new double[hiddenSize, inputSize];
            double[,] dV = new double[outputSize, hiddenSize];
            double[,] dW = new double[hiddenSize, hiddenSize];
            double[] db = new double[hiddenSize];
            double[] dc = new double[outputSize];

            Array.Copy(dy, dc, outputSize);
            dV = Matrix.Add(dV, Matrix.MatrixMultiply(dy, hiddenStates[^1]));

            double[] dh = Matrix.MatrixMultiply(Matrix.Transpose(weightsV), dy);

            for (int t = sequenceLength - 1; t >= 0; t--) {
                double[] temp = Matrix.MultiplyVectorElementwise(Derivative.Tanh(hiddenStates[t]), dh);
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
    
        public double GetLoss(double[] predictedOutput, int target) {
            double[] expectedOutput = new double[outputSize];
            expectedOutput[target] = 1;

            double cost = Loss.CrossEntropy(predictedOutput, expectedOutput);
            return cost;
        }
    
        public int Predict(int[] input) {
            var states = ForwardPropagate(input);
            double[] prediction = states.outputStates[^1];

            Matrix.Display(prediction);
            double maxProb = 0;
            int maxIndex = 0;
            for (int i = 0; i < prediction.Length; i++) {
                if (prediction[i] > maxProb) {
                    maxProb = prediction[i];
                    maxIndex = i;
                }
            }

            return maxIndex;

        }

        public bool PredictAndCheck(int[] input, int target) {
            var states = ForwardPropagate(input);
            double[] prediction = states.outputStates[^1];
            double maxProb = 0;
            int maxIndex = 0;
            for (int i = 0; i < prediction.Length; i++) {
                if (prediction[i] > maxProb) {
                    maxProb = prediction[i];
                    maxIndex = i;
                }
            }

            return maxIndex == target;
        }

        public bool Check(double[] prediction, int target) {
            double maxProb = 0;
            int maxIndex = 0;
            for (int i = 0; i < prediction.Length; i++) {
                if (prediction[i] > maxProb) {
                    maxProb = prediction[i];
                    maxIndex = i;
                }
            }

            return maxIndex == target;
        }

        public bool Train(int[] input, int label) {
            var f = ForwardPropagate(input);
            // Console.WriteLine(GetLoss(f.outputStates[^1], label));
            var b = BackPropagate(f.inputStates, f.hiddenStates, f.outputStates, label);
            UpdateWeightsAndBiases(b.dU, b.dV, b.dW, b.db, b.dc);

            return Check(f.outputStates[^1], label);
        }
    
        public void SaveNetwork(string filename) {
            using (StreamWriter writer = new StreamWriter(filename)) {
                // write the sizes of input, hidden and output
                writer.WriteLine($"{inputSize} {hiddenSize} {outputSize}");
                
                // write U
                for (int i = 0; i < hiddenSize; i++) {
                    for (int j = 0; j < inputSize; j++) {
                        writer.Write(weightsU[i,j]);
                        writer.Write(" ");
                    }
                    writer.WriteLine();
                }

                // write V
                for (int i = 0; i < outputSize; i++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        writer.Write(weightsV[i,j]);
                        writer.Write(" ");
                    }
                    writer.WriteLine();
                }

                // write W
                for (int i = 0; i < hiddenSize; i++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        writer.Write(weightsW[i,j]);
                        writer.Write(" ");
                    }
                    writer.WriteLine();
                }

                // Write b
                for (int i = 0; i < hiddenSize; i++) {
                    writer.Write(biasB[i]);
                    writer.Write(" ");
                }
                writer.WriteLine();

                // Write c
                for (int i = 0; i < outputSize; i++) {
                    writer.Write(biasC[i]);
                    writer.Write(" ");
                }
            }
        }

        public void ImportNetwork(string filename) {
            string[] networkData = File.ReadAllLines(filename);

            // first line is size data
            string[] sizeData = networkData[0].Split(' ');
            inputSize = int.Parse(sizeData[0]);
            hiddenSize = int.Parse(sizeData[1]);
            outputSize = int.Parse(sizeData[2]);

            weightsU = new double[hiddenSize, inputSize];
            weightsV = new double[outputSize, hiddenSize];
            weightsW = new double[hiddenSize, hiddenSize];

            biasB = new double[hiddenSize];
            biasC = new double[outputSize];


            int pointer = 1;

            for (int i = 0; i < hiddenSize; i++) {
                string[] row = networkData[pointer + i].Split(' ');
                for (int j = 0; j < inputSize; j++) {
                    weightsU[i,j] = double.Parse(row[j]);
                }
            }

            pointer += hiddenSize;

            for (int i = 0; i < outputSize; i++) {
                string[] row = networkData[pointer + i].Split(' ');
                for (int j = 0; j < hiddenSize; j++) {
                    weightsV[i,j] = double.Parse(row[j]);
                }
            }

            pointer += outputSize;

            for (int i = 0; i < hiddenSize; i++) {
                string[] row = networkData[pointer + i].Split(' ');
                for (int j = 0; j < hiddenSize; j++) {
                    weightsW[i,j] = double.Parse(row[j]);
                }
            }

            pointer += hiddenSize;

            string[] bAsString = networkData[pointer].Split(' ');
            for (int i = 0; i < hiddenSize; i++) {
                biasB[i] = double.Parse(bAsString[i]);
            }


            pointer++;

            string[] cAsString = networkData[pointer].Split(' ');
            for (int i = 0; i < outputSize; i++) {
                biasC[i] = double.Parse(cAsString[i]);
            }
        }
    
        public void DisplayWeightsAndBiases() {
            Matrix.Display(weightsU);
            Console.WriteLine();
            Matrix.Display(weightsV);
            Console.WriteLine();
            Matrix.Display(weightsW);
            Console.WriteLine();
            Matrix.Display(biasB);
            Console.WriteLine();
            Matrix.Display(biasC);            
        }

    
    }
}
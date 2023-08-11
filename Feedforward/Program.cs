using System;
using System.Collections.Generic;
using System.Linq;

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
    }
}
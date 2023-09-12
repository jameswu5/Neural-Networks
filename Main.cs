using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetworks.Recurrent;

namespace NeuralNetworks {
    public class Program {
        public static void Main() {
            Classify.TrainLanguages(10, true);
            Classify.TestNetwork();
        }
    }
}
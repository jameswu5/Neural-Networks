using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworks {
    public class Program {
        public static void Main() {
            // Recurrent.Classify.TrainLanguages(10, true);
            // Recurrent.Classify.TestNetwork();
            // new Reinforcement.Game(true);

            // Feedforward.DigitRecognition.TrainDefault();
            // Feedforward.DigitRecognition.TestDefault();
            // Feedforward.DigitRecognition.TrainIndividual();

            Feedforward.DigitRecognition.PlayGame();
        }
    }
}
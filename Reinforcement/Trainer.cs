using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.VisualBasic;

namespace NeuralNetworks.Reinforcement {

    public class Trainer {
        
        public Feedforward.Vanilla model;
        public double learnRate;
        public double gamma;
        
        public Trainer(Feedforward.Vanilla model, double learnRate, double gamma) {
            this.model = model;
            this.learnRate = learnRate;
            this.gamma = gamma;
        }

        public void TrainStep(int[] state, int action, int reward, int[] nextState, bool gameOver) {
            // doublize our values
            double[] state0 = new double[state.Length];
            double[] state1 = new double[nextState.Length];
            for (int i = 0; i < state.Length; i++) {
                state0[i] = state[i];
                state1[i] = nextState[i];
            }
            
            double[] prediction = model.ForwardPropagate(state0);
            double[] target = new double[prediction.Length];
            Array.Copy(prediction, target, prediction.Length);

            double Q = reward;
            if (!gameOver) {
                Q += gamma * model.ForwardPropagate(state1).Max();
            }

            target[action] = Q;

            var gradients = model.BackPropagate(target);
            model.UpdateWeightsAndBiases(gradients.weightDerivatives, gradients.biasDerivatives, learnRate);
        }
    }
}
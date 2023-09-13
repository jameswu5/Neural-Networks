

namespace NeuralNetworks.Reinforcement {

    public class Agent {

        const double LearnRate = 0.001;
        const int BatchSize = 1000;
        const int MaxMemory = 100000;

        public int gamesPlayed;
        double gamma;
        List<SnakeMemoryCell> memory;
        Feedforward.Feedforward model;
        Trainer trainer;

        Random rng;

        public Agent() {
            gamesPlayed = 0;
            gamma = 0.9;
            memory = new List<SnakeMemoryCell>();
            model = new Feedforward.Feedforward(new int[] {11, 256, 3});
            trainer = new Trainer(model, LearnRate, gamma);
            rng = new Random();
        }

        public int[] GetGameState(Snake game) {
            int moveL = game.head - 1;
            int moveR = game.head + 1;
            int moveU = game.head - Snake.Width;
            int moveD = game.head + Snake.Width;

            bool dirIsL = game.direction == Direction.Left;
            bool dirIsR = game.direction == Direction.Right;
            bool dirIsU = game.direction == Direction.Up;
            bool dirIsD = game.direction == Direction.Down;

            int foodX = Snake.GetXCoord(game.food);
            int foodY = Snake.GetYCoord(game.food);
            int headX = Snake.GetXCoord(game.head);
            int headY = Snake.GetYCoord(game.head);

            bool[] state0 = {
                // Danger ahead
                (dirIsL && game.CheckCollision(moveL, Direction.Left )) ||
                (dirIsR && game.CheckCollision(moveR, Direction.Right)) ||
                (dirIsU && game.CheckCollision(moveU, Direction.Up   )) ||
                (dirIsD && game.CheckCollision(moveD, Direction.Down )),

                // Danger to the left
                (dirIsL && game.CheckCollision(moveD, Direction.Left )) ||
                (dirIsR && game.CheckCollision(moveU, Direction.Right)) ||
                (dirIsU && game.CheckCollision(moveL, Direction.Up   )) ||
                (dirIsD && game.CheckCollision(moveR, Direction.Down )),

                // Danger to the right
                (dirIsL && game.CheckCollision(moveU, Direction.Left )) ||
                (dirIsR && game.CheckCollision(moveD, Direction.Right)) ||
                (dirIsU && game.CheckCollision(moveR, Direction.Up   )) ||
                (dirIsD && game.CheckCollision(moveL, Direction.Down )),

                dirIsL, dirIsR, dirIsU, dirIsD,

                // Location of food
                foodX < headX,
                foodX > headX,
                foodY < headY,
                foodY > headY
            };

            int[] state = new int[state0.Length];
            for (int i = 0; i < state0.Length; i++) {
                state[i] = state0[i] ? 1 : 0;
            }

            return state;
        }

        public void Remember(int[] state, int action, int reward, int[] nextState, bool gameOver) {
            memory.Add(new SnakeMemoryCell(state, action, reward, nextState, gameOver));
        }

        public void TrainLongMemory() {

            List<SnakeMemoryCell> sample = new();
            if (memory.Count > BatchSize) {
                sample = memory.OrderBy(x => rng.Next()).Take(BatchSize).ToList();
            } else {
                sample = memory;
            }

            foreach (SnakeMemoryCell smc in sample) {
                TrainShortMemory(smc.state, smc.action, smc.reward, smc.nextState, smc.gameOver);
            }
        }

        public void TrainShortMemory(int[] state, int action, int reward, int[] nextState, bool gameOver) {
            trainer.TrainStep(state, action, reward, nextState, gameOver);
        }

        public int GetAction(int[] state) {
            if (rng.Next(0, 200) < 80 - gamesPlayed) {
                return rng.Next(0, 2);
            }

            double[] inputState = new double[state.Length];
            for (int i = 0; i < state.Length; i++) {
                inputState[i] = state[i];
            }

            double[] prediction = model.ForwardPropagate(inputState);
            
            return Array.IndexOf(prediction, prediction.Max());
        }
    }


    public struct SnakeMemoryCell {
        public int[] state;
        public int action;
        public int reward;
        public int[] nextState;
        public bool gameOver;

        public SnakeMemoryCell(int[] state, int action, int reward, int[] nextState, bool gameOver) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.gameOver = gameOver;
        }
    }
}
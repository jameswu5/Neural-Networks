using System.Runtime;
using Microsoft.VisualBasic;
using Raylib_cs;

namespace NeuralNetworks.Reinforcement {

    public class Game {
        List<int> scores;
        List<double> meanScores;
        int totalScore;
        int record;
        Agent agent;
        Snake snake;

        public Game() {
            scores = new List<int>();
            meanScores = new List<double>();
            totalScore = 0;
            record = 0;
            agent = new Agent();
            snake = new Snake();
            Display(snake);
        }

        public void Train() {
            int[] oldState = agent.GetGameState(snake);
            int move = agent.GetAction(oldState);
            var info = snake.Step(move);
            int[] newState = agent.GetGameState(snake);

            agent.TrainShortMemory(oldState, move, info.reward, newState, info.gameOver);

            agent.Remember(oldState, move, info.reward, newState, info.gameOver);

            if (info.gameOver) {
                snake.Reset();
                agent.gamesPlayed++;
                agent.TrainLongMemory();

                if (info.score > record) {
                    record = info.score;
                    // Save the model
                }

                scores.Add(info.score);
                totalScore += info.score;
                meanScores.Add(totalScore * 1.0 / agent.gamesPlayed);

                Console.WriteLine($"Game {agent.gamesPlayed} | Score: {info.score} | Best: {record}");
            }
        }

        public void Display(Snake snake){ 
            Raylib.InitWindow(Snake.Width * Snake.SquareSize, Snake.Height * Snake.SquareSize, "Snake");
            Raylib.SetTargetFPS(60);

            double lastTime = Raylib.GetTime();
            double interval = 0.05;

            while (!Raylib.WindowShouldClose())
            {
                double currentTime = Raylib.GetTime();
                double elapsedTime = currentTime - lastTime;

                if (elapsedTime >= interval) {
                    Train();
                    lastTime = currentTime;
                }

                Raylib.BeginDrawing();

                snake.DisplayUI();

                Raylib.EndDrawing();
            }

            Raylib.CloseWindow();
        }
    }
}
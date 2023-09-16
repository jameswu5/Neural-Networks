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
        double interval;
        const double trainInterval = 0.04;
        const double playerInterval = 0.2;

        const string path = "Reinforcement/Snake/Networks/ver1.txt";

        public Game(bool isPlayer) {
            scores = new List<int>();
            meanScores = new List<double>();
            totalScore = 0;
            record = 0;
            agent = new Agent();
            snake = new Snake();
            interval = isPlayer ? playerInterval : trainInterval;
            Loop(snake, isPlayer);
        }

        public void Train() {
            int[] oldState = agent.GetGameState(snake);
            int move = agent.GetAction(oldState);
            var info = snake.Step(move, true);
            int[] newState = agent.GetGameState(snake);

            agent.TrainShortMemory(oldState, move, info.reward, newState, info.gameOver);

            agent.Remember(oldState, move, info.reward, newState, info.gameOver);

            if (info.gameOver) {
                snake.Reset();
                agent.gamesPlayed++;
                agent.TrainLongMemory();

                if (info.score >= record) {
                    record = info.score;
                    agent.SaveModel(path);
                }

                scores.Add(info.score);
                totalScore += info.score;
                meanScores.Add(totalScore * 1.0 / agent.gamesPlayed);

                Console.WriteLine($"Game {agent.gamesPlayed} | Score: {info.score} | Best: {record}");
            }
        }

        public void Loop(Snake snake, bool isPlayer){ 
            Raylib.InitWindow(Snake.Width * Snake.SquareSize, Snake.Height * Snake.SquareSize, "Snake");
            Raylib.SetTargetFPS(60);

            double lastTime = Raylib.GetTime();

            Direction buffer = snake.direction;

            while (!Raylib.WindowShouldClose())
            {
                double currentTime = Raylib.GetTime();
                double elapsedTime = currentTime - lastTime;

                if (isPlayer) {
                    if (Raylib.IsKeyDown(KeyboardKey.KEY_LEFT) && snake.direction != Direction.Right) {
                        buffer = Direction.Left;
                    }
                    else if (Raylib.IsKeyDown(KeyboardKey.KEY_RIGHT) && snake.direction != Direction.Left) {
                        buffer = Direction.Right;
                    }
                    else if (Raylib.IsKeyDown(KeyboardKey.KEY_UP) && snake.direction != Direction.Down) {
                        buffer = Direction.Up;
                    }
                    else if (Raylib.IsKeyDown(KeyboardKey.KEY_DOWN) && snake.direction != Direction.Up) {
                        buffer = Direction.Down;
                    }
                }

                if (elapsedTime >= interval) {
                    if (isPlayer) {
                        snake.direction = buffer;
                        snake.Step(0);
                        if (snake.gameOver) {
                            snake.Reset();
                            buffer = Direction.Right;
                        }
                    } else {
                        Train();
                    }
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
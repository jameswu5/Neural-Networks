using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using Raylib_cs;

namespace NeuralNetworks.Reinforcement {

    public enum Direction { Up, Right, Down, Left };

    public class Snake {
        public const int Width = 20;
        public const int Height = 15;
        public const int SquareSize = 30;

        static Color DarkGrey = new Color(30, 30, 30, 255);

        readonly Direction[] Clockwise = { Direction.Up, Direction.Right, Direction.Down, Direction.Left };

        public const int Squares = Width * Height;

        public List<int> snake = null!;
        public int head = 0;
        public Direction direction = default!;
        public int score = 0;

        public int food = 0;
        public int steps = 0;
        public bool gameOver = false;

        private readonly Random rng;

        public Snake() {
            rng = new Random();
            steps = 0;
            Reset();
        }

        public void Reset() {
            snake = new List<int>() {GetSquareID(Width / 2, Height / 2), GetSquareID(Width / 2 - 1, Height / 2), GetSquareID(Width / 2 - 2, Height / 2)};
            head = GetSquareID(Width / 2 , Height / 2);
            direction = Direction.Right;
            score = 0;
            food = CreateFood();
            gameOver = false;
        }

        public int CreateFood() {
            int newPos = rng.Next(0, Squares);
            while (snake.Contains(newPos)) {
                newPos = rng.Next(0, Squares);
            }
            return newPos;
        }


        public bool Move(int action) {
            int index = Array.IndexOf(Clockwise, direction);

            if (action == 1) {
                direction = Clockwise[(index + 3) % 4];
            } else if (action == 2) {
                direction = Clockwise[(index + 1) % 4];
            }

            // Need to check for collisions
            switch (direction) {
                case Direction.Right:
                    if (head % Width == Width - 1) return false;
                    head++;
                    break;
                case Direction.Left:
                    if (head % Width == 0) return false;
                    head--;
                    break;
                case Direction.Up:
                    if (head / Width == 0) return false;
                    head -= Width;
                    break;
                case Direction.Down:
                    if (head / Width == Height - 1) return false;
                    head += Width;
                    break;
                default:
                    break;
            }

            return true;
        }

        // action: 0 -> straight | 1 -> left | 2 -> right
        public (int, bool, int) Step(int action) {
            steps++;
            int reward = 0;

            bool legal = Move(action);
            if (!legal || snake.Contains(head) || steps > 50 * snake.Count) {
                reward = -10;
                gameOver = true;
                return (reward, gameOver, score);
            }

            snake.Insert(0, head);

            if (head == food) {
                score++;
                reward = 10;
                food = CreateFood();
            } else {
                snake.RemoveAt(snake.Count - 1);
            }

            DisplayUI();

            return (reward, gameOver, score);
        }

        private int GetSquareID(int x, int y) => y * Width + x;


        private void DisplayUI() {
            Raylib.ClearBackground(DarkGrey);
            DrawSnake();
            DrawFood();
            Raylib.DrawText($"Score: {score}", 20, 20, 25, Color.WHITE);
        }

        private void DrawSnake() {
            foreach (int coord in snake) {
                int x = coord % Width;
                int y = coord / Width;

                Rectangle rect = new Rectangle(x * SquareSize, y * SquareSize, SquareSize, SquareSize);
                Raylib.DrawRectangleRounded(rect, 0.6f, 6, Color.SKYBLUE);
            }
        }

        private void DrawFood() {
            int x = food % Width;
            int y = food / Width;
            Raylib.DrawCircle(x * SquareSize + SquareSize / 2, y * SquareSize + SquareSize / 2, SquareSize / 2, Color.RED);
        }
    }
}
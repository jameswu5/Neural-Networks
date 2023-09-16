using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using Raylib_cs;

namespace NeuralNetworks.Reinforcement {

    public enum Direction { Up, Right, Down, Left };

    public class Snake {
        public const int Width = 24;
        public const int Height = 15;
        public const int SquareSize = 30;

        static Color DarkBlueGrey = new Color(23, 30, 33, 255);
        static Color Red = new Color(227, 132, 132, 255);
        static Color Green1 = new Color(88, 138, 88, 255);
        static Color Green2 = new Color(139, 196, 139, 255);

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
            Reset();
        }

        public void Reset() {
            snake = new List<int>() {GetSquareID(Width / 2, Height / 2), GetSquareID(Width / 2 - 1, Height / 2), GetSquareID(Width / 2 - 2, Height / 2)};
            head = GetSquareID(Width / 2 , Height / 2);
            direction = Direction.Right;
            score = 0;
            steps = 0;
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

        public bool CheckCollision(int point, Direction dir) {
            if (point < 0) return true;
            if (dir == Direction.Right && GetXCoord(point) == 0) return true;
            if (dir == Direction.Left  && GetXCoord(point) == Width - 1) return true;
            if (dir == Direction.Up    && point < 0) return true;
            if (dir == Direction.Down  && GetYCoord(point) >= Height) return true;

            if (snake.Contains(point)) return true;

            return false;
        }

        public void Move(int action) {
            int index = Array.IndexOf(Clockwise, direction);

            if (action == 1) {
                direction = Clockwise[(index + 3) % 4];
            } else if (action == 2) {
                direction = Clockwise[(index + 1) % 4];
            }

            switch (direction) {
                case Direction.Right:
                    head++;
                    break;
                case Direction.Left:
                    head--;
                    break;
                case Direction.Up:
                    head -= Width;
                    break;
                case Direction.Down:
                    head += Width;
                    break;
                default:
                    break;
            }
        }

        // action: 0 -> straight | 1 -> left | 2 -> right
        public (int reward, bool gameOver, int score) Step(int action, bool trainMode = false) {
            steps++;
            int reward;

            double distanceFromFood = GetDistance(head, food);

            Move(action);

            if (CheckCollision(head, direction) || (trainMode && steps > 50)) {
                reward = -10;
                gameOver = true;
                return (reward, gameOver, score);
            }

            snake.Insert(0, head);

            if (head == food) {
                score++;
                reward = 10;
                food = CreateFood();
                steps = 0;
            } else {
                reward = GetDistance(head, food) < distanceFromFood ? 1 : -1;
                snake.RemoveAt(snake.Count - 1);
            }
            return (reward, gameOver, score);
        }

        private static int GetSquareID(int x, int y) => y * Width + x;
        public static int GetXCoord(int i) => i % Width;
        public static int GetYCoord(int i) => i / Width;

        public static double GetDistance(int a, int b) {
            int x1 = GetXCoord(a), y1 = GetYCoord(a);
            int x2 = GetXCoord(b), y2 = GetYCoord(b);
            return Math.Sqrt(Math.Pow(x1 - x2, 2) + Math.Pow(y1 - y2, 2));
        }

        public void DisplayUI() {
            Raylib.ClearBackground(DarkBlueGrey);
            DrawSnake();
            DrawFood();
            Raylib.DrawText($"Score: {score}", 20, 20, 25, Color.WHITE);
        }

        private void DrawSnake() {
            foreach (int coord in snake) {
                int x = GetXCoord(coord);
                int y = GetYCoord(coord);

                Rectangle rect = new Rectangle(x * SquareSize, y * SquareSize, SquareSize, SquareSize);
                Raylib.DrawRectangleRounded(rect, 0.6f, 6, Green1);

                Rectangle rect2 = new Rectangle(x * SquareSize + 4, y * SquareSize + 4, SquareSize - 8, SquareSize - 8);
                Raylib.DrawRectangleRounded(rect2, 0.6f, 6, Green2);


            }
            // Eyes
            if (direction == Direction.Up || direction == Direction.Left) {
                Raylib.DrawCircle(GetXCoord(head) * SquareSize + SquareSize / 3, GetYCoord(head) * SquareSize + SquareSize / 3, 2, DarkBlueGrey);
            }
            if (direction == Direction.Up || direction == Direction.Right) {
                Raylib.DrawCircle(GetXCoord(head) * SquareSize + 2 * SquareSize / 3, GetYCoord(head) * SquareSize + SquareSize / 3, 2, DarkBlueGrey);
            }
            if (direction == Direction.Down || direction == Direction.Left) {
                Raylib.DrawCircle(GetXCoord(head) * SquareSize + SquareSize / 3, GetYCoord(head) * SquareSize + 2 * SquareSize / 3, 2, DarkBlueGrey);
            }
            if (direction == Direction.Down || direction == Direction.Right) {
                Raylib.DrawCircle(GetXCoord(head) * SquareSize + 2 * SquareSize / 3, GetYCoord(head) * SquareSize + 2 * SquareSize / 3, 2, DarkBlueGrey);
            }

        }

        private void DrawFood() {
            int x = GetXCoord(food);
            int y = GetYCoord(food);
            Raylib.DrawCircle(x * SquareSize + SquareSize / 2, y * SquareSize + SquareSize / 2, SquareSize / 2, Red);
        }
    }
}
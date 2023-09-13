using Raylib_cs;

namespace NeuralNetworks.Reinforcement {

    public class Game {
        Snake snake;

        public Game() {
            snake = new Snake();
            Display();
        }

        public void Display() {
            Raylib.InitWindow(Snake.Width * Snake.SquareSize, Snake.Height * Snake.SquareSize, "Snake");
            Raylib.SetTargetFPS(4);

            while (!Raylib.WindowShouldClose() && !snake.gameOver)
            {
                Raylib.BeginDrawing();

                if (Raylib.IsKeyDown(KeyboardKey.KEY_LEFT)) snake.Step(1);
                else if (Raylib.IsKeyDown(KeyboardKey.KEY_RIGHT)) snake.Step(2);
                else snake.Step(0);


                Raylib.EndDrawing();
            }

            Raylib.CloseWindow();
        }

    }

}
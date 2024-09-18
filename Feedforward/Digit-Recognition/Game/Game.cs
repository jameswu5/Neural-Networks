using System;
using Raylib_cs;
using static Raylib_cs.Raylib;

namespace NeuralNetworks.Feedforward;

public class Game
{
    public const int FrameRate = 240;
    public const int ScreenWidth = 1080;
    public const int ScreenHeight = 720;
    public static readonly Color BackgroundColor = new(40, 40, 40, 255);
    public static readonly Color PastelGreen = new(193, 225, 193, 255);

    public const int HorPadding = 80;
    public const int VerPadding = 80;

    public Vanilla network;
    public Canvas canvas;

    public const Stroke.Type strokeType = Stroke.Type.Solid;
    public Stroke stroke;

    public Game(Vanilla network)
    {
        this.network = network;
        canvas = new Canvas(HorPadding, VerPadding);
        stroke = Stroke.Create(strokeType);
    }

    private void Update()
    {
        HandleInput();
        canvas.Draw();
    }

    private void HandleInput()
    {
        if (IsMouseButtonDown(MouseButton.MOUSE_BUTTON_LEFT))
        {
            stroke.Draw(GetMouseX(), GetMouseY(), canvas);
        }
        else if (IsMouseButtonDown(MouseButton.MOUSE_BUTTON_RIGHT))
        {
            canvas.Modify(GetMouseX(), GetMouseY(), 0);
        }
        else if (IsKeyPressed(KeyboardKey.KEY_SPACE))
        {
            canvas.Clear();
        }
    }

    public void Simulate()
    {
        InitWindow(ScreenWidth, ScreenHeight, "Digit Recognition");
        SetTargetFPS(FrameRate);

        while (!WindowShouldClose())
        {
            BeginDrawing();
            ClearBackground(BackgroundColor);
            Update();
            EndDrawing();
        }

        CloseWindow();
    }
}
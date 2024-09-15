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

    public const int Resolution = 28;
    public const int SideLength = 20;
    public const int CanvasLength = Resolution * SideLength;
    public const int HorPadding = 80;
    public const int VerPadding = 80;

    public int[,] pixels;

    public Vanilla network;

    public Game(Vanilla network)
    {
        this.network = network;
        pixels = new int[Resolution, Resolution];
    }

    private void Update()
    {
        HandleInput();
        DrawCanvas();
        TestCentreOfMass();
    }

    private void HandleInput()
    {
        if (IsMouseButtonDown(MouseButton.MOUSE_BUTTON_LEFT))
        {
            ModifyCanvas(GetMouseX(), GetMouseY(), true);
        }
        else if (IsMouseButtonDown(MouseButton.MOUSE_BUTTON_RIGHT))
        {
            ModifyCanvas(GetMouseX(), GetMouseY(), false);
        }
        else if (IsKeyPressed(KeyboardKey.KEY_SPACE))
        {
            Clear();
        }
    }

    private void DrawCanvas()
    {
        // Draw pixels
        for (int x = 0; x < Resolution; x++)
        {
            for (int y = 0; y < Resolution; y++)
            {
                int value = pixels[x, y];
                DrawRectangle(HorPadding + x * SideLength, VerPadding + y * SideLength, SideLength, SideLength, new Color(value, value, value, 255));
            }
        }

        // Draw outline
        DrawRectangleLines(HorPadding, VerPadding, CanvasLength, CanvasLength, Color.WHITE);
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

    private void SetPixel(int x, int y, int value)
    {
        pixels[x, y] = value;
    }

    private static (int, int)? GetCoordinates(int x, int y)
    {
        int xCoord = (x - HorPadding) / SideLength;
        int yCoord = (y - VerPadding) / SideLength;

        if (xCoord >= 0 && xCoord < Resolution && yCoord >= 0 && yCoord < Resolution)
        {
            return (xCoord, yCoord);
        }

        return null;
    }

    private void ModifyCanvas(int mouseX, int mouseY, bool isDraw)
    {
        (int, int)? coordinates = GetCoordinates(mouseX, mouseY);

        if (coordinates != null)
        {
            (int x, int y) = (coordinates.Value);
            SetPixel(x, y, isDraw ? 255 : 0);
        }
    }

    private void Clear()
    {
        Array.Clear(pixels);
    }


    // Process canvas
    private (int, int)? GetCentreOfMass()
    {
        int totalMass = 0;
        int HorMoment = 0;
        int VerMoment = 0;

        for (int i = 0; i < Resolution; i++)
        {
            for (int j = 0; j < Resolution; j++)
            {
                int mass = pixels[i, j];
                totalMass += mass;
                HorMoment += i * mass;
                VerMoment += j * mass;
            }
        }

        if (totalMass == 0)
        {
            return null;
        }

        return (HorMoment / totalMass, VerMoment / totalMass);
    }

    private void TestCentreOfMass()
    {
        (int x, int y)? centreOfMass = GetCentreOfMass();
        if (centreOfMass != null)
        {
            (int x, int y) = centreOfMass.Value;
            DrawText($"Centre of mass: ({x}, {y})", 20, 20, 20, Color.WHITE);

            // Label the pixel at the centre of mass
            DrawRectangle(HorPadding + x * SideLength, VerPadding + y * SideLength, SideLength, SideLength, PastelGreen);
        }
    }
}
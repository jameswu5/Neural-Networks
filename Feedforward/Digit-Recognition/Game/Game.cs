using System;
using System.Linq;
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

    public const int DefaultResolution = 28;
    public const int DefaultPixelSize = 20;
    public const int HorPadding = 80;
    public const int VerPadding = 80;

    public Vanilla network;
    public Canvas canvas;
    public Canvas inputCanvas; // for testing on the processed image

    public const Stroke.Type strokeType = Stroke.Type.Solid;
    public Stroke stroke;

    public const int FontSize = 35;
    public const int Padding = 10;
    public const int TextHorPadding = 800;
    public const int TextVerPadding = (ScreenHeight - (FontSize + Padding) * 10 - Padding) / 2;

    public (int, double)[] results;

    public Game(Vanilla network)
    {
        this.network = network;
        canvas = new Canvas(DefaultResolution, DefaultPixelSize, HorPadding, VerPadding);
        inputCanvas = new Canvas(DefaultResolution, 10, HorPadding * 2 + canvas.canvasLength, VerPadding);
        stroke = Stroke.Create(strokeType);
        results = new (int, double)[10];
        GetResults();
    }

    private void Update()
    {
        bool modified = HandleInput();
        if (modified)
        {
            GetResults();
        }
        canvas.Draw();
        // inputCanvas.Draw();
        DisplayResults();
    }

    // Returns true if the canvas has been modified
    private bool HandleInput()
    {
        if (IsMouseButtonDown(MouseButton.MOUSE_BUTTON_LEFT))
        {
            stroke.Draw(GetMouseX(), GetMouseY(), canvas);
            return true;
        }
        if (IsMouseButtonDown(MouseButton.MOUSE_BUTTON_RIGHT))
        {
            canvas.Modify(GetMouseX(), GetMouseY(), 0);
            return true;

        }
        if (IsKeyPressed(KeyboardKey.KEY_SPACE))
        {
            canvas.Clear();
            return true;
        }

        return false;
    }

    private void GetResults()
    {
        int[,] processedCanvas = canvas.ProcessCanvas();
        inputCanvas.pixels = processedCanvas;
        int[] flattenedCanvas = Matrix.Flatten(processedCanvas, rowMajor: false);
        double[] inputVector = flattenedCanvas.Select(x => x / 255.0).ToArray();
        double[] outputVector = network.ForwardPropagate(inputVector);

        for (int i = 0; i < 10; i++)
        {
            results[i] = (i, outputVector[i]);
        }

        results = results.OrderByDescending(x => x.Item2).ToArray();
    }

    private void DisplayResults()
    {
        for (int i = 0; i < 10; i++)
        {
            Color color = i == 0 ? PastelGreen : Color.WHITE;

            Raylib.DrawText($"{results[i].Item1}:", TextHorPadding, TextVerPadding + i * (FontSize + Padding), FontSize, color);
            Raylib.DrawText($"{results[i].Item2:0.00}", TextHorPadding + 40, TextVerPadding + i * (FontSize + Padding), FontSize, color);
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

    private void LoadImage(int index)
    {
        List<Image> images = DigitDataReader.ReadTestData();
        double[] data = images[index].dataArray;
        int[] pixels = data.Select(d => (int)d).ToArray();
        foreach(int pixel in pixels)
        canvas.pixels = Matrix.Unflatten(pixels, DefaultResolution, DefaultResolution, rowMajor: false);
    }
}
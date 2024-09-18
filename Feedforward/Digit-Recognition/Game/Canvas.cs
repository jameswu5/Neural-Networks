using System;
using Raylib_cs;
using static Raylib_cs.Raylib;

namespace NeuralNetworks.Feedforward;

public class Canvas
{
    public const int Resolution = 28;
    public const int SideLength = 20;
    public const int CanvasLength = Resolution * SideLength;

    private int horOffset;
    private int verOffset;

    private int[,] pixels;

    public Canvas(int horOffset, int verOffset, int[,]? pixels = null)
    {
        this.horOffset = horOffset;
        this.verOffset = verOffset;
        this.pixels = pixels ?? new int[Resolution, Resolution];
    }

    public void Draw()
    {
        // Draw pixels
        for (int x = 0; x < Resolution; x++)
        {
            for (int y = 0; y < Resolution; y++)
            {
                int value = pixels[x, y];
                DrawRectangle(horOffset + x * SideLength, verOffset + y * SideLength, SideLength, SideLength, new Color(value, value, value, 255));
            }
        }

        // Draw outline
        DrawRectangleLines(horOffset, verOffset, CanvasLength, CanvasLength, Color.WHITE);
    }

    private (int, int)? GetCoordinates(int x, int y)
    {
        int xCoord = (x - horOffset) / SideLength;
        int yCoord = (y - verOffset) / SideLength;

        if (xCoord >= 0 && xCoord < Resolution && yCoord >= 0 && yCoord < Resolution)
        {
            return (xCoord, yCoord);
        }

        return null;
    }

    public void Modify(int mouseX, int mouseY, int value, bool takeMax = false)
    {
        (int, int)? coordinates = GetCoordinates(mouseX, mouseY);

        if (coordinates != null)
        {
            (int x, int y) = (coordinates.Value);
            pixels[x, y] = takeMax ? Math.Max(pixels[x, y], value) : value;
        }
    }

    public void Clear()
    {
        Array.Clear(pixels);
    }

    // Process canvas
    private static (int, int)? GetCentreOfMass(int[,] matrix)
    {
        int totalMass = 0;
        int horMoment = 0;
        int verMoment = 0;

        for (int i = 0; i < Resolution; i++)
        {
            for (int j = 0; j < Resolution; j++)
            {
                int mass = matrix[i, j];
                totalMass += mass;
                horMoment += i * mass;
                verMoment += j * mass;
            }
        }

        if (totalMass == 0)
        {
            return null;
        }

        return (horMoment / totalMass, verMoment / totalMass);
    }

    // private void ProcessCanvas()
    // {
    //     // Remove the zero rows and columns
    //     int[,] resizedPixels = (int[,])pixels.Clone();
    //     int total = resizedPixels.Cast<int>().Sum();
    //     if (total != 0)
    //     {
    //         while (resizedPixels.GetLength(0) > 0 && SumRow(resizedPixels, 0) == 0)
    //         {
    //             resizedPixels = RemoveRow(resizedPixels, 0);
    //         }

    //         while (resizedPixels.GetLength(0) > 0 && SumRow(resizedPixels, resizedPixels.GetLength(0) - 1) == 0)
    //         {
    //             resizedPixels = RemoveRow(resizedPixels, resizedPixels.GetLength(0) - 1);
    //         }

    //         while (resizedPixels.GetLength(1) > 0 && SumColumn(resizedPixels, 0) == 0)
    //         {
    //             resizedPixels = RemoveColumn(resizedPixels, 0);
    //         }

    //         while (resizedPixels.GetLength(1) > 0 && SumColumn(resizedPixels, resizedPixels.GetLength(1) - 1) == 0)
    //         {
    //             resizedPixels = RemoveColumn(resizedPixels, resizedPixels.GetLength(1) - 1);
    //         }
    //     }

    //     // Resize the image to 20 by 20
    //     int rows = resizedPixels.GetLength(0);
    //     int cols = resizedPixels.GetLength(1);
    //     double factor;

    //     if (rows > cols)
    //     {
    //         factor = 20.0 / rows;
    //         rows = 20;
    //         cols = (int)(Math.Round(cols * factor));
    //         CvInvoke.Resize(resizedPixels, resizedPixels, new Size(rows, cols), 0, 0, Inter.Linear);
    //     }
    //     else
    //     {
    //         factor = 20.0 / rows;
    //         cols = 20;
    //         rows = (int)(Math.Round(rows * factor));
    //         CvInvoke.Resize(resizedPixels, resizedPixels, new Size(rows, cols), 0, 0, Inter.Linear);
    //     }

    //     // Shift by centre of mass
    //     (int x, int y)? centreOfMass = GetCentreOfMass();

    //     if (centreOfMass != null)
    //     {
    //         Array.Clear(input);
    //         (int x, int y) = centreOfMass.Value;
    //         int xShift = 15 - x;
    //         int yShift = 15 - y;

    //         // This is not efficient, but it is simple
    //         for (int i = 0; i < rows; i++)
    //         {
    //             for (int j = 0; j < cols; j++)
    //             {
    //                 if (i + xShift >= 0 && i + xShift < 28 && j + yShift >= 0 && j + yShift < 28)
    //                 {
    //                     input[i + xShift, j + yShift] = resizedPixels[i, j];
    //                 }
    //             }
    //         }
    //     }
    // }
}
using System;
using Raylib_cs;
using static Raylib_cs.Raylib;

namespace NeuralNetworks.Feedforward;

public class Canvas
{
    public int resolution = 28;
    public int sideLength = 20;
    public int canvasLength;

    private readonly int horOffset;
    private readonly int verOffset;

    public int[,] pixels;

    public Canvas(int resolution, int sideLength, int horOffset, int verOffset, int[,]? pixels = null)
    {
        this.resolution = resolution;
        this.sideLength = sideLength;
        canvasLength = resolution * sideLength;
        this.horOffset = horOffset;
        this.verOffset = verOffset;
        this.pixels = pixels ?? new int[resolution, resolution];
    }

    public void Draw()
    {
        // Draw pixels
        for (int x = 0; x < resolution; x++)
        {
            for (int y = 0; y < resolution; y++)
            {
                int value = pixels[x, y];
                DrawRectangle(horOffset + x * sideLength, verOffset + y * sideLength, sideLength, sideLength, new Color(value, value, value, 255));
            }
        }

        // Draw outline
        DrawRectangleLines(horOffset, verOffset, canvasLength, canvasLength, Color.WHITE);
    }

    private (int, int)? GetCoordinates(int x, int y)
    {
        int xCoord = (x - horOffset) / sideLength;
        int yCoord = (y - verOffset) / sideLength;

        if (xCoord >= 0 && xCoord < resolution && yCoord >= 0 && yCoord < resolution)
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

        for (int i = 0; i < matrix.GetLength(0); i++)
        {
            for (int j = 0; j < matrix.GetLength(1); j++)
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

    public int[,] ProcessCanvas()
    {
        // Identify (and ignore) the zero rows and columns
        (int top, int bottom, int left, int right) r = FindSmallestRectangle(pixels);

        // Extract the subarray
        int subRows = r.bottom - r.top + 1;
        int subCols = r.right - r.left + 1;
        int[,] subarray = new int[subRows, subCols];
        for (int i = 0; i < subRows; i++)
        {
            for (int j = 0; j < subCols; j++)
            {
                subarray[i, j] = pixels[i + r.top, j + r.left];
            }
        }

        // Resize the subarray to 20 by 20
        int[,] resized = ResizeImage(subarray, 20, 20);

        // Shift by centre of mass
        (int x, int y)? centreOfMass = GetCentreOfMass(resized);
        int[,] result = new int[28, 28];

        if (centreOfMass != null)
        {
            Array.Clear(result);
            (int x, int y) = centreOfMass.Value;
            int xShift = 13 - x;
            int yShift = 13 - y;

            // This is not efficient, but it is simple
            for (int i = 0; i < 20; i++)
            {
                for (int j = 0; j < 20; j++)
                {
                    if (i + xShift >= 0 && i + xShift < 28 && j + yShift >= 0 && j + yShift < 28)
                    {
                        result[i + xShift, j + yShift] = resized[i, j];
                    }
                }
            }
        }

        return result;
    }

    // The coordinates given are inclusive
    private static (int, int, int, int) FindSmallestRectangle(int[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);

        int top = rows;
        int bottom = -1;
        int left = cols;
        int right = -1;

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                if (matrix[i, j] != 0)
                {
                    top = Math.Min(top, i);
                    bottom = Math.Max(bottom, i);
                    left = Math.Min(left, j);
                    right = Math.Max(right, j);
                }
            }
        }

        // If no non-zero values found, return original rectangle
        if (bottom == -1 || right == -1)
        {
            return (0, rows - 1, 0, cols - 1);
        }

        return (top, bottom, left, right);
    }

    private static int[,] ResizeImage(int[,] original, int newRows, int newCols)
    {
        int rows = original.GetLength(0);
        int cols = original.GetLength(1);
        
        // Create a new array to hold the resized image
        int[,] resized = new int[newRows, newCols];

        // Calculate the scaling factors
        float heightScale = (float)(rows - 1) / (newRows - 1);
        float widthScale = (float)(cols - 1) / (newCols - 1);

        for (int i = 0; i < newRows; i++)
        {
            for (int j = 0; j < newCols; j++)
            {
                // Calculate the position in the original image
                float originalY = i * heightScale;
                float originalX = j * widthScale;

                // Get the surrounding pixel indices
                int y1 = (int)originalY;
                int y2 = Math.Min(y1 + 1, rows - 1);
                int x1 = (int)originalX;
                int x2 = Math.Min(x1 + 1, cols - 1);

                // Interpolation weights
                float yLerp = originalY - y1;
                float xLerp = originalX - x1;

                // Interpolate in x direction for both y1 and y2
                float top = original[y1, x1] * (1 - xLerp) + original[y1, x2] * xLerp;
                float bottom = original[y2, x1] * (1 - xLerp) + original[y2, x2] * xLerp;

                // Interpolate in the y direction
                resized[i, j] = (int)(top * (1 - yLerp) + bottom * yLerp);
            }
        }

        return resized;
    }
}
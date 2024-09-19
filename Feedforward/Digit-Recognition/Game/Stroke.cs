using System;

namespace NeuralNetworks.Feedforward;

public abstract class Stroke
{
    public enum Type { Solid, Brush }

    public static Stroke Create(Type type)
    {
        return type switch
        {
            Type.Solid => new SolidStroke(),
            Type.Brush => new BrushStroke(),
            _ => throw new ArgumentException("Invalid stroke type")
        };
    }

    public abstract void Draw(int mouseX, int mouseY, Canvas canvas);
}

public class SolidStroke : Stroke
{
    public override void Draw(int mouseX, int mouseY, Canvas canvas)
    {
        canvas.Modify(mouseX, mouseY, 255);
    }
}

public class BrushStroke : Stroke
{
    public override void Draw(int mouseX, int mouseY, Canvas canvas)
    {
        // Draw the main pixel
        canvas.Modify(mouseX, mouseY, 255);
        
        // Draw surrounding pixels at half brightness
        canvas.Modify(mouseX + canvas.sideLength, mouseY, 128, true);
        canvas.Modify(mouseX - canvas.sideLength, mouseY, 128, true);
        canvas.Modify(mouseX, mouseY + canvas.sideLength, 128, true);
        canvas.Modify(mouseX, mouseY - canvas.sideLength, 128, true);
    }
}
namespace NeuralNetworkLibrary.Structure.Components.Interfaces
{
    public interface IFunction
    {
        double Compute(double x);
        double ComputeFirstDerivative(double x);
    }
}

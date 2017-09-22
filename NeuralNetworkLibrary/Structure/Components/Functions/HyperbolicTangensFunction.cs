using System;
using NeuralNetworkLibrary.Structure.Components.Interfaces;


namespace NeuralNetworkLibrary.Structure.Components.Functions
{
    internal class HyperbolicTangensFunction : IFunction
    {
        private double alpha = 1;


        internal HyperbolicTangensFunction() { }
        internal HyperbolicTangensFunction(double Alpha)
        {
            alpha = Alpha;
        }


        public double Compute(double x)
        {
            return (Math.Tanh(alpha * x));
        }
        public double ComputeFirstDerivative(double x)
        {
            double t = Compute(x);
            return alpha * (1 - t * t);
        }
    }
}

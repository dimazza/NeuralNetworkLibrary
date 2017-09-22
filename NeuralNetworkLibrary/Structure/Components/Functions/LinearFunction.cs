using System;
using NeuralNetworkLibrary.Structure.Components.Interfaces;

namespace NeuralNetworkLibrary.Structure.Components.Functions
{
    internal class LinearFunction : IFunction
    {
        private double alpha = 1;


        internal LinearFunction() { }
        internal LinearFunction(double Alpha)
        {
            alpha = Alpha;
        }


        public double Compute(double x)
        {
            double result = 0;
            if (x > alpha)
                result = 1;
            else
            {
                if (x < 0)
                    result = 0;
                else result = x;
            }
            return result;
        }

        public double ComputeFirstDerivative(double x)
        {
            double result = alpha;
            if (x < 0 || x > 1)
                result = 0;
            return result;
        }
    }
}

using System;
using NeuralNetworkLibrary.Structure.Components.Interfaces;

namespace NeuralNetworkLibrary.Structure.Components.Functions
{
    internal class SigmoidFunction : IFunction
    {
        private double alpha = 1;
        

        internal SigmoidFunction() { }
        internal SigmoidFunction(double Alpha)
        {
            alpha = Alpha;
        }


        public double Compute(double x)
        {
            double r = (1 / (1 + Math.Exp(-alpha * x)));
            return r;
        }

        public double ComputeFirstDerivative(double x)
        {
            double temp = Compute(x);
            return alpha * temp * (1 - temp);
        }
    }
}

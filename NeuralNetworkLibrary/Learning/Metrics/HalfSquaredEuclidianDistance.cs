using System;
using NeuralNetworkLibrary.Learning.Interfaces;

namespace NeuralNetworkLibrary.Learning.Metrics
{
    internal class HalfSquaredEuclidianDistance : IMetrics<double>
    {
        public double Calculate(double[] a1, double[] a2)
        {
            double d = 0;
            for (int i = 0; i < a1.Length; i++)
            {
                
                d += Math.Pow(a1[i] - a2[i], 2);
            }
            return 0.5 * d;
        }

        public double CalculatePartialDerivativeByA2Index(double[] a1, double[] a2, int index)
        {
            return a1[index] - a2[index];
        }
    }
}

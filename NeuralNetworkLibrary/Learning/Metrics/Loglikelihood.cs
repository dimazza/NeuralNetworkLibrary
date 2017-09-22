using System;
using NeuralNetworkLibrary.Learning.Interfaces;

namespace NeuralNetworkLibrary.Learning.Metrics
{
    internal class Loglikelihood : IMetrics<double>
    {
        public double Calculate(double[] a1, double[] a2)
        {
            double d = 0;
            for (int i = 0; i < a1.Length; i++)
            {
                d += a1[i] * Math.Log(a2[i]) + (1 - a1[i]) * Math.Log(1 - a2[i]);
            }
            return -d;
        }

        public double CalculatePartialDerivativeByA2Index(double[] a1, double[] a2, int index)
        {
            return -(a1[index] / a2[index] - (1 - a1[index]) / (1 - a2[index]));
        }
    }
}

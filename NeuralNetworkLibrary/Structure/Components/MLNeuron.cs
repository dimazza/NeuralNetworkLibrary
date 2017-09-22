using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Structure.Components.Interfaces;

namespace NeuralNetworkLibrary.Structure.Components
{
    public class MLNeuron : Neuron
    {
        public MLNeuron(List<double> Weights) : base()
        {
            weights = Weights;
        }
        public MLNeuron(int Dimension, Random rnd)
        {
            //rnd = random;
            weights = new List<double>();
            for (int k = 0; k < Dimension; k++)
                weights.Add((rnd.Next(1, 3) == 1 ? -1 : 1) * rnd.Next(100) / 200f);
        }
    }
}

using System;
using NeuralNetworkLibrary.Learning.Interfaces;
using NeuralNetworkLibrary.Structure.Components.Interfaces;
using NeuralNetworkLibrary.Structure.Networks.Interfaces;
using System.Collections.Generic;

namespace NeuralNetworkLibrary.Learning.GeneticOperators
{
    internal class Mutation : IMutation
    {
        List<double> mutationsRange;
        Random _rnd;
        int maxMutations;
        internal Mutation(Random rnd, List<double> MutationsRange, int MaxMutations)
        {
            _rnd = rnd;
            mutationsRange = MutationsRange;
            maxMutations = MaxMutations;
        }
        public void mutate(IMultilayerNeuralNetwork network)
        {
            int mutations = _rnd.Next(1, maxMutations);
            for (int i = 0; i < mutations; i++)
            {
                double rate = _rnd.NextDouble();
                if (rate < 0.2)
                {
                    int layer = _rnd.Next(0, network.Layers.Length);
                    int neuron = _rnd.Next(0, network.Layers[layer].Neurons.Length);
                    int weigth = _rnd.Next(0, network.Layers[layer].Neurons[neuron].Weights.Count);
                    network.Layers[layer].Neurons[neuron].Weights[weigth] = mutationsRange[_rnd.Next(0, mutationsRange.Count)];
                }
            }
        }
    }
}

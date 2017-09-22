using System;
using NeuralNetworkLibrary.Learning.Interfaces;
using NeuralNetworkLibrary.Structure.Components.Interfaces;
using NeuralNetworkLibrary.Structure.Networks.Interfaces;
using NeuralNetworkLibrary.Structure.Networks;
using System.Collections.Generic;

namespace NeuralNetworkLibrary.Learning.GeneticOperators
{
    public class Crossingover
    {
        public Crossingover()
        {

        }
        public MultiLayerNetwork[] copulation(IMultilayerNeuralNetwork N1, IMultilayerNeuralNetwork N2, int k)
        {
            MultiLayerNetwork[]  childs = new MultiLayerNetwork[2];
            int[] neurons = new int[N1.Layers.Length];
            for (int j = 0; j < N1.Layers.Length; j++)
            {
                neurons[j] = N1.Layers[j].Neurons.Length;
            }
            for (int j = 0; j < childs.Length; j++)
            {
                childs[j] = new MultiLayerNetwork(neurons, N1.InputDimension);
            }

            int i = 0;

            for (int layer = 0; layer < N1.Layers.Length; layer++)
            {
                for (int j = 0; j < N1.Layers[layer].Neurons.Length; j++)
                {
                    for (int n = 0; n < N1.Layers[layer].Neurons[j].Weights.Count; n++, i++)
                    {
                        if (i < k)
                        {
                            double a = N1.Layers[layer].Neurons[j].Weights[n], b = N2.Layers[layer].Neurons[j].Weights[n];
                            childs[0].Layers[layer].Neurons[j].Weights[n] = a;
                            childs[1].Layers[layer].Neurons[j].Weights[n] = b;
                        }

                        else
                        {
                            double a = N1.Layers[layer].Neurons[j].Weights[n], b = N2.Layers[layer].Neurons[j].Weights[n];
                            childs[0].Layers[layer].Neurons[j].Weights[n] = b;
                            childs[1].Layers[layer].Neurons[j].Weights[n] = a;
                        }
                    }
                }
            }
            return childs;
        }
        static void swap(ref double a,ref double b)
        {
            double c = a;
            a = b;
            b = c;
        }
    }
}

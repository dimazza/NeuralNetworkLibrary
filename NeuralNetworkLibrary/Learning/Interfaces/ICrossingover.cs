using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetworkLibrary.Structure.Networks.Interfaces;

namespace NeuralNetworkLibrary.Learning.Interfaces
{
    interface ICrossingover
    {
        void copulation(IMultilayerNeuralNetwork N1, IMultilayerNeuralNetwork N2, int k, out IMultilayerNeuralNetwork[] childs);
    }
}

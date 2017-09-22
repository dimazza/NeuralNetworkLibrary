using NeuralNetworkLibrary.Structure.Components.Interfaces;
using System.Collections.Generic;
using NeuralNetworkLibrary.Data;
namespace NeuralNetworkLibrary.Structure.Networks.Interfaces
{
    public interface IMultilayerNeuralNetwork : INeuralNetwork<double,DoubleData>
    {
        /// <summary>
        /// Get array of layers of network
        /// </summary>
        ILayer<double>[] Layers { get; }
         int InputDimension { get; }
         int ConectionsCount { get; }
    }
}

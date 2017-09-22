using NeuralNetworkLibrary.Structure.Networks.Interfaces;

namespace NeuralNetworkLibrary.Learning.Interfaces
{
    interface IMutation
    {
        void mutate(IMultilayerNeuralNetwork network);
    }
}

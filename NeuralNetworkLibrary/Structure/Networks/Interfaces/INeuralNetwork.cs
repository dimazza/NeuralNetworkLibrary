using System.Collections.Generic;
using NeuralNetworkLibrary.Data;
using System.IO;

namespace NeuralNetworkLibrary.Structure.Networks.Interfaces
{
    public interface INeuralNetwork<T1,T2>
    {
        /// <summary>
        /// Compute output vector by input vector
        /// </summary>
        /// <param name="inputVector">Input vector (double[])</param>
        /// <returns>Output vector (double[])</returns>
        T1[] ComputeOutput(double[] inputVector);

        void Save(string path);
        void Load(Stream S);
        /// <summary>
        /// Train network with given inputs and outputs
        /// </summary>
        /// <param name="inputs">Set of input vectors</param>
        /// <param name="outputs">Set if output vectors</param>
        void Train(IList<T2> data);
    }
}

using System.Collections.Generic;
using NeuralNetworkLibrary.Data;

namespace NeuralNetworkLibrary.Learning.Interfaces
{
    public interface ILearningStrategy<T1>
    {
        /// <summary>
        /// Train neural network
        /// </summary>
        /// <param name="data">Set of input, output vectors</param>
        void train(IList<DoubleData> data);
        void train(IList<DoubleData> train, IList<DoubleData> validate, IList<DoubleData> check);

    }
}

namespace NeuralNetworkLibrary.Structure.Components.Interfaces
{
    public interface ILayer<T>
    {
        /// <summary>
        /// Get last output of the layer
        /// </summary>
        T[] LastOutput { get; }

        /// <summary>
        /// Get neurons of the layer
        /// </summary>
        INeuron<T>[] Neurons { get; }

        /// <summary>
        /// Get input dimension of neurons
        /// </summary>
        int InputDimension { get; }


        /// <summary>
        /// Compute output of the layer
        /// </summary>
        /// <param name="inputVector">Input vector</param>
        /// <returns>Output vector</returns>
        T[] Compute(double[] inputVector);
    }
}

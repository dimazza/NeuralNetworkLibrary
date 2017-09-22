using System.Collections.Generic;
using NeuralNetworkLibrary.Structure.Components.Interfaces;

namespace NeuralNetworkLibrary.Structure.Components.Interfaces
{
    public interface INeuron<T>
    {
        /// <summary>
        /// Weights of the neuron
        /// </summary>
        List<double> Weights { get; }

        /// <summary>
        /// Offset/bias of neuron (default is 0)
        /// </summary>
        double Bias { get; set; }
    
        /// <summary>
        /// Last calculated state in Activate
        /// </summary>
        T LastState { get; set; }

        /// <summary>
        /// Last calculated NET in NET
        /// </summary>
        T WeightedSum { get; set; }

        IList<INeuron<T>> Children { get; }

        IList<INeuron<T>> Parents { get; }

        IFunction ActivationFunction { get; set; }

        double ActivationFunctionDerivative { get; }

        double ErrorDerivative { get; set; }

        /// <summary>
        /// Compute NET of the neuron by input vector
        /// </summary>
        /// <param name="inputVector">input vector (must be the same dimension as was set in SetDimension)</param>
        /// <returns>NET of neuron</returns>
        T computeWeightedSum(T[] inputVector);
        /// <summary>
        /// Compute state of neuron
        /// </summary>
        /// <param name="inputVector">input vector (must be the same dimension as was set in SetDimension)</param>
        /// <returns>State of neuron</returns>
        T Activate(T[] inputVector);
    }
}

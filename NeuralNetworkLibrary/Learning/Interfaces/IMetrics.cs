namespace NeuralNetworkLibrary.Learning.Interfaces
{
    public interface IMetrics<T>
    {
        T Calculate(T[] a1, T[] a2);

        /// <summary>
        /// Calculate value of partial derivative by a2[index]
        /// </summary>
        T CalculatePartialDerivativeByA2Index(T[] a1, T[] a2, int index);
    }
}

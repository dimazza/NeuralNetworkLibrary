namespace NeuralNetworkLibrary.Data.Interfaces
{
    public interface INormalize<T>
    {
        void Normalize(T coef);
        void DeNormalize(T coef);
    }
}

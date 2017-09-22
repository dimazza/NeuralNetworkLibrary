namespace NeuralNetworkLibrary.Data.Interfaces
{
    public interface ITrainingDataItem<T>
    {
        T[] Input
        { get; set; }

        T[] Output
        { get; set; }
    }
}

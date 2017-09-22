using NeuralNetworkLibrary.Data.Interfaces;

namespace NeuralNetworkLibrary.Data
{
    abstract public class DataItem<T1,T2>:ITrainingDataItem<T1>,INormalize<T2>
    {
        protected T1[] input = null;
        protected T1[] output = null;

        public DataItem() { }

        public DataItem(T1[] Input, T1[] Output)
        {
            input = Input;
            output = Output;
        }

        public T1[] Input
        {
            get { return input; }
            set { input = value; }
        }

        public T1[] Output
        {
            get { return output; }
            set { output = value; }
        }
        public abstract void Normalize(T2 coef);
        public abstract void DeNormalize(T2 coef);
    }
}

using NeuralNetworkLibrary.Learning.Interfaces;
using NeuralNetworkLibrary.Learning.Metrics;

namespace NeuralNetworkLibrary.Learning
{
    public class LearningConfig
    {
        private double learningRate = 10;
        private double learningChange=0.9;
        private int batchSize = -1;
        private double regularizationFactor = 0.1;
        private int maxEpoches = 10000;
        private double minError = 0.001;
        private double minErrorChange = 0.00001;
        private IMetrics<double> errorFunction = new HalfSquaredEuclidianDistance();


        public LearningConfig() { }
        public LearningConfig(double LearningRate, int MaxEpoches)
        {
            learningRate = LearningRate;
            maxEpoches = MaxEpoches;
        }
        public LearningConfig(double LearningRate, int MaxEpoches, double MaxError, double MinErrorChange)
        {
            learningRate = LearningRate;
            maxEpoches = MaxEpoches;
            minError = MaxError;
            minErrorChange = MinErrorChange;
        }
        public LearningConfig(double LearningRate, int MaxEpoches, double MaxError, double MinErrorChange, IMetrics<double> ErrorFunction)
        {
            this.learningRate = LearningRate;
            maxEpoches = MaxEpoches;
            minError = MaxError;
            minErrorChange = MinErrorChange;
            errorFunction = ErrorFunction;
        }


        /// <summary>
        /// Learning rate means coefficient of gradient descent
        /// </summary>
        public double LearningRate { get { return learningRate; } set { learningRate = value; } }

        // <summary>
        /// Learning rate means coefficient of gradient descent
        /// </summary>
        public double LearningChange { get { return learningChange; } set { learningChange = value; } }

        /// <summary>
        /// Size of the batch. -1 means fullbatch size. 
        /// </summary>
        public int BatchSize { get { return batchSize; } set { batchSize = value; } }

        public double RegularizationFactor { get { return regularizationFactor; } set { regularizationFactor = value; } }

        public int MaxEpoches { get { return maxEpoches; } set { maxEpoches = value; } }

        /// <summary>
        /// If acumulative error for all training examples is less then MinError, then algorithm stops 
        /// </summary>
        public double MinError { get { return minError; } set { minError = value; } }

        /// <summary>
        /// If acumulative error change for all training examples is less then MinErrorChange, then algorithm stops 
        /// </summary>
        public double MinErrorChange { get { return minErrorChange; } set { minErrorChange = value; } }

        /// <summary>
        /// Function to minimize
        /// </summary>
        public IMetrics<double> ErrorFunction { get { return errorFunction; } set { errorFunction = value; } }

    }
}

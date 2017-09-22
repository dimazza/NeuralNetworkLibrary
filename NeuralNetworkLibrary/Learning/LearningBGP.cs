using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Learning.Interfaces;
using NeuralNetworkLibrary.Structure.Networks.Interfaces;
using NeuralNetworkLibrary.Data;
using NLog;

namespace NeuralNetworkLibrary.Learning
{
    internal class LearningBGP : ILearningStrategy<IMultilayerNeuralNetwork>
    {

        private LearningConfig _config = null;
        private Random _random = null;
        // private static Logger logger;
        IMultilayerNeuralNetwork network;
        private double currentValidateError = Single.MaxValue;
        private double lastValidateError1 = 0;
        private double lastValidateError2 = 0;
        private double currentCheckError = 0;
        internal LearningBGP(IMultilayerNeuralNetwork NetWork, LearningConfig config)
        {
            _config = config;
            //_random = new Random(Helper.GetSeed());
            network = NetWork;
            _random = new Random(1);
        }


        public void train(IList<DoubleData> train, IList<DoubleData> validate, IList<DoubleData> check)
        {
            if (_config.BatchSize < 1 || _config.BatchSize > train.Count)
            {
                _config.BatchSize = train.Count;
            }
            double currentError = Single.MaxValue;
            double lastError = 0;
            int epochNumber = 0;
            //Logger.Instance.Log("Start learning...");

            string Train = @"E:\Дима\NeuralNetworks\train.txt";
            /*string Validate = @"E:\Дима\NeuralNetworks\validate.txt";
            string Check = @"E:\Дима\NeuralNetworks\check.txt";*/
            System.IO.StreamWriter srT = new System.IO.StreamWriter(Train, false);
            /*System.IO.StreamWriter srV = new System.IO.StreamWriter(Validate, false);
            System.IO.StreamWriter srC = new System.IO.StreamWriter(Check, false);*/
            do
            {
                lastError = currentError;
                DateTime dtStart = DateTime.Now;

                #region one epoche

                //preparation for epoche
                int[] trainingIndices = new int[train.Count];
                for (int i = 0; i < train.Count; i++)
                {
                    trainingIndices[i] = i;
                }
                if (_config.BatchSize > 0)
                {
                    trainingIndices = Shuffle(trainingIndices);
                }


                //process data set
                int currentIndex = 0;
                do
                {
                    //accumulated error for batch, for weights and biases
                    double[][][] nablaWeights = new double[network.Layers.Length][][];
                    double[][] nablaBiases = new double[network.Layers.Length][];
                    nablaWeights[network.Layers.Length - 1] = new double[network.Layers[network.Layers.Length - 1].Neurons.Length][];
                    nablaBiases[network.Layers.Length - 1] = new double[network.Layers[network.Layers.Length - 1].Neurons.Length];
                    for (int j = 0; j < network.Layers[network.Layers.Length - 1].Neurons.Length; j++)
                        nablaWeights[network.Layers.Length - 1][j] = new double[network.Layers[network.Layers.Length - 1].Neurons[j].Weights.Count];
                    for (int hiddenLayerIndex = network.Layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
                    {
                        nablaWeights[hiddenLayerIndex] = new double[network.Layers[hiddenLayerIndex].Neurons.Length][];
                        nablaBiases[hiddenLayerIndex] = new double[network.Layers[hiddenLayerIndex].Neurons.Length];
                        for (int j = 0; j < network.Layers[hiddenLayerIndex].Neurons.Length; j++)
                            nablaWeights[hiddenLayerIndex][j] = new double[network.Layers[hiddenLayerIndex].Neurons[j].Weights.Count];
                    }

                    //process one batch
                    for (int inBatchIndex = currentIndex; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < train.Count; inBatchIndex++)
                    {
                        Iteration(network, train[trainingIndices[inBatchIndex]], nablaWeights, nablaBiases);
                    }

                    //update weights and bias
                    UpdateNetwork(network, nablaWeights, nablaBiases);

                    currentIndex += _config.BatchSize;
                } while (currentIndex < train.Count);

                //recalculating error on training data
                currentError = 0;
                for (int i = 0; i < train.Count; i++)
                {
                    double[] realOutput = network.ComputeOutput(train[i].Input);
                    currentError += _config.ErrorFunction.Calculate(train[i].Output, realOutput);
                }
                currentError *= 1d / train.Count;

                lastValidateError2 = lastValidateError1;
                lastValidateError1 = currentValidateError;
                //recalculating error on all validate
                currentValidateError = 0;
                for (int i = 0; i < validate.Count; i++)
                {
                    double[] realOutput = network.ComputeOutput(validate[i].Input);
                    currentValidateError += _config.ErrorFunction.Calculate(validate[i].Output, realOutput);
                }
                currentValidateError /= validate.Count;


                currentCheckError = 0;
                for (int i = 0; i < check.Count; i++)
                {
                    double[] realOutput = network.ComputeOutput(check[i].Input);
                    currentCheckError += _config.ErrorFunction.Calculate(check[i].Output, realOutput);
                }
                currentCheckError /= check.Count;
                #endregion

                epochNumber++;
                if (Math.Abs(currentError - lastError) < _config.MinErrorChange)
                    _config.LearningRate /= 2;
                Console.WriteLine("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());
                string ss = epochNumber.ToString() + "," + currentError.ToString().Replace(",", ".") + "," + currentValidateError.ToString().Replace(",", ".") + "," + currentCheckError.ToString().Replace(",", ".");
                srT.WriteLine(ss);
                /*srV.WriteLine(epochNumber.ToString() + "," + currentValidateError.ToString().Replace(",", "."));
                srC.WriteLine(epochNumber.ToString() + "," + currentCheckError.ToString().Replace(",", "."));*/
                /*Logger.Instance.Log("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());*/
            } while //(epochNumber < _config.MaxEpoches && currentError > _config.MinError && (currentValidateError <= lastValidateError1 || currentValidateError <= lastValidateError2));
             (epochNumber < _config.MaxEpoches && currentError > _config.MinError);
            srT.Close();
            /*srV.Close();
            srC.Close();*/
        }

        public void train(IList<DoubleData> data)
        {
            if (_config.BatchSize < 1 || _config.BatchSize > data.Count)
            {
                _config.BatchSize = data.Count;
            }
            double currentError = Single.MaxValue;
            double lastError = 0;
            int epochNumber = 0;
            //Logger.Instance.Log("Start learning...");

            string Train = @"E:\Дима\NeuralNetworks\train.txt";
            System.IO.StreamWriter sr = new System.IO.StreamWriter(Train, false);
            do
            {
                lastError = currentError;
                DateTime dtStart = DateTime.Now;

                #region one epoche

                //preparation for epoche
                int[] trainingIndices = new int[data.Count];
                for (int i = 0; i < data.Count; i++)
                {
                    trainingIndices[i] = i;
                }
                if (_config.BatchSize > 0)
                {
                    trainingIndices = Shuffle(trainingIndices);
                }


                //process data set
                int currentIndex = 0;
                do
                {
                    //accumulated error for batch, for weights and biases
                    double[][][] nablaWeights = new double[network.Layers.Length][][];
                    double[][] nablaBiases = new double[network.Layers.Length][];
                    nablaWeights[network.Layers.Length - 1] = new double[network.Layers[network.Layers.Length - 1].Neurons.Length][];
                    nablaBiases[network.Layers.Length - 1] = new double[network.Layers[network.Layers.Length - 1].Neurons.Length];
                    for (int j = 0; j < network.Layers[network.Layers.Length - 1].Neurons.Length; j++)
                        nablaWeights[network.Layers.Length - 1][j] = new double[network.Layers[network.Layers.Length - 1].Neurons[j].Weights.Count];
                    for (int hiddenLayerIndex = network.Layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
                    {
                        nablaWeights[hiddenLayerIndex] = new double[network.Layers[hiddenLayerIndex].Neurons.Length][];
                        nablaBiases[hiddenLayerIndex] = new double[network.Layers[hiddenLayerIndex].Neurons.Length];
                        for (int j = 0; j < network.Layers[hiddenLayerIndex].Neurons.Length; j++)
                            nablaWeights[hiddenLayerIndex][j] = new double[network.Layers[hiddenLayerIndex].Neurons[j].Weights.Count];
                    }

                    //process one batch
                    for (int inBatchIndex = currentIndex; inBatchIndex < currentIndex + _config.BatchSize && inBatchIndex < data.Count; inBatchIndex++)
                    {
                        Iteration(network, data[trainingIndices[inBatchIndex]], nablaWeights, nablaBiases);
                    }

                    //update weights and bias
                    UpdateNetwork(network, nablaWeights, nablaBiases);

                    currentIndex += _config.BatchSize;
                } while (currentIndex < data.Count);

                //recalculating error on training data
                currentError = 0;
                for (int i = 0; i < data.Count; i++)
                {
                    double[] realOutput = network.ComputeOutput(data[i].Input);
                    currentError += _config.ErrorFunction.Calculate(data[i].Output, realOutput);
                }
                currentError *= 1d / data.Count;

                #endregion

                epochNumber++;
                if (Math.Abs(currentError - lastError) < _config.MinErrorChange)
                    _config.LearningRate /= 2;
                Console.WriteLine("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());

                sr.WriteLine(epochNumber.ToString() + "," + currentError.ToString().Replace(",", "."));
                /*Logger.Instance.Log("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());*/
            } while (epochNumber < _config.MaxEpoches && currentError > _config.MinError);
            sr.Close();
        }
        void Iteration(IMultilayerNeuralNetwork network, DoubleData data, double[][][] nablaWeights, double[][] nablaBiases)
        {
            //forward pass
            double[] realOutput = network.ComputeOutput(data.Input);

            //backward pass, error propagation
            //last layer
            for (int j = 0; j < network.Layers[network.Layers.Length - 1].Neurons.Length; j++)
            {
                double a = _config.ErrorFunction.CalculatePartialDerivativeByA2Index(data.Output, realOutput, j);
                double b = (data.Output[j] - realOutput[j]);
                network.Layers[network.Layers.Length - 1].Neurons[j].ErrorDerivative =
                    a *
                     network.Layers[network.Layers.Length - 1].Neurons[j].ActivationFunction.
                        ComputeFirstDerivative(network.Layers[network.Layers.Length - 1].Neurons[j].WeightedSum);

                nablaBiases[network.Layers.Length - 1][j] += _config.LearningRate *
                                                            network.Layers[network.Layers.Length - 1].Neurons[j].ErrorDerivative;
                for (int i = 0; i < network.Layers[network.Layers.Length - 1].Neurons[j].Weights.Count; i++)
                {
                    nablaWeights[network.Layers.Length - 1][j][i] +=
                        _config.LearningRate * network.Layers[network.Layers.Length - 1].Neurons[j].ErrorDerivative *
                            (network.Layers.Length > 1 ?
                                network.Layers[network.Layers.Length - 2].Neurons[i].LastState : data.Input[i]);
                }
            }

            //hidden layers
            for (int hiddenLayerIndex = network.Layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
            {
                for (int j = 0; j < network.Layers[hiddenLayerIndex].Neurons.Length; j++)
                {
                    network.Layers[hiddenLayerIndex].Neurons[j].ErrorDerivative = 0;
                    for (int k = 0; k < network.Layers[hiddenLayerIndex + 1].Neurons.Length; k++)
                    {
                        network.Layers[hiddenLayerIndex].Neurons[j].ErrorDerivative +=
                            network.Layers[hiddenLayerIndex + 1].Neurons[k].Weights[j] *
                            network.Layers[hiddenLayerIndex + 1].Neurons[k].ErrorDerivative;
                    }
                    network.Layers[hiddenLayerIndex].Neurons[j].ErrorDerivative *=
                        network.Layers[hiddenLayerIndex].Neurons[j].ActivationFunction.
                            ComputeFirstDerivative(network.Layers[hiddenLayerIndex].Neurons[j].WeightedSum);

                    nablaBiases[hiddenLayerIndex][j] += _config.LearningRate *
                                                       network.Layers[hiddenLayerIndex].Neurons[j].ErrorDerivative;
                    for (int i = 0; i < network.Layers[hiddenLayerIndex].Neurons[j].Weights.Count; i++)
                    {
                        nablaWeights[hiddenLayerIndex][j][i] += _config.LearningRate *
                            network.Layers[hiddenLayerIndex].Neurons[j].ErrorDerivative *
                            (hiddenLayerIndex > 0 ? network.Layers[hiddenLayerIndex - 1].Neurons[i].LastState : data.Input[i]);
                    }
                }
            }
        }

        void UpdateNetwork(IMultilayerNeuralNetwork network, double[][][] nablaWeights, double[][] nablaBiases)
        {
            for (int layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < network.Layers[layerIndex].Neurons.Length; neuronIndex++)
                {
                    network.Layers[layerIndex].Neurons[neuronIndex].Bias += nablaBiases[layerIndex][neuronIndex];
                    for (int weightIndex = 0; weightIndex < network.Layers[layerIndex].Neurons[neuronIndex].Weights.Count; weightIndex++)
                    {
                        network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] +=
                            nablaWeights[layerIndex][neuronIndex][weightIndex];
                    }
                }
            }
        }

        private int[] Shuffle(int[] arr)
        {
            for (int i = 0; i < arr.Length - 1; i++)
            {
                if (_random.NextDouble() >= 0.3d)
                {
                    int newIndex = _random.Next(arr.Length);
                    int tmp = arr[i];
                    arr[i] = arr[newIndex];
                    arr[newIndex] = tmp;
                }
            }
            return arr;
        }
    }
}

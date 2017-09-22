using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Learning.Interfaces;
using NeuralNetworkLibrary.Structure.Networks.Interfaces;
using NeuralNetworkLibrary.Data;
using NLog;

namespace NeuralNetworkLibrary.Learning
{
    internal class BackpropagationMLNLearningAlgorithm : ILearningStrategy<IMultilayerNeuralNetwork>
    {
        private LearningConfig config;
        private Random rnd;
        private static Logger logger = LogManager.GetCurrentClassLogger();
        private IMultilayerNeuralNetwork network;


        internal BackpropagationMLNLearningAlgorithm(IMultilayerNeuralNetwork Network,LearningConfig Config)
        {
            config = Config;
            network = Network;
            //_random = new Random(Helper.GetSeed());
            rnd = new Random();
        }


        // Train to fulfill config settings
        public void train(IList<DoubleData> data)
        {
            if (config.BatchSize < 1 || config.BatchSize > data.Count)
            {
                config.BatchSize = data.Count;
            }

            double currentError = Single.MaxValue;
            double lastError = 0;
            int epochNumber = 0;
            logger.Info("Start learning...");

            do
            {
                lastError = currentError;
                DateTime dtStart = DateTime.Now;

                Iteration(data);

                //recalculating error on all data
                currentError = 0;
                for (int i = 0; i < data.Count; i++)
                {
                    double[] realOutput = network.ComputeOutput(data[i].Input);
                    currentError += config.ErrorFunction.Calculate(data[i].Output, realOutput);
                }
                currentError *= 1d / data.Count;

                epochNumber++;
                if (Math.Abs(currentError - lastError) < config.MinErrorChange)
                    config.LearningRate /= 2;
                Console.WriteLine("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());
                logger.Trace("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());
                
            } while (epochNumber < config.MaxEpoches && currentError > config.MinError);
        }

        // One time data pass
        public void Iteration(IList<DoubleData> data)
        {
            // Preparation for epoche
            int[] trainingIndices = new int[data.Count];

            for (int i = 0; i < data.Count; i++)
            {
                trainingIndices[i] = i;
            }
            if (config.BatchSize > 0)
            {
                trainingIndices = Shuffle(trainingIndices);
            }

            // Process data set
            int currentIndex = 0;
            do
            {
                /*
                // Prepare variables for accumulated error for a data batch for weights and biases correction
                double[][][] nablaWeights = new double[network.Layers.Length][][];
                double[][] nablaBiases = new double[network.Layers.Length][];
                for (int layerIndex = network.Layers.Length - 1; layerIndex >= 0; layerIndex--)
                {
                    nablaWeights[layerIndex] = new double[network.Layers[layerIndex].Neurons.Length][];
                    nablaBiases[layerIndex] = new double[network.Layers[layerIndex].Neurons.Length];
                    for (int j = 0; j < network.Layers[layerIndex].Neurons.Length; j++)
                        nablaWeights[layerIndex][j] = new double[network.Layers[layerIndex].Neurons[j].Weights.Count];
                }*/
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

                // Process one batch
                for (int inBatchIndex = currentIndex; inBatchIndex < currentIndex + config.BatchSize && inBatchIndex < data.Count; inBatchIndex++)
                {
                    // Forward pass
                    double[] networkOutput = network.ComputeOutput(data[trainingIndices[inBatchIndex]].Input);

                    // Start backward pass for error propagation

                    // Last layer
                    // Declare layer index
                    int layerIndex = network.Layers.Length - 1;
                    for (int i = 0; i < network.Layers[layerIndex].Neurons.Length; i++)
                    {
                        network.Layers[layerIndex].Neurons[i].ErrorDerivative = network.Layers[layerIndex].Neurons[i].ActivationFunctionDerivative
                            * config.ErrorFunction.CalculatePartialDerivativeByA2Index(data[trainingIndices[inBatchIndex]].Output, networkOutput, i);
                            
                        nablaBiases[layerIndex][i] += config.LearningRate * network.Layers[layerIndex].Neurons[i].ErrorDerivative;
                        
                        for (int j = 0; j < network.Layers[layerIndex].Neurons[i].Weights.Count; j++)
                        {
                            /*
                            double ls = _network.Layers[layerIndex].Neurons[i].LastState;
                            double w = _network.Layers[layerIndex].Neurons[i].Weights[j];
                            nablaWeights[layerIndex][i][j] = config.LearningRate *
                                (neuronErrorDerivative * _network.Layers[layerIndex].Neurons[i].LastState
                                + config.RegularizationFactor * _network.Layers[layerIndex].Neurons[i].Weights[j] / data.Count);*/

                            nablaWeights[layerIndex][i][j] += config.LearningRate * network.Layers[layerIndex].Neurons[i].ErrorDerivative
                                * network.Layers[layerIndex - 1].Neurons[j].LastState;
                        }
                    }

                    // First and hidden layers
                    for (layerIndex = network.Layers.Length - 2; layerIndex >= 0; layerIndex--)
                    {
                        for (int i = 0; i < network.Layers[layerIndex].Neurons.Length; i++)
                        {
                            network.Layers[layerIndex].Neurons[i].ErrorDerivative = 0;
                            for (int k = 0; k < network.Layers[layerIndex + 1].Neurons.Length; k++)
                            {
                                network.Layers[layerIndex].Neurons[i].ErrorDerivative +=
                                    network.Layers[layerIndex + 1].Neurons[k].Weights[i] * 
                                    network.Layers[layerIndex + 1].Neurons[k].ErrorDerivative;
                            }
                            network.Layers[layerIndex].Neurons[i].ErrorDerivative *= 
                                network.Layers[layerIndex].Neurons[i].ActivationFunctionDerivative;

                            nablaBiases[layerIndex][i] += config.LearningRate * network.Layers[layerIndex].Neurons[i].ErrorDerivative;

                            for (int j = 0; j < network.Layers[layerIndex].Neurons[i].Weights.Count; j++)
                            {
                                /*
                                nablaWeights[layerIndex][i][j] = config.LearningRate *
                                    (neuronErrorDerivative *
                                    (layerIndex > 0 ? _network.Layers[layerIndex - 1].Neurons[j].LastState : data[inBatchIndex].input[j])
                                     + config.RegularizationFactor * _network.Layers[layerIndex].Neurons[i].Weights[j] / data.Count);
                                     */

                                nablaWeights[layerIndex][i][j] += config.LearningRate * network.Layers[layerIndex].Neurons[i].ErrorDerivative
                                    * (layerIndex > 0 ? network.Layers[layerIndex - 1].Neurons[j].LastState : data[inBatchIndex].Input[j]);
                            }
                        }
                    }
                }
                
                // Update weights and bias
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
                currentIndex += config.BatchSize;
            } while (currentIndex < data.Count);
        }

        // Shuffle data indexes
        private int[] Shuffle(int[] array)
        {
            for (int i = 0; i < array.Length - 1; i++)
            {
                if (rnd.NextDouble() >= 0.3d)
                {
                    int newIndex = rnd.Next(array.Length);
                    int tmp = array[i];
                    array[i] = array[newIndex];
                    array[newIndex] = tmp;
                }
            }
            return array;
        }
    }
}

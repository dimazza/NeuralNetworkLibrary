using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Learning.Interfaces;
using NeuralNetworkLibrary.Structure.Networks.Interfaces;
using NeuralNetworkLibrary.Data;
using NLog;

namespace NeuralNetworkLibrary.Learning
{
    internal class learning //: ILearningStrategy<IMultilayerNeuralNetwork>
    {
        private LearningConfig config;
        private Random rnd;
        private static Logger logger = LogManager.GetCurrentClassLogger();
        private IMultilayerNeuralNetwork network;
        private double currentError = Single.MaxValue;
        private double lastError = 0;
        //private double lastBatchError = float.MaxValue;
        internal learning(IMultilayerNeuralNetwork Network, LearningConfig Config)
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
                currentError /= data.Count;

                epochNumber++;



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
                //accumulated error for batch, for weights and biases
                double[][][] deltaWeights = new double[network.Layers.Length][][];
                double[][] deltaBiases = new double[network.Layers.Length][];
                deltaWeights[network.Layers.Length - 1] = new double[network.Layers[network.Layers.Length - 1].Neurons.Length][];
                deltaBiases[network.Layers.Length - 1] = new double[network.Layers[network.Layers.Length - 1].Neurons.Length];
                for (int j = 0; j < network.Layers[network.Layers.Length - 1].Neurons.Length; j++)
                    deltaWeights[network.Layers.Length - 1][j] = new double[network.Layers[network.Layers.Length - 1].Neurons[j].Weights.Count];
                for (int hiddenLayerIndex = network.Layers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
                {
                    deltaWeights[hiddenLayerIndex] = new double[network.Layers[hiddenLayerIndex].Neurons.Length][];
                    deltaBiases[hiddenLayerIndex] = new double[network.Layers[hiddenLayerIndex].Neurons.Length];
                    for (int j = 0; j < network.Layers[hiddenLayerIndex].Neurons.Length; j++)
                        deltaWeights[hiddenLayerIndex][j] = new double[network.Layers[hiddenLayerIndex].Neurons[j].Weights.Count];
                }
                //double currentBatchError = 0;
                
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
                        network.Layers[layerIndex].Neurons[i].ErrorDerivative =
                            network.Layers[layerIndex].Neurons[i].ActivationFunctionDerivative *
                            config.ErrorFunction.CalculatePartialDerivativeByA2Index(data[trainingIndices[inBatchIndex]].Output, networkOutput, i);

                        deltaBiases[layerIndex][i] += config.LearningRate * network.Layers[layerIndex].Neurons[i].ErrorDerivative;

                        for (int j = 0; j < network.Layers[layerIndex].Neurons[i].Weights.Count; j++)
                        {
                            /*deltaWeights[layerIndex][i][j] = config.LearningRate *
                                (network.Layers[layerIndex].Neurons[i].ErrorDerivative
                                * network.Layers[layerIndex - 1].Neurons[j].LastState
                                - config.RegularizationFactor * network.Layers[layerIndex].Neurons[i].Weights[j] / data.Count);*/

                            deltaWeights[layerIndex][i][j] += config.LearningRate *
                                network.Layers[layerIndex].Neurons[i].ErrorDerivative
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

                            deltaBiases[layerIndex][i] += config.LearningRate * network.Layers[layerIndex].Neurons[i].ErrorDerivative;

                            for (int j = 0; j < network.Layers[layerIndex].Neurons[i].Weights.Count; j++)
                            {
                                /*
                                deltaWeights[layerIndex][i][j] = config.LearningRate *
                                    (network.Layers[layerIndex].Neurons[i].ErrorDerivative
                                    * (layerIndex > 0 ? network.Layers[layerIndex - 1].Neurons[j].LastState : data[inBatchIndex].Input[j])
                                    - config.RegularizationFactor * network.Layers[layerIndex].Neurons[i].Weights[j] / data.Count);
                                    */
                               

                                deltaWeights[layerIndex][i][j] += config.LearningRate * network.Layers[layerIndex].Neurons[i].ErrorDerivative
                                    * (layerIndex > 0 ? network.Layers[layerIndex - 1].Neurons[j].LastState : data[trainingIndices[inBatchIndex]].Input[j]);

                            }
                        }
                    }
                }
                
                UpdateWeightes(deltaWeights, deltaBiases, config.BatchSize);
                currentIndex += config.BatchSize;
                //recalculating error on batch
                /* currentBatchError = 0;
                 for (int i = currentIndex; i < config.BatchSize; i++)
                 {
                     double[] realOutput = network.ComputeOutput(data[trainingIndices[i]].Input);
                     currentBatchError += config.ErrorFunction.Calculate(data[trainingIndices[i]].Output, realOutput);
                 }
                 currentBatchError /= config.BatchSize;
                 if (currentBatchError > lastBatchError)
                 {
                     RestoreWeightes(deltaWeights, deltaBiases, config.BatchSize);
                     config.LearningRate *= config.LearningChange;
                 }
                 else
                 {
                     currentIndex += config.BatchSize;
                     lastBatchError = currentBatchError;
                 }*/

            } while (currentIndex < data.Count);
        }
        public void UpdateWeightes(double[][][] deltaWeights,double[][] deltaBiases,int BatchSize)
        {
            // Update weights and bias
            for (int layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < network.Layers[layerIndex].Neurons.Length; neuronIndex++)
                {
                    network.Layers[layerIndex].Neurons[neuronIndex].Bias += deltaBiases[layerIndex][neuronIndex];
                    for (int weightIndex = 0; weightIndex < network.Layers[layerIndex].Neurons[neuronIndex].Weights.Count; weightIndex++)
                    {
                        /*network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] +=
                            deltaWeights[layerIndex][neuronIndex][weightIndex];*/

                        network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] +=
                            deltaWeights[layerIndex][neuronIndex][weightIndex] - config.LearningRate * config.RegularizationFactor * network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] / BatchSize;
                    }
                }
            }
        }
        public void RestoreWeightes(double[][][] deltaWeights, double[][] deltaBiases, int BatchSize)
        {
            // Update weights and bias
            for (int layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < network.Layers[layerIndex].Neurons.Length; neuronIndex++)
                {
                    network.Layers[layerIndex].Neurons[neuronIndex].Bias -= deltaBiases[layerIndex][neuronIndex];
                    for (int weightIndex = 0; weightIndex < network.Layers[layerIndex].Neurons[neuronIndex].Weights.Count; weightIndex++)
                    {
                        /*network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] +=
                            deltaWeights[layerIndex][neuronIndex][weightIndex];*/

                        network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] -=
                            deltaWeights[layerIndex][neuronIndex][weightIndex] - config.LearningRate * config.RegularizationFactor * network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] / BatchSize;
                    }
                }
            }
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

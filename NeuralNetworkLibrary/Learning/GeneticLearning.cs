using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Learning.Interfaces;
using NeuralNetworkLibrary.Structure.Networks.Interfaces;
using NeuralNetworkLibrary.Data;
using NLog;
using NeuralNetworkLibrary.Structure.Networks;
using NeuralNetworkLibrary.Learning.GeneticOperators;

namespace NeuralNetworkLibrary.Learning
{
    class GeneticLearning: ILearningStrategy<IMultilayerNeuralNetwork>
    {
        private LearningConfig config;
        private Random rnd;
        private static Logger logger = LogManager.GetCurrentClassLogger();
        private IMultilayerNeuralNetwork network;
        List<double> weigthesRange;
        private double currentTrainError = Single.MaxValue;
        private double lastTrainError = 0;
        private double currentValidateError = Single.MaxValue;
        private double lastValidateError1 = 0;
        private double lastValidateError2 = 0;
        private double lastValidateError3 = 0;
        private double currentCheckError = 0;
        Crossingover crossingover;
        Mutation mutation;
        int elite = 2;
        private IMultilayerNeuralNetwork networkL1;
        private IMultilayerNeuralNetwork networkL2;
        internal GeneticLearning(IMultilayerNeuralNetwork Network, LearningConfig Config)
        {
            config = Config;
            network = Network;
            //_random = new Random(Helper.GetSeed());
            rnd = new Random();
        }
        public void train(IList<DoubleData> train, IList<DoubleData> validate, IList<DoubleData> check)
        {
            int epochNumber = 0;
            logger.Info("Start learning...");
            MultiLayerNetwork[] population = Initialize(128);
            string Train = @"E:\Дима\NeuralNetworks\result.txt";
            /*string Validate = @"E:\Дима\NeuralNetworks\validate.txt";
            string Check = @"E:\Дима\NeuralNetworks\check.txt";*/
            System.IO.StreamWriter srT = new System.IO.StreamWriter(Train, false);
            /*System.IO.StreamWriter srV = new System.IO.StreamWriter(Validate, true);
            System.IO.StreamWriter srC = new System.IO.StreamWriter(Check, true);*/
            do
            {
                lastTrainError = currentTrainError;
                DateTime dtStart = DateTime.Now;

                population=Iteration(population,train);

                //recalculating error on all train
                currentTrainError = 0;
                for (int i = 0; i < train.Count; i++)
                {
                    double[] realOutput = population[0].ComputeOutput(train[i].Input);
                    currentTrainError += config.ErrorFunction.Calculate(train[i].Output, realOutput);
                }
                currentTrainError /= train.Count;

                lastValidateError3 = lastValidateError2;
                lastValidateError2 = lastValidateError1;
                lastValidateError1 = currentValidateError;
                //recalculating error on all validate
                currentValidateError = 0;
                for (int i = 0; i < validate.Count; i++)
                {
                    double[] realOutput = population[0].ComputeOutput(validate[i].Input);
                    currentValidateError += config.ErrorFunction.Calculate(validate[i].Output, realOutput);
                }
                currentValidateError /= validate.Count;


                currentCheckError = 0;
                for (int i = 0; i < check.Count; i++)
                {
                    double[] realOutput = population[0].ComputeOutput(check[i].Input);
                    currentCheckError += config.ErrorFunction.Calculate(check[i].Output, realOutput);
                }
                currentCheckError /= check.Count;
                epochNumber++;



                Console.WriteLine("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentTrainError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());
                logger.Trace("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentTrainError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());
                string ss = epochNumber.ToString() + "," + currentTrainError.ToString().Replace(",", ".") + "," + currentValidateError.ToString().Replace(",", ".") + "," + currentCheckError.ToString().Replace(",", ".");
                srT.WriteLine(ss);
                /*srV.WriteLine(epochNumber.ToString() + "," + currentiValidateError.ToString().Replace(",", "."));
                srC.WriteLine(epochNumber.ToString() + "," + currentCheckError.ToString().Replace(",", "."));*/

            } //while (epochNumber < config.MaxEpoches && currentTrainError > config.MinError&&(currentValidateError<=lastValidateError1||currentValidateError<=lastValidateError2||currentValidateError<=lastValidateError3));
            while (epochNumber < config.MaxEpoches && currentTrainError > config.MinError);
            srT.Close();
            /*srV.Close();
            srC.Close();*/
            UpdateWeightes(population[0]);
        }

        public void train(IList<DoubleData> train)
        {
            int epochNumber = 0;
            logger.Info("Start learning...");
            MultiLayerNetwork[] population = Initialize(128);
            string Train = @"E:\Дима\NeuralNetworks\train.txt";
            System.IO.StreamWriter srT = new System.IO.StreamWriter(Train, false);
            do
            {
                lastTrainError = currentTrainError;
                DateTime dtStart = DateTime.Now;

                population = Iteration(population, train);

                //recalculating error on all train
                currentTrainError = 0;
                for (int i = 0; i < train.Count; i++)
                {
                    double[] realOutput = population[0].ComputeOutput(train[i].Input);
                    currentTrainError += config.ErrorFunction.Calculate(train[i].Output, realOutput);
                }
                currentTrainError /= train.Count;

                epochNumber++;



                Console.WriteLine("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentTrainError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());
                logger.Trace("Eposh #" + epochNumber.ToString() +
                                    " finished; current error is " + currentTrainError.ToString() +
                                    "; it takes: " +
                                    (DateTime.Now - dtStart).Duration().ToString());

                srT.WriteLine(epochNumber.ToString() + "," + currentTrainError.ToString().Replace(",", "."));
            } while (epochNumber < config.MaxEpoches && currentTrainError > config.MinError);
            srT.Close();
            UpdateWeightes(population[0]);
        }
        public void UpdateWeightes(MultiLayerNetwork newNetwork)
        {
            // Update weights and bias
            for (int layerIndex = 0; layerIndex < network.Layers.Length; layerIndex++)
            {
                for (int neuronIndex = 0; neuronIndex < network.Layers[layerIndex].Neurons.Length; neuronIndex++)
                {
                    network.Layers[layerIndex].Neurons[neuronIndex].Bias = newNetwork.Layers[layerIndex].Neurons[neuronIndex].Bias;
                    for (int weightIndex = 0; weightIndex < network.Layers[layerIndex].Neurons[neuronIndex].Weights.Count; weightIndex++)
                    {
                        network.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex] = newNetwork.Layers[layerIndex].Neurons[neuronIndex].Weights[weightIndex];
                    }
                }
            }
        }
        public MultiLayerNetwork[] Iteration(MultiLayerNetwork[] population, IList<DoubleData> data)
        {
            MultiLayerNetwork[] childs = new MultiLayerNetwork[population.Length];
            double[] errors = new double[population.Length];
            for (int n = 0; n < population.Length; n++)
            {
                for (int i = 0; i < data.Count; i++)
                {
                    double[] realOutput = population[n].ComputeOutput(data[i].Input);
                    errors[n] += config.ErrorFunction.Calculate(data[i].Output, realOutput);
                }
                errors[n] /= data.Count;
            }
            Array.Sort(errors, population);
            double sum = Sum(errors);
            for (int i=0;i< errors.Length ; i++) 
            {
                errors[i] = errors[i] / sum;
            }
            for (int i = 0; i < errors.Length; i++)
            {
                errors[i] = 1 / errors[i];
            }
            sum = Sum(errors);
            for (int i = 0; i < errors.Length; i++)
            {
                errors[i] = errors[i] / sum;
            }
            for (int i = 1; i < errors.Length; i++)
            {
                errors[i] += errors[i-1];
            }
            for (int i = 0; i <  elite; i++)
            {
                childs[i] = population[i];
            }
            for (int i =  elite; i < population.Length; i+=2)
            {
                int[] indexes= selectParents(errors);
                MultiLayerNetwork[] temp = crossingover.copulation(population[indexes[0]], population[indexes[1]],rnd.Next(0,network.ConectionsCount));
                foreach(MultiLayerNetwork n in temp)
                {
                    mutation.mutate(n);
                }
                childs[i] = temp[0];
                childs[i + 1] = temp[1];
            }
            return childs;
        }
        MultiLayerNetwork[] Initialize(int N)
        {
            MultiLayerNetwork[] population = new MultiLayerNetwork[N];
            crossingover = new Crossingover();
            
            weigthesRange = new List<double>();
            int M = 1000000;
            for(int i=0;i<M;i++)
            {
                weigthesRange.Add(((double)i / M) * 2 - 1);
            }
            mutation = new Mutation(rnd, weigthesRange,10);
            int[] neurons = new int[network.Layers.Length];
            for (int j = 0; j < network.Layers.Length; j++)
            {
                neurons[j] = network.Layers[j].Neurons.Length;
            }
            for (int j = 0; j < population.Length; j++)
            {
                population[j] = new MultiLayerNetwork(neurons, network.InputDimension);
            }

            for (int layer = 0; layer < network.Layers.Length; layer++)
            {
                for (int j = 0; j < network.Layers[layer].Neurons.Length; j++)
                {
                    for (int n = 0; n < network.Layers[layer].Neurons[j].Weights.Count; n++)
                    {
                        for (int k = 0; k < N; k++)
                        {
                            double A = weigthesRange[rnd.Next(0, M)];
                            population[k].Layers[layer].Neurons[j].Weights[n] = A;
                        }
                            
                    }
                }
            }
            return population;
        }
        int[] selectParents(double[] errors)
        {
            int[] parentsIndexes = new int[2];
            double random = rnd.NextDouble();
            for(int i=0;i<errors.Length;i++)
            {
                if (random < errors[i])
                {
                    parentsIndexes[0] = i;
                    break;
                }
            }
            do
            {
                random = rnd.NextDouble();
                for (int i = 0; i < errors.Length; i++)
                {
                    if (random < errors[i])
                    {
                        parentsIndexes[1] = i;
                        break;
                    }
                }
            } while (parentsIndexes[1] == parentsIndexes[0]);
            return parentsIndexes;
        }
        double Sum(double[] array)
        {
            double result = 0;
            foreach(double x in array)
            {
                result += x;
            }
            return result;
        }
    }
}

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
    class GibridLearning:ILearningStrategy<IMultilayerNeuralNetwork>
    {
        private LearningConfig config1;
        private LearningConfig config2;
        private Random rnd;
        private static Logger logger = LogManager.GetCurrentClassLogger();
        private IMultilayerNeuralNetwork network;
        ILearningStrategy<IMultilayerNeuralNetwork> backProgation;
        ILearningStrategy<IMultilayerNeuralNetwork> genetic;

        internal GibridLearning(IMultilayerNeuralNetwork Network, LearningConfig Config1, LearningConfig Config2)
        {
            config1 = new LearningConfig();
            config1.BatchSize = -1;
            config1.MaxEpoches = 0;
            config1.MinError = 0.000001;
            config1.MinErrorChange = 0.0000000001;

            config2 = new LearningConfig();
            config2.BatchSize = 20;
            config2.MaxEpoches = 200;
            config2.MinError = 0.000001;
            config2.MinErrorChange = 0.0000000001;
            config2.LearningRate = 0.5;

            network = Network;
            //_random = new Random(Helper.GetSeed());
            rnd = new Random();
            genetic = new GeneticLearning(network, config1);
            backProgation = new LearningBGP(network, config2);
        }
        public void train(IList<DoubleData> data)
        {
            genetic.train(data);
            backProgation.train(data);
        }
        public void train(IList<DoubleData> data, IList<DoubleData> validate, IList<DoubleData> check)
        {
            genetic.train(data, validate, check);
            backProgation.train(data, validate, check);
        }
    }
}

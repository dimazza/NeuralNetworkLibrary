using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Structure.Components.Interfaces;
using NeuralNetworkLibrary.Structure.Networks.Interfaces;
using NeuralNetworkLibrary.Structure.Components;
using NeuralNetworkLibrary.Data;
using NeuralNetworkLibrary.Learning;
using NeuralNetworkLibrary.Learning.Interfaces;
using System.IO;
using System.Linq;

namespace NeuralNetworkLibrary.Structure.Networks
{
    static class Extensions
    {
        public static List<T> Clone<T>(this List<T> listToClone) where T : ICloneable
        {
            return listToClone.Select(item => (T)item.Clone()).ToList();
        }
    }
    public class MultiLayerNetwork:IMultilayerNeuralNetwork
    {
        Layer[] layers;
        ILearningStrategy<IMultilayerNeuralNetwork> learnStrategy;
        int inputDimension=0;
        int conectionsCount = 0;

        public MultiLayerNetwork(int[] NeuronsInLayers, int InputDimension)
        {
            Random rnd = new Random();
            inputDimension = InputDimension;
            layers = new Layer[NeuronsInLayers.Length];
            layers[0] = new Layer(NeuronsInLayers[0], InputDimension);
            double[] weights = new double[InputDimension];
            for (int i = 0; i < InputDimension; i++)
            {
                weights[i] = 1;
            }
            for (int i = 0; i < NeuronsInLayers[0]; i++)
            {
                layers[0].AddNeuron(new MLNeuron(weights.ToList()));
                conectionsCount += weights.Length;
            }
            for (int i = 1; i < NeuronsInLayers.Length; i++)
            {
                layers[i] = new Layer(NeuronsInLayers[i], NeuronsInLayers[i - 1]);
                for (int j = 0; j < NeuronsInLayers[i]; j++)
                    layers[i].AddNeuron(new MLNeuron(NeuronsInLayers[i - 1], rnd));
                conectionsCount += NeuronsInLayers[i - 1];
            }
            LearningConfig config = new LearningConfig();
            config.BatchSize = -1;
            config.MaxEpoches = 1000;
            config.MinError = 0.000001;
            config.MinErrorChange = 0.0000000001;
            learnStrategy = new GibridLearning(this, config,config);
            //learnStrategy = new GeneticLearning(this, config);
        }


        public ILayer<double>[] Layers
        {
            get { return layers; }
        }
        public int InputDimension
        {
            get { return inputDimension; }
        }
        public int ConectionsCount
        {
            get { return conectionsCount; }
        }
        public double[] ComputeOutput(double[] inputVector)
        {
            double[] outputVector = layers[0].Compute(inputVector);
            for (int i = 1; i < layers.Length; i++)
            {
                outputVector = layers[i].Compute(outputVector);
            }
            return outputVector;
        }
        
        public void Train(IList<DoubleData> Data)
        {
            learnStrategy.train(Data);
        }
        public void Train(IList<DoubleData> Train, IList<DoubleData> Validate, IList<DoubleData> Check)
        {
            learnStrategy.train(Train,Validate,Check);
        }
        /*public void Train(IList<DoubleData> Data, LearningConfig config)
        {
            learnStrategy = new BackpropagationMLNLearningAlgorithm(this, config);
            learnStrategy.train(Data);
        }*/
        public void Save(string Path)
        {
            StreamWriter f = new StreamWriter(Path);
            f.WriteLine(layers.Length);
            foreach (Layer layer in layers)
            {
                f.WriteLine(layer.Neurons.Length);
                f.WriteLine(layer.InputDimension);
                foreach (INeuron<double> neuron in layer.Neurons)
                {
                    foreach (double a in neuron.Weights)
                        f.Write(a + " ");
                    f.Write(";");
                }
                f.WriteLine();
            }
            f.Close();
        }

        public void Load(Stream S)
        {
            StreamReader sr = new StreamReader(S);
            string s;
            int i = 0;
            s = sr.ReadLine();
            layers = new Layer[int.Parse(s)];
            while (!sr.EndOfStream)
            {
                s = sr.ReadLine();
                int n;
                int m;
                if (int.TryParse(s, out n))
                {
                    int.TryParse(sr.ReadLine(), out m);
                    layers[i] = new Layer(n, m);
                }
                else
                {
                    string[] st = s.Split(';');
                    List<double> weigth = new List<double>();
                    foreach(string line in st)
                    {
                        weigth.Add(double.Parse(line));
                    }
                    layers[i].AddNeuron(new MLNeuron(weigth));
                }
            }
        }
        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < layers.Length; i++)
            {
                s += String.Format("{0} слой", i) + "\r\n";
                s += layers[i].ToString() + "\r\n";
            }
            return s;
        }
    }
}

using System;
using System.Collections.Generic;
using NeuralNetworkLibrary.Structure.Components.Interfaces;

namespace NeuralNetworkLibrary.Structure.Components
{
    public class Neuron : INeuron<double>
    {
        protected Random rnd;
        protected List<double> weights;
        protected double bias = 0;
        protected double weightedSum = 0;
        protected double lastState = 0;
        protected double errorDerivative;
        //protected IFunction function = new Functions.LinearFunction();
        protected IFunction function=new Functions.SigmoidFunction();
        //protected IFunction function = new Functions.HyperbolicTangensFunction();
        protected List<INeuron<double>> children;
        protected List<INeuron<double>> parents;


        protected Neuron()
        {
            parents = new List<INeuron<double>>();
            children = new List<INeuron<double>>();
        }
        public Neuron(Random random) : this()
        {
            rnd = random;
        }
        public Neuron(List<INeuron<double>> Parents, List<INeuron<double>> Children, Random random)
        {
            parents = Parents;
            children = Children;
            rnd = random;
            weights = new List<double>();
            for (int k = 0; k < parents.Count; k++)
                weights.Add((rnd.Next(1, 3) == 1 ? -1 : 1) * rnd.Next(100) / 200f);
        }
        public Neuron(List<INeuron<double>> Parents, List<INeuron<double>> Children, List<double> Weights)
        {
            parents = Parents;
            children = Children;
            weights = Weights;
        }
        public Neuron(List<INeuron<double>> Parents, List<INeuron<double>> Children, List<double> Weights, IFunction Function)
        {
            parents = Parents;
            children = Children;
            weights = Weights;
            function = Function;
        }
        

        public List<double> Weights { get { return weights; } }
        public double Bias
        {
            get { return bias; }
            set { bias = value; }
        }
        public double LastState
        {
            get { return lastState; }
            set { lastState = value; }
        }
        public double WeightedSum
        {
            get { return weightedSum; }
            set { weightedSum = value; }
        }
        public IList<INeuron<double>> Parents { get { return parents; } }
        public IList<INeuron<double>> Children { get { return children; } }
        public double ActivationFunctionDerivative
        {
            get
            {
                return function.ComputeFirstDerivative(WeightedSum);
            }
        }
        public double ErrorDerivative
        {
            get { return errorDerivative; }
            set { errorDerivative = value; }
        }
        public IFunction ActivationFunction { get { return function; } set { function = value; } }

        public void addParent(INeuron<double> Neuron)
        {
            parents.Add(Neuron);
        }
        public void addChild(INeuron<double> Neuron)
        {
            children.Add(Neuron);
        }
        public double computeWeightedSum()
        {
            /*double[] inputVector = new double[parents.Count];
            for(int i=0;i<parents.Count;i++)
            {
                inputVector[i] = parents[i].LastState;
            }
            weightedSum = bias;
            for (int i = 0; i < inputVector.Length; i++)
                weightedSum += weights[i] * inputVector[i];*/

            weightedSum = bias;
            for (int i = 0; i < parents.Count; i++)
                weightedSum += weights[i] * parents[i].LastState;
            return weightedSum;
        }
        public double computeWeightedSum(double[] inputVector)
        {
            weightedSum = bias;
            for (int i = 0; i < inputVector.Length; i++)
                weightedSum += weights[i] * inputVector[i];
            return weightedSum;
        }
        public double Activate(double[] inputVector)
        {
            lastState = function.Compute(computeWeightedSum(inputVector));
            return lastState;
        }

        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < weights.Count; i++)
                s += weights[i].ToString() + " ";
            return s;
        }
    }
}

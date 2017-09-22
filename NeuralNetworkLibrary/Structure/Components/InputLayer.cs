using NeuralNetworkLibrary.Structure.Components.Interfaces;

namespace NeuralNetworkLibrary.Structure.Components
{
    /*public class InputLayer : ILayer
    {
        INeuron[] neurons;
        double[] lastOutput;
        int Count = 0;
        

        public InputLayer(int Neurons)
        {
            neurons = new INeuron[Neurons];
            lastOutput = new double[Neurons];
        }

        public double[] LastOutput { get { return LastOutput; } }
        public INeuron[] Neurons { get { return neurons; } }
        public int InputDimension { get { return neurons[0].Weights.Count; } }


        public void AddNeuron(INeuron neuron)
        {
            neurons[Count] = neuron;
            Count++;
        }
        public double[] Compute(double[] inputVector)
        {
            for (int i = 0; i < neurons.Length; i++)
                lastOutput[i] = neurons[i].Activate(inputVector);
            return lastOutput;
        }
    }*/
}

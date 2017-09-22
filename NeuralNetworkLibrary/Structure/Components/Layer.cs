using NeuralNetworkLibrary.Structure.Components.Interfaces;

namespace NeuralNetworkLibrary.Structure.Components
{
    public class Layer : ILayer<double>
    {
        INeuron<double>[] neurons;
        double[] lastOutput;
        int count = 0;
        int inputDimension;


        public Layer(int Neurons, int InputDimension)
        {
            neurons = new INeuron<double>[Neurons];
            lastOutput = new double[Neurons];
            inputDimension = InputDimension;
        }


        public double[] LastOutput { get { return lastOutput; } }
        public INeuron<double>[] Neurons { get { return neurons; } }
        public int InputDimension { get { return inputDimension; } }
        public int Count { get { return count; } }


        public void AddNeuron(INeuron<double> neuron)
        {
            neurons[count] = neuron;
            count++;
        }
        public double[] Compute(double[] inputVector)
        {
            for (int i = 0; i < neurons.Length; i++)
                lastOutput[i] = neurons[i].Activate(inputVector);
            return lastOutput;
        }
        public override string ToString()
        {
            string s = "";
            for (int i = 0; i < Neurons.Length; i++)
                s += Neurons[i].ToString() + "\r\n";
            return s;
        }
    }
}

namespace NeuralNetworkLibrary.Data
{
    public class DoubleData : DataItem<double, double[][]>
    {
        public DoubleData() : base() { }
        public DoubleData(double[] Input, double[] Output) : base(Input, Output) { }


        public override void Normalize(double[][] Coef)
        {
            for (int i = 0; i < input.Length; i++)
            {
                input[i] /= Coef[0][i];
                input[i] -= 0.5;
                input[i] *= 2;
            }
            /*for (int i = 0; i < output.Length; i++)
            {
                output[i] /= Coef[1][i];
                output[i] -= 0.5;
                output[i] *= 2;
            }*/
        }
        public override void DeNormalize(double[][] Coef)
        {
            for (int i = 0; i < input.Length; i++)
            {
                input[i] /= 2;
                input[i] += 0.5;
                input[i] *= Coef[0][i];
            }
            for (int i = 0; i < output.Length; i++)
            {
                output[i] /= 2;
                output[i] += 0.5;
                output[i] *= Coef[1][i];
            }
        }
    }
}

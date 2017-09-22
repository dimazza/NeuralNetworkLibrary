using System.Collections.Generic;

namespace NeuralNetworkLibrary.Data
{
    public class DoubleDataList
    {
        private List<DoubleData> data;
        private double[][] coef;


        public DoubleDataList()
        {
            data = new List<DoubleData>();
        }
        public DoubleDataList(List<DoubleData> data)
        {
            this.data = data;
        }

        public DoubleData this[int i] { get { return data[i]; } }
        public List<DoubleData> DataList { get { return data; } }
        public double[][] Coef { get { return coef; } }


        public void Normalize()
        {
            coef = new double[2][];
            coef[0] = new double[data[0].Input.Length];
            coef[1] = new double[data[0].Output.Length];

            for (int i = 0; i < data.Count; i++)
            {
                for (int j = 0; j < data[i].Input.Length; j++)
                    if (coef[0][j] < data[i].Input[j])
                        coef[0][j] = data[i].Input[j];

                for (int j = 0; j < data[i].Output.Length; j++)
                    if (coef[1][j] < data[i].Output[j])
                        coef[1][j] = data[i].Output[j];

            }

            for (int i = 0; i < data.Count; i++)
                data[i].Normalize(coef);
        }
        public void Normalize(double[][] Coef)
        {
            coef = Coef;
            for (int i = 0; i < data.Count; i++)
                data[i].Normalize(coef);

        }
        public void Denormalize()
        {
            for (int i = 0; i < data.Count; i++)
                data[i].DeNormalize(coef);
        }
        public void Denormalize(double[][] Coef)
        {
            coef = Coef;
            for (int i = 0; i < data.Count; i++)
                data[i].DeNormalize(coef);
        }
    }
}

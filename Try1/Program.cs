using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetworkLibrary.Structure.Networks;
using NeuralNetworkLibrary.Data;
using System.IO;
using System.Diagnostics;
using NeuralNetworkLibrary.Learning;
using NeuralNetworkLibrary.Learning.Metrics;
using NeuralNetworkLibrary.Learning.Interfaces;
namespace Try1
{
    public class Abolone
    {
        public Abolone(double age, double sex, double PainType, double bloodPressure, double Serum, double bloodSugar, double restingElectrocardiographic,
            double maximumHeartRate, double inducedAngina, double oldpeakST, double peakST, double majorVessel, double thal)
        {
            Sex = sex;
            this.PainType = PainType;
            this.bloodPressure = bloodPressure;
            this.Serum = Serum;
            this.bloodSugar = bloodSugar;
            this.restingElectrocardiographic = restingElectrocardiographic;
            this.maximumHeartRate = maximumHeartRate;
            this.inducedAngina = inducedAngina;
            this.oldpeakST = oldpeakST;
            this.peakST = peakST;
            this.majorVessel = majorVessel;
            this.thal = thal;
            Age = age;
        }
        public double Age;
        public double Sex;
        public double PainType;
        public double bloodPressure;
        public double Serum;
        public double bloodSugar;
        public double restingElectrocardiographic;
        public double maximumHeartRate;
        public double inducedAngina;
        public double oldpeakST;
        public double peakST;
        public double majorVessel;
        public double thal;
    }
    class Program
    {
        static void Main(string[] args)
        {
            List<DoubleData> all = new List<DoubleData>();
            List<DoubleData> train = new List<DoubleData>();
            List<DoubleData> validate = new List<DoubleData>();
            List<DoubleData> check = new List<DoubleData>();
            using (StreamReader sr = new StreamReader(@"E:\Дима\FNN\NeuralNetworks\OboloneAge\bin\Debug\data.txt"))
            {
                while (!sr.EndOfStream)
                {
                    string[] s = sr.ReadLine().Replace('.', ',').Split(' ');
                    double[] input = new double[] { double.Parse(s[0]), double.Parse(s[1]), double.Parse(s[2]), double.Parse(s[3]), double.Parse(s[4]), double.Parse(s[5]),
                        double.Parse(s[6]), double.Parse(s[7]), double.Parse(s[8]), double.Parse(s[9]), double.Parse(s[10]), double.Parse(s[11]),double.Parse(s[12]) };
                    double[] output = new double[2];
                    if (double.Parse(s[13]) == 1)
                        output[0] = 1;
                    else
                        output[1] = 1;
                    all.Add(new DoubleData(input, output));

                }
            }
            MultiLayerNetwork MLN = new MultiLayerNetwork(new int[] { 20, 2 }, 13);
            DoubleDataList Data = new DoubleDataList(all);
            Data.Normalize();
            all = Data.DataList;
            double[][] coef = Data.Coef;

            for (int i = 0; i < all.Count; i++)
            {
                if (i % 10 < 7)
                    train.Add(all[i]);
                else
                    if (i % 10 < 9)
                    validate.Add(all[i]);
                else
                    check.Add(all[i]);
            }

            Stopwatch stopWatch = new Stopwatch();
            stopWatch.Start();
            MLN.Train(train, validate, check);
            stopWatch.Stop();
            Console.WriteLine(stopWatch.Elapsed);
            Console.WriteLine();
            //MLN.Save("a.txt");

            double error = 0, errorT = 0; ;
            for (int i = 0; i < 10; i++)
            {
                double result1 = MLN.ComputeOutput(check[i].Input)[0];
                double result2 = MLN.ComputeOutput(check[i].Input)[1];
                double output1 = check[i].Output[0];
                double output2 = check[i].Output[1];
                errorT = Math.Pow(result1 - output1, 2);
                errorT += Math.Pow(result2 - output2, 2);
                Console.WriteLine(output1.ToString() + "   " + result1.ToString() + " : " + output2.ToString() + "   " + result2.ToString());
            }
            Console.WriteLine(error / check.Count);
        }
    }
}

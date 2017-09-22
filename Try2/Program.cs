using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuralNetworkLibrary.Structure.Networks;
using NeuralNetworkLibrary.Data;
using System.IO;
using System.Diagnostics;
using NeuralNetworkLibrary.Learning;
using NeuralNetworkLibrary.Learning.Interfaces;

namespace Try2
{
    public class Car
    {
        public Car(string Buying, string Maint, string Doors, string Persons, string Lug_boot, string Safety, string Type)
        {
            buying = Buying;
            maint = Maint;
            doors = Doors;
            persons = Persons;
            lug_boot = Lug_boot;
            safety = Safety;
            type = Type;
        }
        public string buying;
        public string maint;
        public string doors;
        public string persons;
        public string lug_boot;
        public string safety;
        public string type;
    }
    class Program
    {
       
        public static List<Car> ParseC(string pass)
        {
            List<Car> result = new List<Car>();
            using (StreamReader sr = new StreamReader(pass))
            {
                while (!sr.EndOfStream)
                {
                    string[] s = sr.ReadLine().Replace('.', ',').Split(',');
                    result.Add(new Car(s[0], s[1], s[2], s[3], s[4], s[5],s[6]));

                }
            }
            return result;
        }
        static void Main(string[] args)
        {
            List<DoubleData> all = new List<DoubleData>();
            List<DoubleData> train = new List<DoubleData>();
            List<DoubleData> check = new List<DoubleData>();
            List<Car> cars = ParseC(@"E:\Дима\NeuralNetworks\NeuralNetworkLibrary\Try1\bin\Debug\test.txt");
            for(int i=0;i<cars.Count;i++)
            {
                double[] input = new double[6];
                double[] output = new double[1];
                switch (cars[i].buying)
                {
                    case "low":input[0] = 1;break;
                    case "med": input[0] = 2; break;
                    case "high": input[0] = 3; break;
                    case "vhigh": input[0] = 4; break;
                }
                switch (cars[i].maint)
                {
                    case "low": input[1] = 1; break;
                    case "med": input[1] = 2; break;
                    case "high": input[1] = 3; break;
                    case "vhigh": input[1] = 4; break;
                }
                switch (cars[i].doors)
                {
                    case "2": input[2] = 1; break;
                    case "3": input[2] = 2; break;
                    case "4": input[2] = 3; break;
                    case "5more": input[2] = 4; break;
                }
                switch (cars[i].persons)
                {
                    case "2": input[3] = 1; break;
                    case "4": input[3] = 2; break;
                    case "more": input[3] = 3; break;
                }
                switch (cars[i].lug_boot)
                {
                    case "small": input[4] = 1; break;
                    case "med": input[4] = 2; break;
                    case "big": input[4] = 3; break;
                }
                switch (cars[i].safety)
                {
                    case "low": input[5] = 1; break;
                    case "med": input[5] = 2; break;
                    case "high": input[1] = 3; break;
                }
                switch (cars[i].type)
                {
                    case "unacc": output[0] = 1; break;
                    case "acc": output[0] = 2; break;
                    case "good": output[0] = 3; break;
                    case "vgood": output[0] = 4; break;
                }
                all.Add(new DoubleData(input, output));
            }
            MultiLayerNetwork MLN = new MultiLayerNetwork(new int[] { 12, 6, 1 }, 6);
            DoubleDataList Data = new DoubleDataList(all);
            Data.Normalize();
            all = Data.DataList;
            for (int i = 0; i < all.Count; i++)
            {
                if (i % 8 != 7)
                    train.Add(all[i]);
                else
                    check.Add(all[i]);
            }
            MLN.Train(train);
            double[][] coef = Data.Coef;
            double error = 0, errorT = 0; ;
            int N = 0;
            for (int i = 0; i < check.Count; i++)
            {
                double result = MLN.ComputeOutput(check[i].Input)[0];
                double output = check[i].Output[0];
                output = (output / 3 * 2 + 0.5) * coef[1][0];
                result = (result / 3 * 2 + 0.5) * coef[1][0];
                errorT = Math.Abs(result - output);
                error += errorT;
                if (errorT > 0.1)
                    N++;
                Console.WriteLine(output.ToString() + "   " + result.ToString());
            }
            Console.WriteLine(error / check.Count);
            Console.WriteLine(N);
        }
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworkLibrary.Data
{
    class FuzzyDouble
    {
        double mediana;
        double fuzzyLeft;
        double fuzzyRigth;
        public FuzzyDouble() { }
        public FuzzyDouble(double A)
        {
            mediana = A;
            fuzzyLeft = 0;
            fuzzyRigth = 0;
        }
        public FuzzyDouble(double FuzzyLeft, double Mediana, double FuzzyRigth)
        {
            mediana = Mediana;
            //fuzzyLeft = Mediana - FuzzyLeft;
            //fuzzyRigth = FuzzyRigth - Mediana;
            fuzzyLeft = FuzzyLeft;
            fuzzyRigth = FuzzyRigth;
        }
        public static FuzzyDouble operator +(FuzzyDouble obj1, FuzzyDouble obj2)
        {
            FuzzyDouble result = new FuzzyDouble();
            result.mediana = obj1.mediana + obj2.mediana;
            result.fuzzyLeft = obj1.mediana + obj2.mediana-obj1.fuzzyLeft - obj2.fuzzyLeft;
            result.fuzzyRigth = obj1.fuzzyRigth + obj2.fuzzyRigth;
            return result;
        }
        public static FuzzyDouble operator -(FuzzyDouble obj1, FuzzyDouble obj2)
        {
            FuzzyDouble result = new FuzzyDouble();
            result.mediana = obj1.mediana - obj2.mediana;
            result.fuzzyLeft = obj1.fuzzyLeft - obj2.fuzzyLeft;
            result.fuzzyRigth = obj1.fuzzyRigth - obj2.fuzzyRigth;
            return result;
        }
        public static FuzzyDouble operator *(FuzzyDouble obj1, FuzzyDouble obj2)
        {
            FuzzyDouble result = new FuzzyDouble();
            result.mediana = obj1.mediana * obj2.mediana;
            result.fuzzyLeft = obj1.fuzzyLeft * obj2.fuzzyLeft;
            result.fuzzyRigth = obj1.fuzzyRigth * obj2.fuzzyRigth;
            return result;
        }
        public static FuzzyDouble operator /(FuzzyDouble obj1, FuzzyDouble obj2)
        {
            FuzzyDouble result = new FuzzyDouble();
            result.mediana = obj1.mediana / obj2.mediana;
            result.fuzzyLeft = obj1.fuzzyLeft / obj2.fuzzyRigth;
            result.fuzzyRigth = obj1.fuzzyRigth / obj2.fuzzyLeft;
            return result;
        }
        public double Mediana
        {
            get { return mediana; }
            set { mediana = value; }
        }
        public override string ToString()
        {
            return String.Format("({0}, {1}, {2})", mediana - fuzzyLeft, mediana, mediana + fuzzyRigth);
        }
    }
}

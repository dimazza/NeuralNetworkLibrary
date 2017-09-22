using NeuralNetworkLibrary.Data.Interfaces;

namespace NeuralNetworkLibrary.Data
{
    /*public class FuzzyData: INormalize
    {
        double moda;
        double func;
        double coef;
        public FuzzyData() { }
        public FuzzyData(double Moda, double Func)
        {
            moda = Moda;
            func = Func;
        }
        public double Moda
        {
            get { return moda; }
            set { moda = value; }
        }
        public double Func
        {
            get { return func; }
            set { func = value; }
        }
        public override string ToString()
        {
            return moda.ToString() + "  " + func.ToString();
        }
        public void Normalize(double Coef)
        {
            moda /= coef;
            moda -= 0.5;
            func -= 0.5;
            coef = Coef;
        }
        public void Restore()
        {
            moda = (moda + 0.5) * coef;
            func += 0.5;
        }
    }*/
}

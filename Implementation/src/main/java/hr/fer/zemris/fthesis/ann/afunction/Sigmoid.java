package hr.fer.zemris.fthesis.ann.afunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public double valueAt(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivativeValueAt(double x) {
        double temp = valueAt(x);
        return temp * (1 - temp);
    }

    @Override
    public String toString() {
        return "sigmoid";
    }

}

package hr.fer.zemris.fthesis.ann.afunction;

public class Sigmoid implements ActivationFunction {

    @Override
    public double valueAt(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivativeValueAt(double x) {
        return valueAt(x) * (1 - valueAt(x));
    }

}

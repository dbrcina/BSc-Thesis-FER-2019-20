package hr.fer.zemris.fthesis.ann.afunction;

public class ReLU implements ActivationFunction {

    @Override
    public double valueAt(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivativeValueAt(double x) {
        return x >= 0 ? 1 : 0;
    }

}

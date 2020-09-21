package hr.fer.zemris.bscthesis.ann.afunction;

public class ReLU implements ActivationFunction {

    @Override
    public double valueAt(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivativeValueAt(double x) {
        return x > 0 ? 1 : 0;
    }

    @Override
    public String toString() {
        return "relu";
    }

}
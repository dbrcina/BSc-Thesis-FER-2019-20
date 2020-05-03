package hr.fer.zemris.fthesis.ann.afunction;

public class Tanh implements ActivationFunction {

    private final Sigmoid sigmoid = new Sigmoid();

    @Override
    public double valueAt(double x) {
        return 2 * sigmoid.valueAt(2 * x) - 1;
    }

    @Override
    public double derivativeValueAt(double x) {
        double temp = valueAt(x);
        return 1 - temp * temp;
    }

}

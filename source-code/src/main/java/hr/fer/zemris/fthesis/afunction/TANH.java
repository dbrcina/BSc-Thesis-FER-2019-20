package hr.fer.zemris.fthesis.afunction;

public class TANH implements ActivationFunction {

    @Override
    public double valueAt(double net) {
        return Math.tanh(net);
    }

    @Override
    public double derivativeValueAt(double net) {
        Sigmoid sigmoid = new Sigmoid();
        return 2 * sigmoid.valueAt(2 * net) - 1;
    }

}

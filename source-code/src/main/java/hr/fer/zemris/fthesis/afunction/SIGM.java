package hr.fer.zemris.fthesis.afunction;

public class SIGM implements ActivationFunction {

    @Override
    public double valueAt(double net) {
        return 1 / (1 + Math.exp(-net));
    }

    @Override
    public double deriveValueAt(double net) {
        return valueAt(net) * (1 - valueAt(net));
    }

}

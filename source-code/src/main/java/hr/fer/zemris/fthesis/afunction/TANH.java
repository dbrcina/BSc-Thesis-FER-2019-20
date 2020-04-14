package hr.fer.zemris.fthesis.afunction;

public class TANH implements ActivationFunction {

    @Override
    public double valueAt(double net) {
        return Math.tanh(net);
    }

    @Override
    public double deriveValueAt(double net) {
        SIGM sigm = new SIGM();
        return 2 * sigm.valueAt(2 * net) - 1;
    }

}

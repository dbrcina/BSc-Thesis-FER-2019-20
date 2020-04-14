package hr.fer.zemris.fthesis.afunction;

public class LRELU implements ActivationFunction {

    private final double alpha;

    public LRELU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double valueAt(double net) {
        return net > 0 ? net : alpha * net;
    }

    @Override
    public double deriveValueAt(double net) {
        return net > 0 ? 1 : alpha;
    }

}

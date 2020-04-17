package hr.fer.zemris.fthesis.afunction;

public class LReLU implements ActivationFunction {

    private final double alpha;

    public LReLU(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double valueAt(double x) {
        return x > 0 ? x : alpha * x;
    }

    @Override
    public double derivativeValueAt(double x) {
        return x > 0 ? 1 : alpha;
    }

}

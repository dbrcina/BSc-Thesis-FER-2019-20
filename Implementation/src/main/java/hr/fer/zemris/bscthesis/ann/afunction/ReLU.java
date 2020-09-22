package hr.fer.zemris.bscthesis.ann.afunction;

/**
 * An implementation of {@link ActivationFunction}. It represents <b>RECTIFIED LINEAR UNIT</b> activation function.
 *
 * @author dbrcina
 */
public class ReLU extends ActivationFunction {

    public ReLU() {
        super("ReLU");
    }

    @Override
    public double value(double x) {
        return Math.max(0, x);
    }

    @Override
    public double derivativeValue(double x) {
        return x > 0 ? 1 : 0;
    }

}

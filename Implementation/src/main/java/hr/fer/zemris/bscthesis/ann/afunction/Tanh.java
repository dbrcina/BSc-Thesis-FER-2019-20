package hr.fer.zemris.bscthesis.ann.afunction;

/**
 * An implementation of {@link ActivationFunction}. It represents <b>TANGENT HYPERBOLIC</b> activation function.
 *
 * @author dbrcina
 */
public class Tanh extends ActivationFunction {

    public Tanh() {
        super("Tanh");
    }

    @Override
    public double value(double x) {
        return Math.tanh(x);
    }

    @Override
    public double derivativeValue(double x) {
        double tanh = value(x);
        return 1 - tanh * tanh;
    }

}

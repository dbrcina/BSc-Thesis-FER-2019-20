package hr.fer.zemris.bscthesis.ann.afunction;

/**
 * An implementation of {@link ActivationFunction}. It represents <b>SIGMOID</b> activation function.
 *
 * @author dbrcina
 */
public class Sigmoid implements ActivationFunction {

    @Override
    public double value(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double derivativeValue(double x) {
        double sigma = value(x);
        return sigma * (1 - sigma);
    }

    @Override
    public String toString() {
        return "sigmoid";
    }

}

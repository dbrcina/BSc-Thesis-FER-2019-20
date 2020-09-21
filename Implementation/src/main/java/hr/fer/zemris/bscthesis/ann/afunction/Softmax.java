package hr.fer.zemris.bscthesis.ann.afunction;

import java.util.Arrays;

/**
 * An implementation of {@link ActivationFunction}. It represents <b>SOFTMAX</b> activation function.
 *
 * @author dbrcina
 */
public class Softmax implements ActivationFunction {

    private final double sum;

    public Softmax(double[] vector) {
        sum = Arrays.stream(vector)
                .map(Math::exp)
                .sum();
    }

    @Override
    public double value(double x) {
        return Math.exp(x) / sum;
    }

    @Override
    public double derivativeValue(double x) {
        double temp = value(x);
        return temp * (1 - temp);
    }

}

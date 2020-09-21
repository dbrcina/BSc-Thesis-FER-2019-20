package hr.fer.zemris.bscthesis.ann.afunction;

import java.util.Arrays;

/**
 * An implementation of {@link ActivationFunction}. It represents <b>SOFTMAX</b> activation function.
 *
 * @author dbrcina
 */
public class Softmax implements ActivationFunction {

    private final double sum;

    /**
     * It expects a vector of values from which sum is calculated as follows:
     * <pre>
     *     vector: [a1, a2, ..., an]
     *     sum = Math.exp(a1) + Math.exp(a2) + ... + Math.exp(an)
     * </pre>
     *
     * @param vector vector.
     */
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

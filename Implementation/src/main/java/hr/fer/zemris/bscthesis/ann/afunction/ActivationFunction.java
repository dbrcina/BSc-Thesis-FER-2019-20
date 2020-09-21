package hr.fer.zemris.bscthesis.ann.afunction;

/**
 * Model of activation function.
 *
 * @author dbrcina
 */
public interface ActivationFunction {

    /**
     * Calculates value at provided point <code>x</code>.
     *
     * @param x point x.
     * @return value at point <code>x</code>.
     */
    double value(double x);

    /**
     * Calculates derivative value at provided point <code>x</code>.
     *
     * @param x point x.
     * @return derivative value at point <code>x</code>.
     */
    double derivativeValue(double x);

}

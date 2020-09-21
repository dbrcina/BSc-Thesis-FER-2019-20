package hr.fer.zemris.fthesis.ann.afunction;

public interface ActivationFunction {

    double valueAt(double x);

    double derivativeValueAt(double x);

    String toString();

}

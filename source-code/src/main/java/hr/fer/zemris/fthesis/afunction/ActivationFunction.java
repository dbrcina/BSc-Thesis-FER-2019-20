package hr.fer.zemris.fthesis.afunction;

public interface ActivationFunction {

    double valueAt(double net);

    double deriveValueAt(double net);

}

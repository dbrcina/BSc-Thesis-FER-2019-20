package hr.fer.zemris.fthesis.ann;

import hr.fer.zemris.fthesis.afunction.ActivationFunction;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.transform.FastHadamardTransformer;

import java.util.Random;

/**
 * Feed forward artificial neural network.
 */
public class FFANN {

    private final Random rand = new Random();

    private final int[] layers;
    private RealMatrix[] biases;
    private RealMatrix[] weights;
    private ActivationFunction afunction;

    /**
     * Constructor.
     *
     * @param layers    an array of layers.
     * @param afunction activation function.
     */
    public FFANN(int[] layers, ActivationFunction afunction) {
        this.layers = layers;
        this.afunction = afunction;
        randomizeWeightsBiases();
    }

    /**
     * Randomizes weights and biases using normal distribution with mean 0 and variance 1.
     */
    public void randomizeWeightsBiases() {
        biases = new RealMatrix[layers.length - 1];  // ignore input layer, [1:]
        weights = new RealMatrix[layers.length - 1]; // ignore input layer, [1:]
        for (int i = 0; i < layers.length - 1; i++) {
            biases[i] = new Array2DRowRealMatrix(layers[i + 1], 1);
            weights[i] = new Array2DRowRealMatrix(layers[i + 1], layers[i]);
        }
        randomize(biases);
        randomize(weights);
    }

    private void randomize(RealMatrix[] elements) {
        for (RealMatrix el : elements) {
            for (int row = 0; row < el.getRowDimension(); row++) {
                for (int column = 0; column < el.getColumnDimension(); column++) {
                    el.setEntry(row, column, rand.nextGaussian());
                }
            }
        }
    }

    /**
     * Calculates outputs for provided inputs.
     *
     * @param inputs inputs.
     * @return outputs.
     */
    public double[] calculateOutputs(double[] inputs) {
        RealMatrix outputs = new Array2DRowRealMatrix(inputs);
        for (int i = 0; i < layers.length - 1; i++) {
            outputs = weights[i].multiply(outputs).add(biases[i]);
            for (int row = 0; row < outputs.getRowDimension(); row++) {
                for (int column = 0; column < outputs.getColumnDimension(); column++) {
                    outputs.setEntry(row, column, afunction.valueAt(outputs.getEntry(row, column)));
                }
            }
        }
        return outputs.getColumnVector(0).toArray();
    }

}

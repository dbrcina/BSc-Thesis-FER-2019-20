package hr.fer.zemris.fthesis.ann;

import hr.fer.zemris.fthesis.ann.afunction.ActivationFunction;
import hr.fer.zemris.fthesis.ann.dataset.ReadOnlyDataset;
import hr.fer.zemris.fthesis.ann.dataset.model.Sample;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Feed forward neural network <i>(Multilayer perceptron)</i> that uses <b>Backpropagation algorithm</b>
 * as learning algorithm.
 */
public class NeuralNetwork {

    private final int[] layers;
    private final ActivationFunction function;
    private final ReadOnlyDataset dataset;

    private final List<RealMatrix> weightsPerLayer = new ArrayList<>();
    private final List<RealMatrix> biasesPerLayer = new ArrayList<>();
    private final List<RealMatrix> outputsPerLayer = new ArrayList<>();
    private final List<RealMatrix> derivativesPerLayer = new ArrayList<>();
    private final List<RealMatrix> deltasPerLayer = new ArrayList<>();

    private final Random rand = new Random();
    private boolean matricesRandomized = false;

    public NeuralNetwork(int[] layers, ActivationFunction function, ReadOnlyDataset dataset) {
        this.layers = layers;
        this.function = function;
        this.dataset = dataset;
        setupMatrices();
    }

    // allocates memory for matrices
    private void setupMatrices() {
        for (int k = 0; k < layers.length; k++) {
            if (k != layers.length - 1) {
                weightsPerLayer.add(MatrixUtils.createRealMatrix(layers[k + 1], layers[k]));
                biasesPerLayer.add(MatrixUtils.createColumnRealMatrix(new double[layers[k + 1]]));
                derivativesPerLayer.add(MatrixUtils.createColumnRealMatrix(new double[layers[k + 1]]));
                deltasPerLayer.add(MatrixUtils.createColumnRealMatrix(new double[layers[k + 1]]));
            }
            outputsPerLayer.add(MatrixUtils.createColumnRealMatrix(new double[layers[k]]));
        }
    }

    // randomize between [-1, 1]
    private void randomizeMatrices() {
        if (!matricesRandomized) {
            matricesRandomized = true;
        }
        for (int k = 0; k < weightsPerLayer.size(); k++) {
            RealMatrix weightsLayerK = weightsPerLayer.get(k);
            RealMatrix biasesLayerK = biasesPerLayer.get(k);
            for (int row = 0; row < weightsLayerK.getRowDimension(); row++) {
                for (int column = 0; column < weightsLayerK.getColumnDimension(); column++) {
                    weightsLayerK.setEntry(row, column, rand.nextDouble() * 2 - 1);
                }
                biasesLayerK.setEntry(row, 0, rand.nextDouble() - 0.5);
            }
        }
    }

    /**
     * Feed forwards provided <i>inputs</i> and returns outputs as an array.
     *
     * @param inputs inputs.
     * @return results.
     * @throws IllegalArgumentException if number of input elements doesn't fit.
     */
    public double[] feedForward(double[] inputs) {
        if (inputs.length != layers[0]) {
            throw new IllegalArgumentException(String.format(
                    "Expected input of %d elements but received %d.",
                    layers[0], inputs.length));
        }
        if (!matricesRandomized) {
            randomizeMatrices();
        }
        outputsPerLayer.get(0).setColumn(0, inputs);
        for (int k = 0; k < weightsPerLayer.size(); k++) {
            RealMatrix weightsLayerK = weightsPerLayer.get(k);
            RealMatrix biasesLayerK = biasesPerLayer.get(k);
            RealMatrix outputsLayerK = outputsPerLayer.get(k);
            // calculate weighted sums
            RealMatrix outputsLayerK1 = (weightsLayerK.multiply(outputsLayerK)).add(biasesLayerK);
            double[] weightedSums = outputsLayerK1.getColumn(0);
            // apply activation function
            outputsPerLayer.get(k + 1).setColumn(
                    0, Arrays.stream(weightedSums).map(function::valueAt).toArray());
            // apply derivative activation function
            derivativesPerLayer.get(k).setColumn(
                    0, Arrays.stream(weightedSums).map(function::derivativeValueAt).toArray());
        }
        return outputsPerLayer.get(outputsPerLayer.size() - 1).getColumn(0);
    }

    public void train(int iterLimit, double maxError, double eta) {
        // randomize weights and biases
        randomizeMatrices();
        int numberOfSamples = dataset.numberOfSamples();
        // start iterations
        for (int iter = 0; iter < iterLimit; iter++) {
            double error = 0.0;
            // for every sample
            for (int i = 0; i < numberOfSamples; i++) {
                Sample sample = dataset.getSample(i);
                // feed forward sample
                double[] predictedOutputs = feedForward(sample.getInputs());
                double[] expectedOutputs = sample.getOutputs();
                // accumulate error
                for (int j = 0; j < expectedOutputs.length; j++) {
                    double subtract = expectedOutputs[j] - predictedOutputs[j];
                    error += subtract * subtract;
                }
                calculateDeltasPerLayer(expectedOutputs);
                updateWeightsBiases(eta);
            }
            // check accumulated error and print results
            error = error / (2 * numberOfSamples);
            boolean exit = error < maxError;
            if (iter == 0 || exit || (iter + 1) % 1000 == 0) {
                System.out.println("Iter " + (iter + 1) + "., error = " + error);
                if (exit) {
                    System.out.println("Found closest error! Exiting...");
                    break;
                }
            }
        }
    }

    private void calculateDeltasPerLayer(double[] expectedOutputs) {
        // calculate deltas for output layer
        RealVector target = new ArrayRealVector(expectedOutputs);
        RealVector actual = outputsPerLayer.get(outputsPerLayer.size() - 1).getColumnVector(0);
        RealVector subtraction = target.subtract(actual);
        // apply derivatives
        RealVector derivativesOutputLayer = derivativesPerLayer.get(deltasPerLayer.size() - 1)
                .getColumnVector(0);
        for (int i = 0; i < subtraction.getDimension(); i++) {
            subtraction.setEntry(i, derivativesOutputLayer.getEntry(i) * subtraction.getEntry(i));
        }
        deltasPerLayer.get(deltasPerLayer.size() - 1).setColumnVector(0, subtraction);
        // ------------------ //
        // calculate deltas for hidden layers
        for (int k = deltasPerLayer.size() - 2; k >= 0; k--) {
            RealVector derivativesLayerK = derivativesPerLayer.get(k).getColumnVector(0);
            RealMatrix weightsLayerK1 = weightsPerLayer.get(k + 1);
            RealVector biasesLayerK1 = biasesPerLayer.get(k + 1).getColumnVector(0);
            RealMatrix deltasLayerK1 = deltasPerLayer.get(k + 1);
            RealMatrix deltasLayerK = deltasPerLayer.get(k);
            double[] weightedSums = (weightsLayerK1.transpose().multiply(deltasLayerK1)).getColumn(0);
            // add biases
            for (int i = 0; i < weightedSums.length; i++) {
                for (int j = 0; j < biasesLayerK1.getDimension(); j++) {
                    weightedSums[i] += biasesLayerK1.getEntry(j) * deltasLayerK1.getEntry(j, 0);
                }
            }
            // apply derivatives
            for (int i = 0; i < deltasLayerK.getRowDimension(); i++) {
                deltasLayerK.setEntry(i, 0, derivativesLayerK.getEntry(i) * weightedSums[i]);
            }
        }
    }

    private void updateWeightsBiases(double eta) {
        for (int k = 0; k < weightsPerLayer.size(); k++) {
            RealMatrix weightsLayerK = weightsPerLayer.get(k);
            RealMatrix biasesPerLayerK = biasesPerLayer.get(k);
            RealVector outputsLayerK = outputsPerLayer.get(k).getColumnVector(0);
            RealVector deltasLayerK1 = deltasPerLayer.get(k).getColumnVector(0);
            for (int row = 0; row < weightsLayerK.getRowDimension(); row++) {
                for (int column = 0; column < weightsLayerK.getColumnDimension(); column++) {
                    double weight = weightsLayerK.getEntry(row, column);
                    weight += eta * outputsLayerK.getEntry(column) * deltasLayerK1.getEntry(row);
                    weightsLayerK.setEntry(row, column, weight);
                }
                double bias = biasesPerLayerK.getEntry(row, 0);
                bias += eta * deltasLayerK1.getEntry(row);
                biasesPerLayerK.setEntry(row, 0, bias);
            }
        }
    }

    public void statistics() {
        int numberOfSamples = dataset.numberOfSamples();
        int goodCounter = 0;
        int badCounter = 0;
        List<String> badInputs = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < numberOfSamples; i++) {
            Sample sample = dataset.getSample(i);
            double[] inputs = sample.getInputs();
            double[] expectedOutputs = sample.getOutputs();
            double[] actualOutputs = feedForward(inputs);
            for (int j = 0; j < actualOutputs.length; j++) {
                actualOutputs[j] = actualOutputs[j] < 0.5 ? 0.0 : 1.0;
            }
            if (Arrays.equals(expectedOutputs, actualOutputs)) goodCounter++;
            else {
                badCounter++;
                sb.append("Ulazi:").append(Arrays.toString(inputs));
                sb.append("\n");
                sb.append("Očekivani izlazi:").append(Arrays.toString(expectedOutputs));
                sb.append("\n");
                sb.append("Dobiveni izlazi: ").append(Arrays.toString(actualOutputs));
                sb.append("\n");
                badInputs.add(sb.toString());
                sb.setLength(0);
            }

        }
        sb.append("\n");
        sb.append("Dobri izlazi:").append(goodCounter);
        sb.append("\n");
        sb.append("Loši izlazi: ").append(badCounter);
        sb.append("\n");
        System.out.println(sb.toString());

        System.out.println("Krivi rezultati za ulaze:");
        badInputs.forEach(System.out::println);
    }

}

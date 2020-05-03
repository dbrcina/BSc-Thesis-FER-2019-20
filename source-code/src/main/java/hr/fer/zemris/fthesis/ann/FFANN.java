package hr.fer.zemris.fthesis.ann;

import hr.fer.zemris.fthesis.ann.afunction.ActivationFunction;
import hr.fer.zemris.fthesis.ann.dataset.ReadOnlyDataset;
import hr.fer.zemris.fthesis.ann.dataset.model.Sample;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Feed forward artificial neural network <i>(Multilayer perceptron)</i>.
 */
public class FFANN {

    private final Random rand = new Random();

    private final int[] layers;
    private final ActivationFunction activationFunction;
    private final ReadOnlyDataset dataset;

    // first dimension represents i-th layer
    private double[][][] weightsPerLayer;
    private double[][] biasesPerLayer;
    private double[][] outputsPerLayer;     // activationFunction.valueAt(net)
    private double[][] derivativesPerLayer; // activationFunction.derivativeValueAt(net)
    private double[][] deltasPerLayer;

    // constructor
    public FFANN(int[] layers, ActivationFunction activationFunction, ReadOnlyDataset dataset) {
        this.layers = layers;
        this.activationFunction = activationFunction;
        this.dataset = dataset;
        initializeMatrices();
    }

    // allocates memory for matrices
    private void initializeMatrices() {
        int numberOfLayers = layers.length;
        weightsPerLayer = new double[numberOfLayers - 1][][];
        biasesPerLayer = new double[numberOfLayers - 1][];
        outputsPerLayer = new double[numberOfLayers][];
        derivativesPerLayer = new double[numberOfLayers - 1][];
        deltasPerLayer = new double[numberOfLayers - 1][];
        for (int i = 0; i < numberOfLayers; i++) {
            if (i != numberOfLayers - 1) {
                weightsPerLayer[i] = new double[layers[i + 1]][layers[i]];
                biasesPerLayer[i] = new double[layers[i + 1]];
                derivativesPerLayer[i] = new double[layers[i + 1]];
                deltasPerLayer[i] = new double[layers[i + 1]];
            }
            outputsPerLayer[i] = new double[layers[i]];
        }
        randomizeWeightsBiases();
    }

    // randomizes weights and biases using [-2.4/m, 2.4m] rule
    // where m is number of inputs for neuron
    private void randomizeWeightsBiases() {
        for (int k = 0; k < layers.length - 1; k++) {
            double[][] weightsLayerK = weightsPerLayer[k];
            double[] biasesLayerK = biasesPerLayer[k];
            int m = k == 0 ? 1 : weightsLayerK[0].length + 1;
            for (int row = 0; row < weightsLayerK.length; row++) {
                for (int column = 0; column < weightsLayerK[0].length; column++) {
                    weightsLayerK[row][column] = -2.4 / m + rand.nextDouble() * 4.8 / m;
                }
                biasesLayerK[row] = -2.4 + rand.nextDouble() * 4.8;
            }
        }
    }

    /**
     * Feed forwards provided <i>inputs</i> and returns result as a 2-dimensional array where first
     * dimension is a number of layer and other is outputs.
     *
     * @param inputs inputs.
     * @return results as 2D array.
     * @throws IllegalArgumentException if number of input elements doesn't fit.
     */
    public double[][] feedForward(double[] inputs) {
        if (inputs.length != layers[0]) {
            throw new IllegalArgumentException(String.format(
                    "Expected input of %d elements but received %d.",
                    layers[0], inputs.length));
        }
        outputsPerLayer[0] = inputs;
        for (int k = 0; k < layers.length - 1; k++) {
            double[][] weightsLayerK = weightsPerLayer[k];
            double[] biasesLayerK = biasesPerLayer[k];
            double[] outputsLayerK = outputsPerLayer[k];
            double[] outputsLayerK1 = outputsPerLayer[k + 1];
            double[] derivativesLayerK = derivativesPerLayer[k];
            for (int row = 0; row < weightsLayerK.length; row++) {
                double net = 0.0;
                for (int column = 0; column < weightsLayerK[0].length; column++) {
                    net += weightsLayerK[row][column] * outputsLayerK[column];
                }
                net += biasesLayerK[row];
                outputsLayerK1[row] = activationFunction.valueAt(net);
                derivativesLayerK[row] = activationFunction.derivativeValueAt(net);
            }
        }
        return outputsPerLayer;
    }

    public void train(int iterLimit, double maxError, double eta) {
        // randomize biases and weights
        randomizeWeightsBiases();
        // start iterations
        for (int iter = 0; iter < iterLimit; iter++) {
            double error = 0.0;
            // for every sample..
            for (int i = 0; i < dataset.numberOfSamples(); i++) {
                Sample sample = dataset.getSample(i);
                // feed forward, calculate outputs for every layer
                feedForward(sample.getInputs());
                // calculate error
                double[] expectedOutputs = sample.getOutputs();
                double[] actualOutputs = outputsPerLayer[outputsPerLayer.length - 1];
                for (int j = 0; j < expectedOutputs.length; j++) {
                    error += Math.pow(expectedOutputs[j] - actualOutputs[j], 2);
                }
                // calculate deltas for every layer
                deltasPerLayer(sample.getOutputs());
                // update weights and biases
                for (int k = 0; k < weightsPerLayer.length; k++) {
                    double[][] weightsLayerK = weightsPerLayer[k];
                    double[] biasesLayerK = biasesPerLayer[k];
                    double[] outputsLayerK = outputsPerLayer[k];
                    double[] deltasLayerK = deltasPerLayer[k];
                    for (int r = 0; r < weightsLayerK.length; r++) {
                        for (int c = 0; c < weightsLayerK[0].length; c++) {
                            weightsLayerK[r][c] += eta * outputsLayerK[c] * deltasLayerK[r];
                        }
                        biasesLayerK[r] += eta * deltasLayerK[r];
                    }
                }
            }
            error /= (2 * dataset.numberOfSamples());
            System.out.println("Iter " + (iter + 1) + ", error = " + error);
            if (error <= maxError) {
                System.out.println("Iter " + (iter + 1) + ", error = " + error);
                System.out.println("Exiting...");
                break;
            }
        }
    }

    private void deltasPerLayer(double[] expectedOutputs) {
        // calculate deltas for output layer
        double[] deltasOutputLayer = deltasPerLayer[deltasPerLayer.length - 1];
        double[] outputsOutputLayer = outputsPerLayer[outputsPerLayer.length - 1];
        double[] derivativesOutputLayer = derivativesPerLayer[derivativesPerLayer.length - 1];
        for (int i = 0; i < deltasOutputLayer.length; i++) {
            double ti = expectedOutputs[i];
            double oi = outputsOutputLayer[i];
            deltasOutputLayer[i] = derivativesOutputLayer[i] * (ti - oi);
        }
        // calculate deltas for hidden layers
        for (int k = deltasPerLayer.length - 2; k >= 0; k--) {
            double[][] weightsLayerK = weightsPerLayer[k + 1];
            double[] biasesLayerK = biasesPerLayer[k + 1];
            double[] derivativesLayerK = derivativesPerLayer[k];
            double[] deltasLayerK = deltasPerLayer[k];
            double[] deltasLayerK1 = deltasPerLayer[k + 1];
            for (int i = 0; i < deltasLayerK.length; i++) {
                // i-th column
                int column = i;
                double[] wi = Arrays.stream(weightsLayerK)
                        .mapToDouble(row -> row[column])
                        .toArray();
                double net = 0.0;
                for (int d = 0; d < deltasLayerK1.length; d++) {
                    net += (wi[d] + biasesLayerK[d]) * deltasLayerK1[d];
                }
                deltasLayerK[i] = derivativesLayerK[i] * net;
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
            feedForward(inputs);
            double[] actualOutputs = outputsPerLayer[outputsPerLayer.length - 1];
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

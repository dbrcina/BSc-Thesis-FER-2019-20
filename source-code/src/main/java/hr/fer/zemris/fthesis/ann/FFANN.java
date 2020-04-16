package hr.fer.zemris.fthesis.ann;

import hr.fer.zemris.fthesis.afunction.ActivationFunction;
import hr.fer.zemris.fthesis.dataset.ReadOnlyDataset;

import java.util.Map;
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
    // other two a real matrix
    private double[][][] weightsPerLayer;
    private double[][][] biasesPerLayer;
    private double[][][] outputsPerLayer;

    public FFANN(int[] layers, ActivationFunction activationFunction, ReadOnlyDataset dataset) {
        this.layers = layers;
        this.activationFunction = activationFunction;
        this.dataset = dataset;
        initializeMatrices();
    }

    // allocates memory for matrices
    private void initializeMatrices() {
        int numberOfLayers = layers.length;
        weightsPerLayer = new double[numberOfLayers - 1][][]; // ignore input layer, [1:](python)
        biasesPerLayer = new double[numberOfLayers - 1][][];  // ignore input layer, [1:](python)
        outputsPerLayer = new double[numberOfLayers][][];
        for (int layer = 0; layer < numberOfLayers; layer++) {
            if (layer != numberOfLayers - 1) {
                weightsPerLayer[layer] = new double[layers[layer + 1]][layers[layer]];
                biasesPerLayer[layer] = new double[layers[layer + 1]][1];
            }
            outputsPerLayer[layer] = new double[layers[layer]][1];
        }
        randomizeWeightsBiases();
    }

    // randomizes weights and biases using normal distribution
    // with mean 0 and variance 1
    private void randomizeWeightsBiases() {
        for (int k = 0; k < layers.length - 1; k++) {
            double[][] weightsLayerK = weightsPerLayer[k];
            double[][] biasesLayerK = biasesPerLayer[k];
            // number of rows is the same
            // biasesLayerK has single column
            for (int row = 0; row < weightsLayerK.length; row++) {
                for (int column = 0; column < weightsLayerK[0].length; column++) {
                    if (column == 0) biasesLayerK[row][0] = rand.nextGaussian();
                    weightsLayerK[row][column] = rand.nextGaussian();
                }
            }
        }
    }

    /**
     * Feed forwards provided <i>input</i> and returns result as a 3-dimensional array where first
     * dimension is a number of layer and other two are matrices.
     *
     * @param inputs inputs.
     * @return results as 3D array.
     * @throws IllegalArgumentException if number of input elements doesn't fit.
     */
    public double[][][] feedForward(double[] inputs) {
        if (inputs.length != layers[0]) {
            throw new IllegalArgumentException(String.format(
                    "Expected input of %d elements but received %d.",
                    layers[0], inputs.length));
        }
        for (int row = 0; row < inputs.length; row++) {
            outputsPerLayer[0][row][0] = inputs[row];
        }
        for (int k = 0; k < layers.length - 1; k++) {
            double[][] weightsLayerK = weightsPerLayer[k];
            double[][] biasesLayerK = biasesPerLayer[k];
            double[][] outputsLayerK = outputsPerLayer[k];
            double[][] outputsLayerK1 = outputsPerLayer[k + 1];
            for (int row = 0; row < weightsLayerK.length; row++) {
                double net = 0.0;
                for (int column = 0; column < weightsLayerK[0].length; column++) {
                    net += weightsLayerK[row][column] * outputsLayerK[column][0];
                    net += biasesLayerK[row][0];
                }
                outputsLayerK1[row][0] = activationFunction.valueAt(net);
            }
        }
        return outputsPerLayer;
    }

    public void train(int iterLimit, double maxError, double eta) {
        // randomize biases and weights
        randomizeWeightsBiases();
        // start iterations
        for (int iter = 0; iter < iterLimit; iter++) {
            // for every sample..
            for (int i = 0; i < dataset.numberOfSamples(); i++) {
                Map.Entry<double[], double[]> sample = dataset.inputsOutputsPair(i);
                // feed forward, calculate outputs for every layer
                feedForward(sample.getKey());
                // calculate deltas for every layer
                double[][][] deltasPerLayer = deltasPerLayer(sample.getValue());

            }
        }
    }

    private double[][][] deltasPerLayer(double[] expectedOutputs) {

    }

    /*private RealMatrix[] errorsPerLayer(double[] expectedOutput) {
        RealMatrix[] errorsPerLayer = new RealMatrix[layers.length - 1];
        // first find errors for output layer
        double[] actualOutput = outputsPerLayer[outputsPerLayer.length - 1].getColumn(0);
        double[] errorsOutputLayer = new double[expectedOutput.length];
        for (int i = 0; i < errorsOutputLayer.length; i++) {
            double deriveValueAt = activationFunction.derivativeValueAt(actualOutput[i]);
            double delta = expectedOutput[i] - actualOutput[i];
            errorsOutputLayer[i] = deriveValueAt * delta;
        }
        errorsPerLayer[errorsPerLayer.length - 1] = new Array2DRowRealMatrix(errorsOutputLayer);

        // back propagate towards other layers
        for (int k = outputsPerLayer.length - 2; k >= 0; k--) {
            // apply derived activation function to outputs in layer k
            RealMatrix derOutputLayerK = outputsPerLayer[k].copy().transpose();
            for (int r = 0; r < derOutputLayerK.getRowDimension(); r++) {
                for (int c = 0; c < derOutputLayerK.getColumnDimension(); c++) {
                    derOutputLayerK.setEntry(r, c, activationFunction.valueAt(derOutputLayerK.getEntry(r, c)));
                }
            }
            double[] outputsLayerK = outputsPerLayer[k].getColumn(0);
            double[] deltasLayerK = new double[outputsLayerK.length];
            double[] deltasLayerK1 = errorsPerLayer[k + 1].getColumn(0);
            double[] biasesLayerK1 = biasesPerLayer[k + 1].getColumn(0);
            for (int i = 0; i < deltasLayerK.length; i++) {
                double deriveValueAt = activationFunction.derivativeValueAt(outputsLayerK[i]);
                double[] weightsID = weightsPerLayer[k + 1].getColumn(i);
                double temp = 0.0;
                for (int d = 0; d < weightsID.length; d++) {
                    temp += (weightsID[d] + biasesLayerK1[d]) * deltasLayerK1[d];
                }
                deltasLayerK[i] = deriveValueAt * temp;
                errorsPerLayer[k] = new Array2DRowRealMatrix(deltasLayerK);
            }
            // delta(k+1) * weights
            //RealMatrix temp = errorsPerLayer[k + 1].transpose()
            // .multiply(weightsPerLayer[k + 1]);
            // Yi * delta * weights
            *//*for (int r = 0; r < derOutputLayerK.getRowDimension(); r++) {
                for (int c = 0; c < derOutputLayerK.getColumnDimension(); c++) {
                    temp.setEntry(r, c, derOutputLayerK.getEntry(r, c) * temp.getEntry(r, c));
                }
            }
            errorsPerLayer[k] = temp.transpose();*//*
        }
        return errorsPerLayer;
    }*/

    /*public void statistics() {
        int numberOfSamples = dataset.numberOfSamples();
        int goodCounter = 0;
        int badCounter = 0;
        List<String> badInputs = new ArrayList<>();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < numberOfSamples; i++) {
            Entry<double[], double[]> sample = dataset.inputsOutputsPair(i);
            double[] inputs = sample.getKey();
            double[] expectedOutputs = sample.getValue();
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
    }*/

}

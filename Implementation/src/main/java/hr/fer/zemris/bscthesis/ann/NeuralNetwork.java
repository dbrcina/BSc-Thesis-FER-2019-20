package hr.fer.zemris.bscthesis.ann;

import hr.fer.zemris.bscthesis.ann.afunction.ActivationFunction;
import hr.fer.zemris.bscthesis.ann.afunction.Softmax;
import hr.fer.zemris.bscthesis.dataset.Dataset;
import hr.fer.zemris.bscthesis.dataset.Sample;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import javax.swing.*;
import java.util.*;

/**
 * Feed forward neural network <i>(Multilayer perceptron)</i> that uses <b>Backpropagation algorithm</b> as a
 * learning algorithm.
 * <br>
 * Default learning type is <b>ONLINE</b>. If the learning type is set to <b>MINI-BATCH</b>, then
 * <code>batchSize</code> needs to be defined, otherwise default value is set to 5.
 *
 * @author dbrcina
 * @see LearningType
 */
public class NeuralNetwork {

    /**
     * Enum that models learning types of Backpropagation algorithm. Valid types are:
     * <ul>
     *     <li>ONLINE -- Stochastic Gradient Descent.</li>
     *     <li>BATCH -- Batch Gradient Descent.</li>
     *     <li>MINI_BATCH -- Mini-Batch Gradient Descent.</li>
     * </ul>
     */
    public enum LearningType {
        ONLINE("Stochastic"),
        BATCH("Batch"),
        MINI_BATCH("Mini-Batch");

        private final String type;

        LearningType(String type) {
            this.type = type;
        }

        @Override
        public String toString() {
            return type;
        }
    }

    /* ---------- GENERAL PARAMETERS FOR NETWORK ---------- */
    // Input + hidden + output.
    private int[] layers;
    // Used only for the hidden layers.
    private ActivationFunction aFunction;
    // Used only for output layer.
    private ActivationFunction outputAFunction;
    private Dataset dataset;
    private LearningType learningType = LearningType.ONLINE;
    private int batchSize = 5;
    /* ---------------------------------------------------- */

    /* ------- ALL MATRICES USED IN TRAINING PROCESS ------ */
    private RealMatrix[] weightsPerLayer;
    private RealMatrix[] biasesPerLayer;
    private RealMatrix[] outputsPerLayer;
    private RealMatrix[] derivativesPerLayer;
    private RealMatrix[] deltasPerLayer;
    // Next two arrays of matrices are needed for updating certain
    // weights/biases in a memory before the real updates take place.
    // This is very necessary!!!
    private RealMatrix[] weightsUpdatesPerLayer;
    private RealMatrix[] biasesUpdatesPerLayer;
    /* ---------------------------------------------------- */

    /* ----------------- HELPER VARIABLES ----------------- */
    private final Random rand = new Random();
    private boolean matricesRandomized;
    /* ---------------------------------------------------- */

    /* ------------------- CONSTRUCTOR -------------------- */

    /**
     * If this constructor is used, certain setters need to be called. The setter for layers, the setter for activation
     * function and the setter for dataset. Setters for learning type and batch size are optional.
     *
     * @see #setLayers(int[])
     * @see #setAFunction(ActivationFunction)
     * @see #setDataset(Dataset)
     * @see #setLearningType(LearningType)
     * @see #setBatchSize(int)
     */
    public NeuralNetwork() {
    }

    /**
     * Constructor.
     *
     * @param layers    input + hidden + output layers.
     * @param aFunction activation function.
     * @param dataset   dataset.
     * @throws NullPointerException     if <code>null</code> value is provided.
     * @throws IllegalArgumentException if definition of <code>layers</code> is invalid.
     */
    public NeuralNetwork(int[] layers,
                         ActivationFunction aFunction,
                         Dataset dataset) {
        setLayers(layers);
        setAFunction(aFunction);
        setDataset(dataset);
    }
    /* ---------------------------------------------------- */

    /* ---------- SETTERS FOR GENERAL PARAMETERS ---------- */

    /**
     * Setter for layers.
     *
     * @param layers input + hidden + output layers.
     * @throws NullPointerException     if <code>null</code> value is provided.
     * @throws IllegalArgumentException if definition of <code>layers</code> is invalid.
     */
    public void setLayers(int[] layers) {
        this.layers = Objects.requireNonNull(layers,
                "NeuralNetwork::setLayers(int[]) null values are not permitted!");
        if (layers.length == 0) {
            throw new IllegalArgumentException("NeuralNetwork::setLayers(int[]) array has a length of 0!");
        }
        if (Arrays.stream(layers).anyMatch(l -> l < 1)) {
            throw new IllegalArgumentException(
                    "NeuralNetwork::setLayers(int[]) layers needs to have at least one neuron!");
        }
        setupMatrices();
    }

    /**
     * Setter for activation function.
     *
     * @param aFunction activation function.
     * @throws NullPointerException if <code>null</code> value is provided.
     */
    public void setAFunction(ActivationFunction aFunction) {
        this.aFunction = Objects.requireNonNull(aFunction,
                "NeuralNetwork::setAFunction(ActivationFunction) null values are not permitted!");
    }

    /**
     * Setter for dataset.
     *
     * @param dataset dataset.
     * @throws NullPointerException if <code>null</code> value is provided.
     */
    public void setDataset(Dataset dataset) {
        this.dataset = Objects.requireNonNull(dataset,
                "NeuralNetwork::setDataset(Dataset) null values are not permitted!");
    }

    /**
     * Setter for learning type. Default value is {@link LearningType#ONLINE}.
     *
     * @param learningType learning type.
     * @throws NullPointerException if <code>null</code> value is provided.
     */
    public void setLearningType(LearningType learningType) {
        this.learningType = Objects.requireNonNull(learningType,
                "NeuralNetwork::setLearningType(LearningType) null values are not permitted!");
    }

    /**
     * Setter for batch size. Provided {@code batchSize} will be used only if learning type is set to
     * {@link LearningType#MINI_BATCH}. Default value is 5. If provided <code>batchSize</code> is <= 0, it will be set
     * to default value.
     *
     * @param batchSize batch size.
     */
    public void setBatchSize(int batchSize) {
        if (batchSize <= 0) return;
        this.batchSize = batchSize;
    }
    /* ---------------------------------------------------- */

    /* ---------- MEMORY ALLOCATION FOR MATRICES ---------- */
    private void setupMatrices() {
        matricesRandomized = false;
        weightsPerLayer = new RealMatrix[layers.length - 1];
        biasesPerLayer = new RealMatrix[layers.length - 1];
        outputsPerLayer = new RealMatrix[layers.length];
        derivativesPerLayer = new RealMatrix[layers.length - 1];
        deltasPerLayer = new RealMatrix[layers.length - 1];
        weightsUpdatesPerLayer = new RealMatrix[layers.length - 1];
        biasesUpdatesPerLayer = new RealMatrix[layers.length - 1];
        for (int k = 0; k < layers.length; k++) {
            if (k != layers.length - 1) {
                weightsPerLayer[k] = MatrixUtils.createRealMatrix(layers[k + 1], layers[k]);
                biasesPerLayer[k] = MatrixUtils.createColumnRealMatrix(new double[layers[k + 1]]);
                derivativesPerLayer[k] = MatrixUtils.createColumnRealMatrix(new double[layers[k + 1]]);
                deltasPerLayer[k] = MatrixUtils.createColumnRealMatrix(new double[layers[k + 1]]);
                weightsUpdatesPerLayer[k] = MatrixUtils.createRealMatrix(layers[k + 1], layers[k]);
                biasesUpdatesPerLayer[k] = MatrixUtils.createColumnRealMatrix(new double[layers[k + 1]]);
            }
            outputsPerLayer[k] = MatrixUtils.createColumnRealMatrix(new double[layers[k]]);
        }
    }
    /* ---------------------------------------------------- */

    /* -------------- XAVIER INITIALIZATION --------------- */
    private void randomizeMatrices() {
        if (!matricesRandomized) {
            matricesRandomized = true;
        }
        for (int k = 0; k < weightsPerLayer.length; k++) {
            RealMatrix weightsLayerK = weightsPerLayer[k];
            RealMatrix biasesLayerK = biasesPerLayer[k];
            // Here we take row dimension because weights and biases matrices
            // have the same row dimension
            for (int i = 0; i < weightsLayerK.getRowDimension(); i++) {
                for (int j = 0; j < weightsLayerK.getColumnDimension(); j++) {
                    double weight = rand.nextGaussian();
                    weight *= Math.sqrt(2.0 / weightsLayerK.getColumnDimension());
                    weightsLayerK.setEntry(i, j, weight);
                }
                double bias = rand.nextGaussian();
                bias *= Math.sqrt(2.0 / weightsLayerK.getColumnDimension());
                // Biases matrices are column matrices, so we use 0 as column index
                biasesLayerK.setEntry(i, 0, bias);
            }
        }
    }
    /* ---------------------------------------------------- */

    /**
     * Feed forwards provided <code>inputs</code> and returns outputs as an array.
     *
     * @param inputs inputs.
     * @return results.
     * @throws NullPointerException     if <code>null</code> value is provided.
     * @throws IllegalArgumentException if number of input elements doesn't fit.
     */
    public double[] feedForward(double[] inputs) {
        Objects.requireNonNull(
                inputs, "NeuralNetwork::feedForward(double[]) null values are not permitted!");
        if (inputs.length != layers[0]) {
            throw new IllegalArgumentException(String.format(
                    "NeuralNetwork::feedForward(double[]) expected input of %d elements but received %d!",
                    layers[0], inputs.length));
        }
        if (!matricesRandomized) {
            randomizeMatrices();
        }
        // Outputs per layer are in column matrix.
        outputsPerLayer[0].setColumn(0, inputs);
        for (int k = 0; k < weightsPerLayer.length; k++) {
            RealMatrix weightsLayerK = weightsPerLayer[k];
            RealMatrix biasesLayerK = biasesPerLayer[k];
            RealMatrix outputsLayerK = outputsPerLayer[k];
            // Column matrix.
            RealMatrix outputsLayerK1 = (weightsLayerK.multiply(outputsLayerK)).add(biasesLayerK);
            double[] weightedSums = outputsLayerK1.getColumn(0);
            boolean isOutputLayer = k == weightsPerLayer.length - 1;
            for (int i = 0; i < weightedSums.length; i++) {
                double weightedSum = weightedSums[i];
                if (isOutputLayer) {
                    outputAFunction = new Softmax(weightedSums);
                }
                // Differentiate hidden layers from output layer!!!
                outputsPerLayer[k + 1].setEntry(i, 0, isOutputLayer ?
                        outputAFunction.value(weightedSum) : aFunction.value(weightedSum));
                derivativesPerLayer[k].setEntry(i, 0, isOutputLayer ?
                        outputAFunction.derivativeValue(weightedSum) : aFunction.derivativeValue(weightedSum));
            }
        }
        return outputsPerLayer[outputsPerLayer.length - 1].getColumn(0);
    }

    /**
     * Performs artificial neural network training.
     *
     * @param epochs   number of epochs.
     * @param maxError maximum error.
     * @param eta      eta constant.
     */
    public void train(int epochs, double maxError, double eta) {
        stop = false;
        System.out.println("Starting " + learningType + " Backpropagation algorithm.");

        // Randomize weights and biases.
        randomizeMatrices();

        // Prepare batches based on the learning type.
        Collection<Collection<Sample>> batches = prepareBatches();
        int numberOfSamples = dataset.numberOfSamples();

        // Start epochs.
        for (int epoch = 0; epoch < epochs && !stop; epoch++) {

            /* Next part is used for continuous updates on GUI */
            if (canvas != null) {
                if ((epoch + 1) % redrawEveryNEpoch == 0) {
                    SwingUtilities.invokeLater(() -> canvas.repaint());
                    try {
                        Thread.sleep(500);
                    } catch (InterruptedException e) {
                        System.out.println("Error occurred while tread was sleeping...");
                    }
                }
            }
            /* ----------------------------------------------- */

            // Variable for accumulating the error.
            double error = 0.0;

            /* Go through every batch */
            for (Collection<Sample> batch : batches) {
                if (stop) break;
                // Reset updates matrices.
                resetUpdates();
                // For every sample:
                for (Sample sample : batch) {
                    if (stop) break;
                    // feed forward sample
                    double[] predictedOutputs = feedForward(sample.getInputs());
                    double[] expectedOutputs = sample.getOutputs();
                    // accumulate error
                    for (int i = 0; i < expectedOutputs.length; i++) {
                        double subtract = expectedOutputs[i] - predictedOutputs[i];
                        error += subtract * subtract;
                    }
                    // Calculate all deltas using Backpropagation algorithm.
                    calculateDeltasPerLayer(expectedOutputs);
                    // Update weights and biases and save to the memory.
                    updateWeightsBiases(eta);
                }
                // Apply updates for weights and biases.
                System.arraycopy(
                        weightsUpdatesPerLayer, 0, weightsPerLayer, 0, weightsUpdatesPerLayer.length);
                System.arraycopy(
                        biasesUpdatesPerLayer, 0, biasesPerLayer, 0, biasesUpdatesPerLayer.length);
            }
            /* ---------------------- */

            /* Check accumulated error and print results */
            if (stop) break;
            error = error / (2 * numberOfSamples);
            boolean exit = error < maxError;
            if (epoch == 0 || exit || (epoch + 1) % 1000 == 0) {
                System.out.println("Epoch " + (epoch + 1) + "., error = " + error);
                if (exit) {
                    System.out.println("Found closest error! Exiting...");
                    break;
                }
            }
            /* ----------------------------------------- */
        }

        if (stop) {
            System.out.println("Stopped.");
        }
    }

    /* PREPARE BATCHES OF SAMPLES BASED ON THE LEARNING TYPE */
    private Collection<Collection<Sample>> prepareBatches() {
        Collection<Collection<Sample>> batches = new ArrayList<>();
        List<Sample> samples = dataset.shuffleSamples();
        if (learningType == LearningType.BATCH) {
            batches.add(samples);
        } else if (learningType == LearningType.ONLINE) {
            samples.forEach(s -> batches.add(List.of(s)));
        } else {
            int numberOfBatches = samples.size() / batchSize;
            int offset = 0;
            for (int i = 0; i < numberOfBatches; i++) {
                List<Sample> batch = new ArrayList<>();
                do {
                    batch.add(samples.get(offset++));
                } while (offset % batchSize != 0);
                batches.add(batch);
            }
            int numOfSamplesLeft = samples.size() % batchSize;
            if (numOfSamplesLeft != 0) {
                List<Sample> samplesLeft = new ArrayList<>();
                for (int i = offset; i < samples.size(); i++) {
                    samplesLeft.add(samples.get(i));
                }
                batches.add(samplesLeft);
            }
        }
        return batches;
    }
    /* ---------------------------------------------------- */

    /* -------------- RESETS UPDATE MATRICES -------------- */
    private void resetUpdates() {
        for (int k = 0; k < weightsUpdatesPerLayer.length; k++) {
            weightsUpdatesPerLayer[k] = weightsPerLayer[k].copy();
            biasesUpdatesPerLayer[k] = biasesPerLayer[k].copy();
        }
    }
    /* ---------------------------------------------------- */

    /* --- CALCULATE DELTAS - BACKPROPAGATION ALGORITHM --- */
    private void calculateDeltasPerLayer(double[] expectedOutputs) {
        // Calculate deltas for output layer.
        RealVector target = new ArrayRealVector(expectedOutputs);
        RealVector actual = outputsPerLayer[outputsPerLayer.length - 1].getColumnVector(0);
        RealVector subtraction = target.subtract(actual);
        // Apply derivatives.
        RealVector derivativesOutputLayer = derivativesPerLayer[deltasPerLayer.length - 1].getColumnVector(0);
        for (int i = 0; i < subtraction.getDimension(); i++) {
            subtraction.setEntry(i, derivativesOutputLayer.getEntry(i) * subtraction.getEntry(i));
        }
        deltasPerLayer[deltasPerLayer.length - 1].setColumnVector(0, subtraction);
        // Calculate deltas for hidden layers.
        for (int k = deltasPerLayer.length - 2; k >= 0; k--) {
            RealVector derivativesLayerK = derivativesPerLayer[k].getColumnVector(0);
            RealMatrix weightsLayerK1 = weightsPerLayer[k + 1];
            RealVector biasesLayerK1 = biasesPerLayer[k + 1].getColumnVector(0);
            RealMatrix deltasLayerK1 = deltasPerLayer[k + 1];
            RealMatrix deltasLayerK = deltasPerLayer[k];
            double[] weightedSums = (weightsLayerK1.transpose()).multiply(deltasLayerK1).getColumn(0);
            // Add biases.
            for (int i = 0; i < weightedSums.length; i++) {
                for (int j = 0; j < biasesLayerK1.getDimension(); j++) {
                    weightedSums[i] += biasesLayerK1.getEntry(j) * deltasLayerK1.getEntry(j, 0);
                }
            }
            // Apply derivatives.
            for (int i = 0; i < deltasLayerK.getRowDimension(); i++) {
                deltasLayerK.setEntry(i, 0, derivativesLayerK.getEntry(i) * weightedSums[i]);
            }
        }
    }
    /* ---------------------------------------------------- */

    /* ---- UPDATE WEIGHTS AND BIASES AFTER ONE SAMPLE ---- */
    private void updateWeightsBiases(double eta) {
        for (int k = 0; k < weightsUpdatesPerLayer.length; k++) {
            RealMatrix updatesWeightsLayerK = weightsUpdatesPerLayer[k];
            RealMatrix updatesBiasesLayerK = biasesUpdatesPerLayer[k];
            RealVector outputsLayerK = outputsPerLayer[k].getColumnVector(0);
            RealVector deltasLayerK1 = deltasPerLayer[k].getColumnVector(0);
            for (int i = 0; i < updatesWeightsLayerK.getRowDimension(); i++) {
                for (int j = 0; j < updatesWeightsLayerK.getColumnDimension(); j++) {
                    double weight = updatesWeightsLayerK.getEntry(i, j);
                    weight += eta * outputsLayerK.getEntry(j) * deltasLayerK1.getEntry(i);
                    updatesWeightsLayerK.setEntry(i, j, weight);
                }
                double bias = updatesBiasesLayerK.getEntry(i, 0);
                bias += eta * deltasLayerK1.getEntry(i);
                updatesBiasesLayerK.setEntry(i, 0, bias);
            }
        }
    }
    /* ---------------------------------------------------- */

    /* ------------ USED FOR GUI VISUALISATION ------------ */
    private volatile boolean stop;
    private JComponent canvas;
    private int redrawEveryNEpoch = -1;

    public void stop() {
        stop = true;
    }

    public void setCanvas(JComponent canvas) {
        this.canvas = canvas;
    }

    public void setRedrawEveryNEpoch(int redrawEveryNEpoch) {
        this.redrawEveryNEpoch = redrawEveryNEpoch;
    }
    /* ---------------------------------------------------- */

}

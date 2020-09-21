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
import java.util.function.Function;

/**
 * Feed forward neural network <i>(Multilayer perceptron)</i> that uses <b>Backpropagation</b> as a
 * learning algorithm.
 */
public class NeuralNetwork {

    /**
     * Enum that models learning types of backpropagation algorithm. Valid types are:
     * <ul>
     *     <li>ONLINE -- Stochastic Gradient Descent</li>
     *     <li>BATCH -- Batch Gradient Descent</li>
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

    /* GENERAL PARAMETERS FOR NETWORK */
    private int[] layers;
    private ActivationFunction aFunction;
    private Dataset dataset;
    private LearningType learningType = LearningType.ONLINE;
    private int batchSize = 5;
    /* ------------------------------ */

    /* ALL MATRICES USED IN TRAINING PROCESS */
    private RealMatrix[] weightsPerLayer;
    private RealMatrix[] biasesPerLayer;
    private RealMatrix[] outputsPerLayer;
    private RealMatrix[] derivativesPerLayer;
    private RealMatrix[] deltasPerLayer;
    private RealMatrix[] weightsUpdatesPerLayer;
    private RealMatrix[] biasesUpdatesPerLayer;
    /* ------------------------------------- */

    /* HELPER VARIABLES */
    private final Random rand = new Random();
    private boolean matricesRandomized;
    /* ---------------- */

    /* CONSTRUCTOR */
    public NeuralNetwork() {
    }

    public NeuralNetwork(int[] layers, ActivationFunction aFunction, Dataset dataset) {
        this.layers = layers;
        this.aFunction = aFunction;
        this.dataset = dataset;
        setupMatrices();
    }
    /* ----------- */

    /* SETTERS FOR GENERAL PARAMETERS */
    public void setLayers(int[] layers) {
        this.layers = layers;
        setupMatrices();
    }

    public void setAFunction(ActivationFunction aFunction) {
        this.aFunction = aFunction;
    }

    public void setDataset(Dataset dataset) {
        this.dataset = dataset;
    }

    /**
     * Setter for learning type. Default value is {@link LearningType#ONLINE}.
     *
     * @param learningType learning type.
     */
    public void setLearningType(LearningType learningType) {
        this.learningType = learningType;
    }

    /**
     * Setter for batch size. Provided {@code batchSize} will be used only if learning type is set to
     * {@link LearningType#MINI_BATCH}. Default value is 5.
     *
     * @param batchSize batch size.
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    /* -------------------------------- */

    /* MEMORY ALLOCATION FOR MATRICES */
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
    /* ------------------------------- */

    /* XAVIER INITIALIZATION */
    private void randomizeMatrices() {
        if (!matricesRandomized) {
            matricesRandomized = true;
        }
        for (int k = 0; k < weightsPerLayer.length; k++) {
            RealMatrix weightsLayerK = weightsPerLayer[k];
            RealMatrix biasesLayerK = biasesPerLayer[k];
            for (int i = 0; i < weightsLayerK.getRowDimension(); i++) {
                for (int j = 0; j < weightsLayerK.getColumnDimension(); j++) {
                    double weight = rand.nextGaussian();
                    weight *= Math.sqrt(2.0 / weightsLayerK.getColumnDimension());
                    weightsLayerK.setEntry(i, j, weight);
                }
                double bias = rand.nextGaussian();
                bias *= Math.sqrt(2.0 / weightsLayerK.getColumnDimension());
                biasesLayerK.setEntry(i, 0, bias);
            }
        }
    }
    /* --------------------- */

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
        outputsPerLayer[0].setColumn(0, inputs);
        for (int k = 0; k < weightsPerLayer.length; k++) {
            RealMatrix weightsLayerK = weightsPerLayer[k];
            RealMatrix biasesLayerK = biasesPerLayer[k];
            RealMatrix outputsLayerK = outputsPerLayer[k];
            RealMatrix outputsLayerK1 = (weightsLayerK.multiply(outputsLayerK)).add(biasesLayerK);
            double[] weightedSums = outputsLayerK1.getColumn(0);
            if (k != weightsPerLayer.length - 1) {
                // apply activation function to hidden layers
                outputsPerLayer[k + 1].setColumn(
                        0, Arrays.stream(weightedSums).map(aFunction::value).toArray());
                derivativesPerLayer[k].setColumn(
                        0, Arrays.stream(weightedSums).map(aFunction::derivativeValue).toArray());
            } else {
                // apply softmax to outputs
                ActivationFunction softmax = new Softmax(weightedSums);
                for (int i = 0; i < weightedSums.length; i++) {
                    outputsPerLayer[k + 1].setEntry(i, 0, softmax.value(weightedSums[i]));
                    derivativesPerLayer[k].setEntry(i, 0, softmax.derivativeValue(weightedSums[i]));
                }
//                outputsPerLayer[k + 1].setColumn(0, SOFTMAX.apply(weightedSums));
//                derivativesPerLayer[k].setColumn(0, SOFTMAX_DERIVATIVE.apply(weightedSums));
            }
        }
        return outputsPerLayer[outputsPerLayer.length - 1].getColumn(0);
    }

    public void train(int epochs, double maxError, double eta) {
        stop = false;
        /*PrintWriter writer = null;
        try {
            writer = new PrintWriter(
                    new File(learningType + "_" + aFunction + "_" + Arrays.toString(layers)) + ".csv");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }*/
        System.out.println("Starting " + learningType + " backpropagation.");

        // randomize weights and biases
        randomizeMatrices();

        // prepare batches based on learning type
        Collection<Collection<Sample>> batches = prepareBatches();
        int numberOfSamples = dataset.numberOfSamples();

        // start epochs
        for (int epoch = 0; epoch < epochs && !stop; epoch++) {

            /* next part is used for continuous updates on GUI */
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

            // variable for accumulating the error
            double error = 0.0;

            /* go through every batch */
            for (Collection<Sample> batch : batches) {
                if (stop) break;
                // reset updates matrices
                resetUpdates();
                // for every sample
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
                    // calculate all deltas using backpropagation
                    calculateDeltasPerLayer(expectedOutputs);
                    // update weights and biases and save to memory
                    updateWeightsBiases(eta);
                }
                // apply updates for weights and biases
                System.arraycopy(
                        weightsUpdatesPerLayer, 0, weightsPerLayer, 0, weightsUpdatesPerLayer.length);
                System.arraycopy(
                        biasesUpdatesPerLayer, 0, biasesPerLayer, 0, biasesUpdatesPerLayer.length);
            }
            /* ---------------------- */
            //if (epoch % 25 == 0) writer.println(epoch + "," + error);
            /* check accumulated error and print results */
            if (stop) break;
            error = error / (2 * numberOfSamples);
            boolean exit = error < maxError;
            if (epoch == 0 || exit || (epoch + 1) % 1000 == 0) {
                System.out.println("Epoch " + (epoch + 1) + "., error = " + error);
                if (exit) {
                    //writer.println(epoch + "," + error);
                    System.out.println("Found closest error! Exiting...");
                    break;
                }
            }
            /* ----------------------------------------- */
        }

        if (stop) {
            System.out.println("Stopped.");
        }
        //writer.flush();
    }

    /* PREPARE BATCHES OF SAMPLES BASED ON LEARNING TYPE */
    private Collection<Collection<Sample>> prepareBatches() {
        Collection<Collection<Sample>> batches = new LinkedList<>();
        List<Sample> samples = dataset.shuffleSamples();
        if (learningType == LearningType.BATCH) {
            batches.add(samples);
        } else if (learningType == LearningType.ONLINE) {
            samples.forEach(s -> batches.add(List.of(s)));
        } else {
            int numberOfBatches = samples.size() / batchSize;
            int offset = 0;
            for (int i = 0; i < numberOfBatches; i++) {
                List<Sample> batch = new LinkedList<>();
                do {
                    batch.add(samples.get(offset++));
                } while (offset % batchSize != 0);
                batches.add(batch);
            }
            int numOfSamplesLeft = samples.size() % batchSize;
            if (numOfSamplesLeft != 0) {
                List<Sample> samplesLeft = new LinkedList<>();
                for (int i = offset; i < samples.size(); i++) {
                    samplesLeft.add(samples.get(i));
                }
                batches.add(samplesLeft);
            }
        }
        return batches;
    }
    /* -------------------------------------------------- */

    /* RESETS UPDATE MATRICES */
    private void resetUpdates() {
        for (int k = 0; k < weightsUpdatesPerLayer.length; k++) {
            weightsUpdatesPerLayer[k] = weightsPerLayer[k].copy();
            biasesUpdatesPerLayer[k] = biasesPerLayer[k].copy();
        }
    }
    /* ---------------------- */

    /* CALCULATE DELTAS - BACKPROPAGATION ALGORITHM */
    private void calculateDeltasPerLayer(double[] expectedOutputs) {
        // calculate deltas for output layer
        RealVector target = new ArrayRealVector(expectedOutputs);
        RealVector actual = outputsPerLayer[outputsPerLayer.length - 1].getColumnVector(0);
        RealVector subtraction = target.subtract(actual);
        // apply derivatives
        RealVector derivativesOutputLayer = derivativesPerLayer[deltasPerLayer.length - 1]
                .getColumnVector(0);
        for (int i = 0; i < subtraction.getDimension(); i++) {
            subtraction.setEntry(i, derivativesOutputLayer.getEntry(i) * subtraction.getEntry(i));
        }
        deltasPerLayer[deltasPerLayer.length - 1].setColumnVector(0, subtraction);
        // ------------------ //
        // calculate deltas for hidden layers
        for (int k = deltasPerLayer.length - 2; k >= 0; k--) {
            RealVector derivativesLayerK = derivativesPerLayer[k].getColumnVector(0);
            RealMatrix weightsLayerK1 = weightsPerLayer[k + 1];
            RealVector biasesLayerK1 = biasesPerLayer[k + 1].getColumnVector(0);
            RealMatrix deltasLayerK1 = deltasPerLayer[k + 1];
            RealMatrix deltasLayerK = deltasPerLayer[k];
            double[] weightedSums = (weightsLayerK1.transpose().multiply(deltasLayerK1)).getColumn(0);
            // add biases
            /*for (int i = 0; i < weightedSums.length; i++) {
                for (int j = 0; j < biasesLayerK1.getDimension(); j++) {
                    weightedSums[i] += biasesLayerK1.getEntry(j) * deltasLayerK1.getEntry(j, 0);
                }
            }*/
            // apply derivatives
            for (int i = 0; i < deltasLayerK.getRowDimension(); i++) {
                deltasLayerK.setEntry(i, 0, derivativesLayerK.getEntry(i) * weightedSums[i]);
            }
        }
    }
    /* -------------------------------------------- */

    /* UPDATE WEIGHTS AND BIASES AFTER ONE SAMPLE */
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
    /* --------------------------------------------------- */

    /* SOFTMAX FUNCTION AND ITS DERIVATIVE - USED FOR OUTPUT LAYER */
    private static final Function<double[], double[]> SOFTMAX = zVector -> {
        double[] tVector = Arrays.stream(zVector).map(Math::exp).toArray();
        double sum = Arrays.stream(tVector).sum();
        return Arrays.stream(tVector).map(t -> t / sum).toArray();
    };

    private static final Function<double[], double[]> SOFTMAX_DERIVATIVE =
            zVector -> Arrays.stream(SOFTMAX.apply(zVector)).map(y -> y * (1 - y)).toArray();
    /* ----------------------------------------------------------- */

    /* USED FOR GUI VISUALISATION */
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
    /* -------------------------- */

}

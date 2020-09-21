package hr.fer.zemris.bscthesis.dataset;

import hr.fer.zemris.bscthesis.classes.ClassType;

/**
 * Models one sample used for training ANN. Each sample consists of:
 * <ul>
 *     <li>inputs,</li>
 *     <li>outputs,</li>
 *     <li>class type - {@link ClassType}.</li>
 * </ul>
 *
 * @author dbrcina
 */
public class Sample {

    private final double[] inputs;
    private final double[] outputs;
    private final ClassType classType;

    /**
     * Constructor.
     *
     * @param inputs    inputs array.
     * @param outputs   outputs array.
     * @param classType class type.
     */
    public Sample(double[] inputs, double[] outputs, ClassType classType) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.classType = classType;
    }

    /**
     * @return samples inputs.
     */
    public double[] getInputs() {
        return inputs;
    }

    /**
     * @return samples outputs.
     */
    public double[] getOutputs() {
        return outputs;
    }

    /**
     * @return samples class type.
     */
    public ClassType getClassType() {
        return classType;
    }

}

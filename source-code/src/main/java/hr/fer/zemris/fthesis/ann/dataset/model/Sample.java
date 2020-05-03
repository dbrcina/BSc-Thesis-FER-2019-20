package hr.fer.zemris.fthesis.ann.dataset.model;

/**
 * Models one sample used for training ANN. Each sample consists of:
 * <ul>
 *     <li>inputs,</li>
 *     <li>outputs,</li>
 *     <li>class type - {@link ClassType}</li>
 * </ul>
 * Initially, class type is set to {@link ClassType#NONE}.
 */
public class Sample {

    private final double[] inputs;
    private final double[] outputs;
    private ClassType classType = ClassType.NONE;

    public Sample(double[] inputs, double[] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public double[] getInputs() {
        return inputs;
    }

    public double[] getOutputs() {
        return outputs;
    }

    public ClassType getClassType() {
        return classType;
    }

    public void setClassType(ClassType classType) {
        this.classType = classType;
    }

}

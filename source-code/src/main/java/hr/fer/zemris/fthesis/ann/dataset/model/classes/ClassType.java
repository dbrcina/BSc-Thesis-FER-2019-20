package hr.fer.zemris.fthesis.ann.dataset.model.classes;

import hr.fer.zemris.fthesis.util.Rectangle2D;

import java.awt.*;
import java.util.Arrays;

/**
 * Each sample class consists of:
 * <ul>
 *     <li>desired outputs,</li>
 *     <li>color.</li>
 * </ul>
 * Every class is represented by unique shape which can be obtained by
 * {@link #createShape(Rectangle2D)} method.
 */
public abstract class ClassType {

    private final double[] outputs;
    private final Color color;

    protected ClassType(double[] outputs, Color color) {
        this.outputs = outputs;
        this.color = color;
    }

    public double[] getOutputs() {
        return outputs;
    }

    public Color getColor() {
        return color;
    }

    /**
     * Creates a shape based on provided rectangle.
     *
     * @param rect rectangle.
     * @return shape.
     */
    public abstract Shape createShape(Rectangle2D rect);

    public abstract String toString();

    public static ClassType fromOutputs(double[] outputs) {
        if (Arrays.equals(outputs, ClassA.OUTPUTS)) {
            return new ClassA();
        } else if (Arrays.equals(outputs, ClassB.OUTPUTS)) {
            return new ClassB();
        } else if (Arrays.equals(outputs, ClassC.OUTPUTS)) {
            return new ClassC();
        } else {
            return null;
        }
    }

}

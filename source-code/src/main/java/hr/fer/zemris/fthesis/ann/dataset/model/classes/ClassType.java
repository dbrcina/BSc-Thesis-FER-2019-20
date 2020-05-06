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
 */
public abstract class ClassType {

    private final double[] outputs;
    private Color color;

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

    private void darkenColorFor(double[] outputs) {
        int r = (int) (outputs[0] * 255);
        int g = (int) (outputs[1] * 255);
        int b = (int) (outputs[2] * 255);
        if (r > 255) r = 255;
        if (g > 255) g = 255;
        if (b > 255) b = 255;
        color = new Color(r, g, b);
    }

    /**
     * Creates a shape based on provided rectangle.
     *
     * @param rect rectangle.
     * @return shape.
     */
    public abstract Shape createShape(Rectangle2D rect);

    @Override
    public abstract String toString();

    /**
     * Determines class type for given <i>outputs</i>.
     *
     * @param outputs outputs.
     * @return class type.
     */
    public static ClassType forOutputs(double[] outputs) {
        double[] modified = Arrays.stream(outputs)
                .map(d -> d = d < 0.5 ? 0 : 1)
                .toArray();
        ClassType classType;
        if (Arrays.equals(modified, ClassA.OUTPUTS)) {
            classType = new ClassA();
        } else if (Arrays.equals(modified, ClassB.OUTPUTS)) {
            classType = new ClassB();
        } else if (Arrays.equals(modified, ClassC.OUTPUTS)) {
            classType = new ClassC();
        } else {
            classType = new ClassNone();
        }
        classType.darkenColorFor(outputs);
        return classType;
    }

}

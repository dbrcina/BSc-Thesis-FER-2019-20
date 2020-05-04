package hr.fer.zemris.fthesis.ann.dataset.model;

import java.awt.*;

/**
 * Simple enum that models sample's class type which has unique color.
 */
public enum ClassType {

    NONE(Color.WHITE),
    CLASS_A(new Color(255, 102, 102), 1, 0, 0),
    CLASS_B(Color.GREEN, 0, 1, 0),
    CLASS_C(new Color(51, 204, 255), 0, 0, 1);

    private final double[] outputs;
    private final Color color;

    ClassType(Color color, double... outputs) {
        this.outputs = outputs;
        this.color = color;
    }

    public double[] getOutputs() {
        if (this == NONE) {
            throw new RuntimeException("Class type 'NONE' doesn't have defined outputs.");
        }
        return outputs;
    }

    public Color getColor() {
        return color;
    }

}

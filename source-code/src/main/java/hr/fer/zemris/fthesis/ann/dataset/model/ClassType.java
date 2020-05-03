package hr.fer.zemris.fthesis.ann.dataset.model;

import java.awt.Color;

/**
 * Simple enum that models sample's class type which has unique color.
 */
public enum ClassType {

    NONE(Color.BLACK),
    CLASS_A(Color.RED),
    CLASS_B(Color.GREEN),
    CLASS_C(Color.BLUE);

    private final Color color;

    ClassType(Color color) {
        this.color = color;
    }

    public Color getColor() {
        return color;
    }

}

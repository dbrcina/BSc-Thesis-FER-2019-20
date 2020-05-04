package hr.fer.zemris.fthesis.ann.dataset.model.classes;

import hr.fer.zemris.fthesis.util.Rectangle2D;

import java.awt.*;
import java.awt.geom.Ellipse2D;

public class ClassB extends ClassType {

    public static final double[] OUTPUTS = {0, 1, 0};

    public ClassB() {
        super(OUTPUTS, Color.GREEN);
    }

    @Override
    public Shape createShape(Rectangle2D rect) {
        return new Ellipse2D.Double(rect.x, rect.y, rect.width, rect.height);
    }

    @Override
    public String toString() {
        return "CLASS_B";
    }

}

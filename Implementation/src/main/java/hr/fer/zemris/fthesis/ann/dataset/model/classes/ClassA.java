package hr.fer.zemris.fthesis.ann.dataset.model.classes;

import hr.fer.zemris.fthesis.util.Rectangle2D;

import java.awt.*;

public class ClassA extends ClassType {

    public static final double[] OUTPUTS = {1, 0, 0};

    public ClassA() {
        super(OUTPUTS, Color.RED);
    }

    @Override
    public Shape createShape(Rectangle2D rect) {
        int[] xPoints = {rect.x, rect.x + rect.width, rect.x + rect.width, rect.x};
        int[] yPoints = {rect.y, rect.y, rect.y + rect.height, rect.y + rect.height};
        int nPoints = 4;
        return new Polygon(xPoints, yPoints, nPoints);
    }

    @Override
    public String toString() {
        return "CLASS_A";
    }

}

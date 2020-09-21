package hr.fer.zemris.fthesis.ann.dataset.model.classes;

import hr.fer.zemris.fthesis.util.Rectangle2D;

import java.awt.*;

public class ClassC extends ClassType {

    public static final double[] OUTPUTS = {0, 0, 1};

    public ClassC() {
        super(OUTPUTS, Color.BLUE);
    }

    @Override
    public Shape createShape(Rectangle2D rect) {
        int[] xPoints = {rect.x, rect.x + rect.width, rect.x + rect.width / 2};
        int[] yPoints = {rect.y, rect.y, rect.y + rect.height};
        int nPoints = 3;
        return new Polygon(xPoints, yPoints, nPoints);
    }

    @Override
    public String toString() {
        return "CLASS_C";
    }

}

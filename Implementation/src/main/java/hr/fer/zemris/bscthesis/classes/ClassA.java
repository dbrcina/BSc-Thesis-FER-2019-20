package hr.fer.zemris.bscthesis.classes;

import java.awt.*;

/**
 * <b>CLASS_A</b> - an implementation of ClassType:
 * <ul>
 *      <li><code>desiredOutputs</code> = [1.0, 0.0, 0.0],</li>
 *      <li><code>color</code>           = Shades of Red,</li>
 *      <li><code>shape</code>           = Rectangle.</li>
 * </ul>
 *
 * @author dbrcina
 */
public class ClassA extends ClassType {

    public ClassA(double[] actualOutputs) {
        super("CLASS_A", actualOutputs, new Polygon());
    }

    public ClassA() {
        this(null);
    }

    @Override
    public Shape createShape(Rectangle rect) {
        Polygon rectangularShape = (Polygon) getShape();
        rectangularShape.xpoints = new int[]{rect.x, rect.x + rect.width, rect.x + rect.width, rect.x};
        rectangularShape.ypoints = new int[]{rect.y, rect.y, rect.y + rect.height, rect.y + rect.height};
        rectangularShape.npoints = 4;
        return rectangularShape;
    }

}

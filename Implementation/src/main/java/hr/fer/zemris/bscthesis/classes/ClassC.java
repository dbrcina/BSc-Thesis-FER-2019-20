package hr.fer.zemris.bscthesis.classes;

import java.awt.*;

/**
 * <b>CLASS_C</b> - an implementation of ClassType:
 * <ul>
 *      <li><code>desiredOutputs</code> = [0.0, 0.0, 1.0],</li>
 *      <li><code>color</code>           = Shades of Blue,</li>
 *      <li><code>shape</code>           = Triangle.</li>
 * </ul>
 *
 * @author dbrcina
 */
public class ClassC extends ClassType {


    public ClassC(double[] actualOutputs) {
        super("CLASS_C", actualOutputs, new Polygon());
    }

    public ClassC() {
        this(null);
    }

    @Override
    public Shape createShape(Rectangle rect) {
        Polygon triangularShape = (Polygon) getShape();
        triangularShape.xpoints = new int[]{rect.x, rect.x + rect.width, rect.x + rect.width / 2};
        triangularShape.ypoints = new int[]{rect.y, rect.y, rect.y + rect.height};
        triangularShape.npoints = 3;
        return triangularShape;
    }

}

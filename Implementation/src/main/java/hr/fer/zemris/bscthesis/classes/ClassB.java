package hr.fer.zemris.bscthesis.classes;

import java.awt.*;
import java.awt.geom.Ellipse2D;

/**
 * <b>CLASS_B</b> - an implementation of ClassType:
 * <ul>
 *      <li><code>desiredOutputs</code> = [0.0, 1.0, 0.0],</li>
 *      <li><code>color</code>           = Shades of Green,</li>
 *      <li><code>shape</code>           = Ellipse.</li>
 * </ul>
 *
 * @author dbrcina
 */
public class ClassB extends ClassType {

    public ClassB(double[] actualOutputs) {
        super("CLASS_B", actualOutputs, new Ellipse2D.Double());
    }

    public ClassB() {
        this(null);
    }

    @Override
    public Shape createShape(Rectangle rect) {
        Ellipse2D.Double ellipticalShape = (Ellipse2D.Double) getShape();
        ellipticalShape.x = rect.x;
        ellipticalShape.y = rect.y;
        ellipticalShape.width = rect.width;
        ellipticalShape.height = rect.height;
        return ellipticalShape;
    }

}

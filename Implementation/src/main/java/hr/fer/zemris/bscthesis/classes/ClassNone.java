package hr.fer.zemris.bscthesis.classes;

import java.awt.*;

/**
 * <b>CLASS_NONE</b> - an implementation of ClassType used when a class type is unknown. <code>color</code> for this
 * kind of class is White.
 *
 * @author dbrcina
 */
public class ClassNone extends ClassType {

    public ClassNone(double[] actualOutputs) {
        super("CLASS_NONE", actualOutputs, null);
    }

    @Override
    public Shape createShape(Rectangle rect) {
        return null;
    }

}

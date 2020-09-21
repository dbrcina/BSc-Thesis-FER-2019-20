package hr.fer.zemris.fthesis.ann.dataset.model.classes;

import hr.fer.zemris.fthesis.util.Rectangle2D;

import java.awt.*;

public class ClassNone extends ClassType {

    public ClassNone() {
        super(new double[0], Color.WHITE);
    }

    @Override
    public Shape createShape(Rectangle2D rect) {
        return null;
    }

    @Override
    public String toString() {
        return null;
    }

}

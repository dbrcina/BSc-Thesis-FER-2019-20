package hr.fer.zemris.bscthesis.demo;

import hr.fer.zemris.bscthesis.ann.afunction.ActivationFunction;
import hr.fer.zemris.bscthesis.classes.ClassType;
import hr.fer.zemris.bscthesis.gui.Window;

import javax.swing.*;

public class Simulation {

    public static void main(String[] args) {
        ClassType.init();
        ActivationFunction.loadAFunctions();
        SwingUtilities.invokeLater(() -> new Window().setVisible(true));
    }

}

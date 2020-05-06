package hr.fer.zemris.fthesis.demo;

import hr.fer.zemris.fthesis.gui.Window;

import javax.swing.*;

public class Simulation {

    public static void main(String[] args) throws Exception {
        SwingUtilities.invokeLater(() -> new Window().setVisible(true));
    }

}

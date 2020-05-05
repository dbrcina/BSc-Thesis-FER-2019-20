package hr.fer.zemris.fthesis.demo;

import hr.fer.zemris.fthesis.IrisDataset;
import hr.fer.zemris.fthesis.ann.NeuralNetwork;
import hr.fer.zemris.fthesis.ann.afunction.ReLU;
import hr.fer.zemris.fthesis.ann.afunction.Sigmoid;
import hr.fer.zemris.fthesis.ann.afunction.Tanh;
import hr.fer.zemris.fthesis.ann.dataset.ReadOnlyDataset;
import hr.fer.zemris.fthesis.gui.Window;

import javax.swing.*;
import java.nio.file.Paths;

public class Simulation {

    public static void main(String[] args) throws Exception {
        SwingUtilities.invokeLater(() -> new Window().setVisible(true));
        /*ReadOnlyDataset dataset = new IrisDataset();
        dataset.loadDataset(Paths.get(args[0]));
        NeuralNetwork nn = new NeuralNetwork(new int[]{4, 5, 3, 3}, new Sigmoid(), dataset);
        nn.train(100000, 0.002, 0.01);
        nn.statistics();*/
    }

}

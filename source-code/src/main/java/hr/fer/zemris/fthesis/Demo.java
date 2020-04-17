package hr.fer.zemris.fthesis;

import hr.fer.zemris.fthesis.afunction.LReLU;
import hr.fer.zemris.fthesis.afunction.Sigmoid;
import hr.fer.zemris.fthesis.afunction.Tanh;
import hr.fer.zemris.fthesis.ann.FFANN;
import hr.fer.zemris.fthesis.dataset.IrisDataset;
import hr.fer.zemris.fthesis.dataset.ReadOnlyDataset;

import java.io.IOException;
import java.nio.file.Paths;

// LReLU : alpha = 0.2, eta = 0.0001
public class Demo {

    public static void main(String[] args) throws IOException {
        ReadOnlyDataset dataset = new IrisDataset();
        dataset.loadDataset(Paths.get(
                "D:/Fer/Vještine/ROPAERUJ/hw07-0036506587/data/07-iris-formatirano.data"));
        FFANN ffann = new FFANN(new int[]{4, 5, 5, 3}, new Sigmoid(), dataset);
        //ffann.feedForward(inputs);
        ffann.train(100000, 0.002, 0.2);
        ffann.statistics();
    }

}

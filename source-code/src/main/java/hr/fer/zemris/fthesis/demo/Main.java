package hr.fer.zemris.fthesis.demo;

import hr.fer.zemris.fthesis.IrisDataset;
import hr.fer.zemris.fthesis.ann.FFANN;
import hr.fer.zemris.fthesis.ann.afunction.Sigmoid;
import hr.fer.zemris.fthesis.ann.dataset.ReadOnlyDataset;

import java.nio.file.Paths;

public class Main {

    public static void main(String[] args) throws Exception {
        ReadOnlyDataset dataset = new IrisDataset();
        String file = "D:/Fer/Vještine/ROPAERUJ/hw07-0036506587/data/07-iris-formatirano.data";
        dataset.loadDataset(Paths.get(file));
        FFANN ffann = new FFANN(new int[]{4, 5, 3, 3}, new Sigmoid(), dataset);
        ffann.train(100000, 0.02, 0.001);
        ffann.statistics();
    }

}

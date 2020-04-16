package hr.fer.zemris.fthesis;

import hr.fer.zemris.fthesis.afunction.Sigmoid;
import hr.fer.zemris.fthesis.ann.FFANN;
import hr.fer.zemris.fthesis.dataset.IrisDataset;
import hr.fer.zemris.fthesis.dataset.ReadOnlyDataset;

import java.io.IOException;
import java.nio.file.Paths;

public class Demo {

    public static void main(String[] args) throws IOException {
        ReadOnlyDataset dataset = new IrisDataset();
        dataset.loadDataset(Paths.get(
                "D:/Fer/Vještine/ROPAERUJ/hw07-0036506587/data/07-iris-formatirano.data"));
        FFANN ffann = new FFANN(new int[]{4, 5, 5, 3}, new Sigmoid(), dataset);
        double[] inputs = {1, 2, 3,5,6};
        ffann.feedForward(inputs);
        //ffann.train(10000, 0.02, 0.1);
        //ffann.statistics();
    }

}

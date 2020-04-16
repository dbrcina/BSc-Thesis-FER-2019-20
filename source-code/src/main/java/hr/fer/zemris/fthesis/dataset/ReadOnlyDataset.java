package hr.fer.zemris.fthesis.dataset;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map.Entry;

public interface ReadOnlyDataset {

    void loadDataset(Path file) throws IOException;

    int numberOfSamples();

    Entry<double[], double[]> inputsOutputsPair(int index);

}

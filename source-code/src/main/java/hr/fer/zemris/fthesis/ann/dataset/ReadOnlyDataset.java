package hr.fer.zemris.fthesis.ann.dataset;

import hr.fer.zemris.fthesis.ann.dataset.model.Sample;

import java.nio.file.Path;

/**
 * Interface which provides some generic methods for working with datasets.
 */
public interface ReadOnlyDataset {

    /**
     * Loads dataset defined in provided <i>file</i>.
     *
     * @param file dataset definition.
     * @throws Exception if something goes wrong.
     */
    void loadDataset(Path file) throws Exception;

    /**
     * @return number of samples in dataset.
     */
    int numberOfSamples();

    /**
     * @param i i-th sample.
     * @return i-th sample where numeration starts from 0.
     * @throws IndexOutOfBoundsException if provided <i>i</i> is invalid.
     */
    Sample getSample(int i);

}
package hr.fer.zemris.bscthesis.dataset;

import java.nio.file.Path;
import java.util.List;

/**
 * Interface which provides some generic methods for working with datasets.
 *
 * @author dbrcina
 */
public interface Dataset extends Iterable<Sample> {

    /**
     * Setter for list of samples.
     *
     * @param samples list of samples.
     * @throws NullPointerException if provided list is <code>null</code>.
     */
    void setSamples(List<Sample> samples);

    /**
     * @return number of samples in dataset.
     */
    int numberOfSamples();

    /**
     * Creates a new list of samples and shuffles it with {@link java.util.Collections#shuffle(List)} method.
     *
     * @return new shuffled list of samples.
     */
    List<Sample> shuffleSamples();

    /**
     * Loads dataset defined in provided <code>file</code>.
     *
     * @param file dataset definition.
     * @throws Exception if something goes wrong.
     */
    void loadDataset(Path file) throws Exception;

}

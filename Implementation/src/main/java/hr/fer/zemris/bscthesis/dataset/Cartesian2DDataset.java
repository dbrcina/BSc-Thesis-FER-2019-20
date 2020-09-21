package hr.fer.zemris.bscthesis.dataset;

import java.nio.file.Path;
import java.util.*;
import java.util.function.Consumer;

/**
 * An implementation of {@link Dataset}. It receives a collection of {@link Sample}s through
 * {@link #setSamples(List)} method where inputs are represented by 2D points from Cartesian coordinate system.
 *
 * @author dbrcina
 */
public class Cartesian2DDataset implements Dataset {

    private List<Sample> samples;

    @Override
    public void setSamples(List<Sample> samples) {
        this.samples = samples;
    }

    @Override
    public int numberOfSamples() {
        return samples.size();
    }

    @Override
    public List<Sample> shuffleSamples() {
        List<Sample> shuffled = new LinkedList<>(samples);
        Collections.shuffle(shuffled);
        return shuffled;
    }

    @Override
    public void loadDataset(Path file) {
        throw new RuntimeException("Cartesian2DDataset::loadDataset(Path) is currently not supported.");
    }

    @Override
    public Iterator<Sample> iterator() {
        return samples.iterator();
    }

    @Override
    public void forEach(Consumer<? super Sample> action) {
        samples.forEach(action);
    }

    @Override
    public Spliterator<Sample> spliterator() {
        return samples.spliterator();
    }

}

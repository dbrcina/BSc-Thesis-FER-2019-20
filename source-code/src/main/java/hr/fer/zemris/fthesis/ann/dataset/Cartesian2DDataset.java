package hr.fer.zemris.fthesis.ann.dataset;

import hr.fer.zemris.fthesis.ann.dataset.model.Sample;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Objects;

/**
 * An implementation of {@link ReadOnlyDataset}. It receives a collection of {@link Sample}s where inputs
 * are represented by 2D points from Cartesian coordinate system.
 */
public class Cartesian2DDataset implements ReadOnlyDataset {

    private final List<Sample> samples;

    public Cartesian2DDataset(Collection<Sample> samples) {
        this.samples = new ArrayList<>(samples);
    }

    @Override
    public int numberOfSamples() {
        return samples.size();
    }

    @Override
    public Sample getSample(int i) {
        Objects.checkIndex(i, numberOfSamples());
        return samples.get(i);
    }

    @Override
    public void loadDataset(Path file) throws Exception {
        throw new RuntimeException("This method is currently not supported.");
    }

}

package hr.fer.zemris.fthesis.ann.dataset;

import hr.fer.zemris.fthesis.ann.dataset.model.Sample;

import java.nio.file.Path;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

/**
 * An implementation of {@link ReadOnlyDataset}. It receives a collection of {@link Sample}s where inputs
 * are represented by 2D points from Cartesian coordinate system.
 */
public class Cartesian2DDataset implements ReadOnlyDataset {

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
    public Sample getSample(int i) {
        Objects.checkIndex(i, numberOfSamples());
        return samples.get(i);
    }

    @Override
    public void loadDataset(Path file) throws Exception {
        throw new RuntimeException("This method is currently not supported.");
    }

    @Override
    public List<Sample> samples() {
        return new LinkedList<>(samples);
    }

}

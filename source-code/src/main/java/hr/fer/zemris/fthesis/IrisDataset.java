package hr.fer.zemris.fthesis;

import hr.fer.zemris.fthesis.ann.dataset.ReadOnlyDataset;
import hr.fer.zemris.fthesis.ann.dataset.model.Sample;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class IrisDataset implements ReadOnlyDataset {

    private final List<Sample> samples = new ArrayList<>();

    @Override
    public void loadDataset(Path file) throws Exception {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(
                Files.newInputStream(file)))
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.replaceAll("[()]", "");
                String[] inputsOutputs = line.split(":");
                double[] inputs = Arrays.stream(inputsOutputs[0].split(","))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                double[] outputs = Arrays.stream(inputsOutputs[1].split(","))
                        .mapToDouble(Double::parseDouble)
                        .toArray();
                samples.add(new Sample(inputs, outputs));
            }
        }
    }

    @Override
    public int numberOfSamples() {
        return samples.size();
    }

    @Override
    public Sample getSample(int i) {
        return samples.get(i);
    }

}

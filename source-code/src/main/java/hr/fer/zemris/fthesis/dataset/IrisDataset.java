package hr.fer.zemris.fthesis.dataset;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Map.Entry;

public class IrisDataset implements ReadOnlyDataset {

    private List<Entry<double[], double[]>> inputsOutputsPairList = new LinkedList<>();

    @Override
    public void loadDataset(Path file) throws IOException {
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
                inputsOutputsPairList.add(new AbstractMap.SimpleEntry<>(inputs, outputs));
            }
        }
    }

    @Override
    public int numberOfSamples() {
        return inputsOutputsPairList.size();
    }

    @Override
    public Entry<double[], double[]> inputsOutputsPair(int index) {
        return inputsOutputsPairList.get(index);
    }

}

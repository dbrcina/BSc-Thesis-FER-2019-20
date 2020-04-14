package hr.fer.zemris.fthesis;

import hr.fer.zemris.fthesis.afunction.SIGM;
import hr.fer.zemris.fthesis.ann.FFANN;

import java.util.Arrays;

public class Demo {

    public static void main(String[] args) {
        FFANN ffann = new FFANN(new int[]{2, 3, 2}, new SIGM());
        double[] inputs = {1, 2};
        System.out.println(Arrays.toString(ffann.calculateOutputs(inputs)));
    }

}

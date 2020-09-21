package hr.fer.zemris.bscthesis.classes;

import java.awt.*;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.lang.reflect.Constructor;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;

/**
 * <p>
 * Each sample class consists of:
 * <ul>
 *     <li><code>id</code>,</li>
 *     <li><code>desiredOutputs</code>,</li>
 *     <li><code>actualOutputs</code>,</li>
 *     <li><code>color and</code></li>
 *     <li><code>shape</code></li>
 * </ul>
 * and two classes are <b>equal</b> if they have the <b>same</b> id.
 * </p>
 *
 * <hr>
 *
 * <p>
 * <code>desiredOutputs</code> are known in advance <b>(*)</b>.<br>
 * <code>color</code> is calculated based on <code>actualOutputs</code> <b>(**)</b>.<br>
 * <code>shape</code> is determined by each ClassType individually.<br>
 * <code>desiredOutputs</code> and <code>actualOutputs</code> are arrays of doubles that are in range [0.0, 1.0].
 * </p>
 *
 * <hr>
 *
 * <p>
 * ClassTypes are determined by {@link #determineFor(double[])} method and their initialization is invoked
 * through {@link #init()} method.
 * </p>
 *
 * <p>
 * <b>(*)</b> - they are not physically present for each class instance and can be found in
 * {@link #desiredOutputsForClasses} map.
 * <br>
 * <b>(**)</b> - this calculation is supported through {@link #rgb(double[])} method.
 * </p>
 *
 * @author dbrcina
 */
public abstract class ClassType {

    /**
     * Map which stores class name as a key and desired outputs for that class as a value.
     */
    private static final Map<String, double[]> desiredOutputsForClasses = new HashMap<>();
    /**
     * Map which stores desired outputs as a key and class as a value. Here we use list rather than double[] as a key
     * because of default equals and hashcode methods.
     */
    private static final Map<List<Double>, Class<? extends ClassType>> classesForDesiredOutputs = new HashMap<>();

    private final String id;
    private final double[] actualOutputs;
    private final Color color;
    private final Shape shape; // only one reference is created for each instance, so it can be reused.

    /**
     * Constructor.<br>
     * If <code>actualOutputs</code> is <code>null</code>, <code>desiredOutputs</code> will be used as "actual".
     *
     * @param id            class types id.
     * @param actualOutputs class types actual outputs.
     * @param shape         class types shape.
     */
    protected ClassType(String id, double[] actualOutputs, Shape shape) {
        this.id = id;
        this.actualOutputs = actualOutputs == null ? getDesiredOutputs() : actualOutputs;
        this.color = rgb(this.actualOutputs);
        this.shape = shape;
    }

    /**
     * Calculates RGB color based on provided array of <code>outputs</code>,
     * which consists of doubles between <code>0.0</code> and <code>1.0</code> both inclusive.
     * <hr>
     * P.S. (FOR NOW) It expects an array with size of 3, so that each output represents RGB component respectively.
     * This was implemented specifically for the multiclass classification with 3 different classes, so for more generic
     * problem, new algorithm should be implemented!
     *
     * @param outputs an array of doubles.
     * @return RGB color.
     */
    private Color rgb(double[] outputs) {
        if (outputs.length != 3) {
            throw new RuntimeException("ClassType::rgb(double[]) doesn't support array of length: " + outputs.length);
        }
        int r = (int) (outputs[0] * 255);
        int g = (int) (outputs[1] * 255);
        int b = (int) (outputs[2] * 255);
        return new Color(r, g, b);
    }

    /**
     * Creates a shape based on provided rectangle.
     *
     * @param rect rectangle.
     * @return shape.
     */
    public abstract Shape createShape(Rectangle rect);

    /**
     * @return class id.
     */
    public String getId() {
        return id;
    }

    /**
     * @return desired outputs.
     */
    public double[] getDesiredOutputs() {
        return desiredOutputsForClasses.get((getClass().getName()));
    }

    /**
     * @return actual outputs.
     */
    public double[] getActualOutputs() {
        return actualOutputs;
    }

    /**
     * @return class color.
     */
    public Color getColor() {
        return color;
    }

    /**
     * @return shape.
     */
    protected Shape getShape() {
        return shape;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ClassType classType = (ClassType) o;
        return id.equals(classType.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    /**
     * Determines ClassType for the given <code>outputs</code>. If a value is <= 0.5, it is rounded to 0.0, otherwise to
     * 1.0. So, [0.6, 0.3, 0.4] will be interpreted as [1.0, 0.0, 0.0] and a ClassType whose desired outputs are
     * equal to interpreted array will be returned.
     * <br>
     * If something goes wrong, program will be terminated.
     *
     * @param outputs outputs.
     * @return an instance of ClassType.
     */
    public static ClassType determineFor(double[] outputs) {
        List<Double> rounded = Arrays.stream(outputs)
                .map(value -> value <= 0.5 ? 0.0 : 1.0)
                .boxed()
                .collect(Collectors.toUnmodifiableList());
        ClassType classType = null;
        Class<? extends ClassType> clazz = classesForDesiredOutputs.getOrDefault(rounded, ClassNone.class);
        try {
            Constructor<? extends ClassType> constructor = clazz.getConstructor(double[].class);
            classType = constructor.newInstance(outputs);
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
        return classType;
    }

    /**
     * Dynamically initialize maps that store information about classes and their desired outputs.
     * Configuration is in <b>.class-types-config</b> file, which is in resources folder.
     * <br>
     * If something goes wrong, program will be terminated.
     */
    public static void init() {
        ClassLoader loader = ClassType.class.getClassLoader();
        try (
                InputStream is = loader.getResourceAsStream(".class-types-config");
                BufferedReader br = new BufferedReader(new InputStreamReader(
                        Objects.requireNonNull(is, ".class-types-config file doesn't exist.")))
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split("\\s*=\\s*");
                desiredOutputsForClasses.put(parts[0], Arrays.stream(parts[1].split("\\s+"))
                        .mapToDouble(Double::parseDouble)
                        .toArray()
                );
            }
            for (Map.Entry<String, double[]> entry : desiredOutputsForClasses.entrySet()) {
                Class<ClassType> clazz = (Class<ClassType>) Class.forName(entry.getKey());
                classesForDesiredOutputs.put(
                        Arrays.stream(entry.getValue()).boxed().collect(Collectors.toList()),
                        clazz);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(-1);
        }
    }

}

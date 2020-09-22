package hr.fer.zemris.bscthesis.ann.afunction;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Model of activation function.
 * <br>
 * Implementations of this abstraction can be loaded through {@link #loadAFunctions()} method and returned through
 * {@link #allAFunctions()} method.
 *
 * @author dbrcina
 */
public abstract class ActivationFunction {

    private static Map<String, ActivationFunction> aFunctionsMap;

    private final String id;

    protected ActivationFunction(String id) {
        this.id = id;
    }

    /**
     * @return activation functions id.
     */
    public String getId() {
        return id;
    }

    /**
     * Calculates value at provided point <code>x</code>.
     *
     * @param x point x.
     * @return value at point <code>x</code>.
     */
    public abstract double value(double x);

    /**
     * Calculates derivative value at provided point <code>x</code>.
     *
     * @param x point x.
     * @return derivative value at point <code>x</code>.
     */
    public abstract double derivativeValue(double x);

    /**
     * @return map where key is activation functions id and value a class instance.
     */
    public static Map<String, ActivationFunction> allAFunctions() {
        return new HashMap<>(aFunctionsMap);
    }

    /**
     * Dynamically loads implementations of ActivationFunction abstraction from <b>.activation-functions-config</b>,
     * which is in resources folder.
     * <br>
     * If something goes wrong, program will be terminated.
     */
    public static void loadAFunctions() {
        boolean initialized = aFunctionsMap != null;
        if (!initialized) {
            aFunctionsMap = new HashMap<>();
            ClassLoader loader = ActivationFunction.class.getClassLoader();
            try (
                    InputStream is = loader.getResourceAsStream(".activation-functions-config");
                    BufferedReader br = new BufferedReader(new InputStreamReader(
                            Objects.requireNonNull(is, ".activation-functions-config file doesn't exist.")))
            ) {
                String className;
                while ((className = br.readLine()) != null) {
                    Class<ActivationFunction> aFunctionClass = (Class<ActivationFunction>) Class.forName(className);
                    ActivationFunction aFunction = aFunctionClass.getConstructor().newInstance();
                    aFunctionsMap.put(aFunction.getId(), aFunction);
                }
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(-1);
            }
        }
    }

}

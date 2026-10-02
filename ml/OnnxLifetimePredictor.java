package ml;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;

/**
 * ONNX Runtime adapter. Reflection keeps the base simulator buildable without
 * the optional native runtime JAR; construction fails clearly when it is absent.
 */
public final class OnnxLifetimePredictor implements LifetimePredictor, AutoCloseable {
    private final Object environment;
    private final Object session;
    private final Class<?> environmentClass;
    private final Class<?> tensorClass;
    private final String inputName;
    private final String probabilityOutput;
    private final int probabilityOutputIndex;
    private final int probabilityClassIndex;
    private final double threshold;
    private final String[] numericColumns;
    private final double[] medians;
    private final double[] means;
    private final double[] scales;
    private final Map<String, Integer> allocationSites = new LinkedHashMap<>();
    private final int vectorSize;

    public OnnxLifetimePredictor(Path modelPath, Path propertiesPath) {
        if (modelPath == null || propertiesPath == null)
            throw new IllegalArgumentException("modelPath and propertiesPath must not be null");
        Properties properties = loadProperties(propertiesPath);
        inputName = required(properties, "input.name");
        probabilityOutput = required(properties, "probability.output");
        probabilityOutputIndex = integer(properties, "probability.output.index");
        probabilityClassIndex = integer(properties, "probability.class.index");
        threshold = decimal(properties, "long_lived.threshold");
        numericColumns = required(properties, "numeric.columns").split(",");
        medians = numbers(properties, "numeric.medians");
        means = numbers(properties, "numeric.means");
        scales = numbers(properties, "numeric.scales");
        vectorSize = integer(properties, "vector.size");
        int siteCount = integer(properties, "allocation.site.count");
        for (int index = 0; index < siteCount; index++)
            allocationSites.put(required(properties, "allocation.site." + index), index);
        if (numericColumns.length != medians.length || medians.length != means.length
                || means.length != scales.length)
            throw new IllegalArgumentException("inconsistent numeric preprocessing metadata");
        if (vectorSize != numericColumns.length + allocationSites.size() + 1)
            throw new IllegalArgumentException("vector.size does not match preprocessing metadata");

        try {
            environmentClass = Class.forName("ai.onnxruntime.OrtEnvironment");
            Class<?> sessionOptionsClass = Class.forName("ai.onnxruntime.OrtSession$SessionOptions");
            tensorClass = Class.forName("ai.onnxruntime.OnnxTensor");
            environment = environmentClass.getMethod("getEnvironment").invoke(null);
            Object options = sessionOptionsClass.getConstructor().newInstance();
            session = environmentClass.getMethod("createSession", String.class, sessionOptionsClass)
                    .invoke(environment, modelPath.toAbsolutePath().toString(), options);
        } catch (ClassNotFoundException exception) {
            throw new IllegalStateException(
                    "ONNX Runtime Java is not on the classpath; add com.microsoft.onnxruntime:onnxruntime",
                    exception);
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException("could not initialize ONNX Runtime", exception);
        }
    }

    @Override
    public Prediction predict(ObjectFeatures features) {
        if (features == null) throw new IllegalArgumentException("features must not be null");
        long start = System.nanoTime();
        Object tensor = null;
        Object result = null;
        try {
            float[][] batch = new float[][] { vectorize(features) };
            Method createTensor = tensorClass.getMethod("createTensor", environmentClass, Object.class);
            tensor = createTensor.invoke(null, environment, batch);
            Map<String, Object> inputs = new LinkedHashMap<>();
            inputs.put(inputName, tensor);
            result = session.getClass().getMethod("run", Map.class).invoke(session, inputs);
            Object outputValue = namedOutput(result);
            Object value = outputValue.getClass().getMethod("getValue").invoke(outputValue);
            double probability = probability(value, probabilityClassIndex);
            return new Prediction(probability >= threshold, probability,
                    System.nanoTime() - start);
        } catch (ReflectiveOperationException exception) {
            throw new IllegalStateException("ONNX inference failed", exception);
        } finally {
            closeQuietly(result);
            closeQuietly(tensor);
        }
    }

    private float[] vectorize(ObjectFeatures features) {
        float[] vector = new float[vectorSize];
        for (int index = 0; index < numericColumns.length; index++) {
            double value = numericValue(features, numericColumns[index]);
            if (!Double.isFinite(value)) value = medians[index];
            vector[index] = (float) ((value - means[index]) / scales[index]);
        }
        Integer site = allocationSites.get(features.getAllocationSite());
        int category = site == null ? allocationSites.size() : site;
        vector[numericColumns.length + category] = 1.0f;
        return vector;
    }

    private Object namedOutput(Object result) throws ReflectiveOperationException {
        Object optionalValue = result.getClass().getMethod("get", String.class)
                .invoke(result, probabilityOutput);
        if (optionalValue instanceof Optional && ((Optional<?>) optionalValue).isPresent())
            return ((Optional<?>) optionalValue).get();
        return result.getClass().getMethod("get", int.class).invoke(result, probabilityOutputIndex);
    }

    private static double probability(Object value, int classIndex) {
        if (value == null || !value.getClass().isArray() || Array.getLength(value) == 0)
            throw new IllegalStateException("probability output is not a non-empty tensor");
        Object row = Array.get(value, 0);
        if (row == null || !row.getClass().isArray() || Array.getLength(row) <= classIndex)
            throw new IllegalStateException("probability tensor has an unexpected shape");
        Object number = Array.get(row, classIndex);
        if (!(number instanceof Number))
            throw new IllegalStateException("probability tensor is not numeric");
        return ((Number) number).doubleValue();
    }

    private static double numericValue(ObjectFeatures features, String column) {
        switch (column) {
            case "size_bytes": return features.getSizeBytes();
            case "incoming_reference_count": return features.getIncomingReferenceCount();
            case "outgoing_reference_count": return features.getOutgoingReferenceCount();
            case "heap_utilization_at_allocation": return features.getHeapUtilizationAtAllocation();
            case "allocation_rate": return features.getAllocationRate();
            case "gc_cycles_survived": return features.getGcCyclesSurvived();
            case "age_at_prediction": return features.getAgeAtPrediction();
            default: throw new IllegalArgumentException("unknown numeric feature " + column);
        }
    }

    private static Properties loadProperties(Path path) {
        Properties properties = new Properties();
        try (InputStream stream = Files.newInputStream(path)) {
            properties.load(stream);
            return properties;
        } catch (IOException exception) {
            throw new IllegalArgumentException("could not read model properties " + path, exception);
        }
    }

    private static String required(Properties properties, String name) {
        String value = properties.getProperty(name);
        if (value == null) throw new IllegalArgumentException("missing model property " + name);
        return value;
    }

    private static int integer(Properties properties, String name) {
        return Integer.parseInt(required(properties, name));
    }

    private static double decimal(Properties properties, String name) {
        return Double.parseDouble(required(properties, name));
    }

    private static double[] numbers(Properties properties, String name) {
        String[] values = required(properties, name).split(",");
        double[] result = new double[values.length];
        for (int index = 0; index < values.length; index++)
            result[index] = Double.parseDouble(values[index]);
        return result;
    }

    private static void closeQuietly(Object value) {
        if (value == null) return;
        try {
            value.getClass().getMethod("close").invoke(value);
        } catch (ReflectiveOperationException ignored) {
            // Best-effort cleanup while preserving the original inference result/error.
        }
    }

    @Override
    public void close() {
        closeQuietly(session);
    }
}

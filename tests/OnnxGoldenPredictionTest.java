package tests;

import java.io.BufferedReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;
import ml.ObjectFeatures;
import ml.OnnxLifetimePredictor;
import ml.Prediction;

/** Cross-language golden test; requires the ONNX Runtime JAR/native library. */
public final class OnnxGoldenPredictionTest {
    private OnnxGoldenPredictionTest() {}

    public static void main(String[] args) throws Exception {
        if (args.length != 3)
            throw new IllegalArgumentException("usage: model.onnx model.properties golden.csv");
        Path model = Paths.get(args[0]);
        Path propertiesPath = Paths.get(args[1]);
        Path golden = Paths.get(args[2]);
        Properties properties = new Properties();
        try (java.io.InputStream stream = Files.newInputStream(propertiesPath)) {
            properties.load(stream);
        }
        double tolerance = Double.parseDouble(properties.getProperty("prediction.tolerance"));
        int rows = 0;
        try (OnnxLifetimePredictor predictor = new OnnxLifetimePredictor(model, propertiesPath);
             BufferedReader reader = Files.newBufferedReader(golden, StandardCharsets.UTF_8)) {
            String header = reader.readLine();
            if (header == null || !header.endsWith("expected_probability"))
                throw new AssertionError("unexpected golden CSV header");
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",", -1);
                ObjectFeatures features = new ObjectFeatures(
                        Integer.parseInt(values[0]), Integer.parseInt(values[1]),
                        Integer.parseInt(values[2]), Double.parseDouble(values[3]),
                        Double.parseDouble(values[4]), Integer.parseInt(values[5]),
                        Integer.parseInt(values[6]), values[7]);
                double expected = Double.parseDouble(values[8]);
                Prediction prediction = predictor.predict(features);
                if (Math.abs(expected - prediction.getProbability()) > tolerance)
                    throw new AssertionError("prediction mismatch at golden row " + (rows + 1));
                rows++;
            }
        }
        if (rows == 0) throw new AssertionError("golden dataset must not be empty");
        System.out.println("ONNX golden predictions matched for " + rows + " rows.");
    }
}

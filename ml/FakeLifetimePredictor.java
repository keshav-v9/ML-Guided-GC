package ml;

import java.util.function.ToDoubleFunction;

/** Deterministic predictor for collector and integration tests. */
public final class FakeLifetimePredictor implements LifetimePredictor {
    private final ToDoubleFunction<ObjectFeatures> probabilityFunction;
    private final double threshold;

    public FakeLifetimePredictor(ToDoubleFunction<ObjectFeatures> probabilityFunction,
                                 double threshold) {
        if (probabilityFunction == null)
            throw new IllegalArgumentException("probabilityFunction must not be null");
        if (threshold < 0.0 || threshold > 1.0)
            throw new IllegalArgumentException("threshold must be between 0 and 1");
        this.probabilityFunction = probabilityFunction;
        this.threshold = threshold;
    }

    @Override
    public Prediction predict(ObjectFeatures features) {
        if (features == null) throw new IllegalArgumentException("features must not be null");
        long start = System.nanoTime();
        double probability = probabilityFunction.applyAsDouble(features);
        if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0)
            throw new IllegalStateException("probability must be finite and between 0 and 1");
        return new Prediction(probability >= threshold, probability, System.nanoTime() - start);
    }
}

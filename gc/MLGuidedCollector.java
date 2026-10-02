package gc;

import core.Generation;
import core.Heap;
import core.HeapObject;
import ml.LifetimePredictor;
import ml.ObjectFeatures;
import ml.Prediction;

/**
 * Applies lifetime predictions as a safe promotion policy around an exact
 * tracing collector. Predictions never participate in reachability decisions.
 */
public final class MLGuidedCollector implements GarbageCollector {
    private final GarbageCollector delegate;
    private final LifetimePredictor predictor;
    private final int minimumAge;
    private long collectionCount;
    private long predictionCount;
    private long promotionCount;
    private long predictionFailureCount;
    private long inferenceNanos;

    public MLGuidedCollector(GarbageCollector delegate, LifetimePredictor predictor) {
        this(delegate, predictor, 1);
    }

    public MLGuidedCollector(GarbageCollector delegate, LifetimePredictor predictor,
                             int minimumAge) {
        if (delegate == null || predictor == null)
            throw new IllegalArgumentException("delegate and predictor must not be null");
        if (minimumAge < 0)
            throw new IllegalArgumentException("minimumAge must not be negative");
        this.delegate = delegate;
        this.predictor = predictor;
        this.minimumAge = minimumAge;
    }

    @Override
    public void collect(Heap heap) {
        if (heap == null) throw new IllegalArgumentException("heap must not be null");

        // Establish liveness first. Only survivors are worth scoring, and the model
        // cannot influence which objects the tracing collector retains.
        delegate.collect(heap);
        collectionCount++;
        for (int address = 0; address < heap.capacity(); address++) {
            HeapObject object = heap.get(address);
            if (object == null || object.getGeneration() != Generation.YOUNG
                    || object.getAge() < minimumAge) continue;
            try {
                Prediction prediction = predictor.predict(ObjectFeatures.from(object));
                predictionCount++;
                inferenceNanos += Math.max(0L, prediction.getInferenceNanos());
                if (prediction.isLongLived()) {
                    object.setGeneration(Generation.OLD);
                    promotionCount++;
                }
            } catch (RuntimeException failure) {
                // Fail open: keep the object young and let exact tracing preserve safety.
                predictionFailureCount++;
            }
        }
    }

    public long getCollectionCount() { return collectionCount; }
    public long getPredictionCount() { return predictionCount; }
    public long getPromotionCount() { return promotionCount; }
    public long getPredictionFailureCount() { return predictionFailureCount; }
    public long getInferenceNanos() { return inferenceNanos; }

    public void resetStatistics() {
        collectionCount = 0L;
        predictionCount = 0L;
        promotionCount = 0L;
        predictionFailureCount = 0L;
        inferenceNanos = 0L;
    }
}

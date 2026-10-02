package ml;

public final class Prediction {
    private final boolean longLived;
    private final double probability;
    private final long inferenceNanos;

    public Prediction(boolean longLived, double probability, long inferenceNanos) {
        this.longLived = longLived;
        this.probability = probability;
        this.inferenceNanos = inferenceNanos;
    }

    public boolean isLongLived() { return longLived; }
    public double getProbability() { return probability; }
    public long getInferenceNanos() { return inferenceNanos; }
}

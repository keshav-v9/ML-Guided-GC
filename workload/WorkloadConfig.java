package workload;

/** Immutable configuration for one reproducible workload run. */
public final class WorkloadConfig {
    private final WorkloadType type;
    private final int heapCapacity;
    private final int objectCount;
    private final long seed;
    private final int collectionInterval;
    private final int shortLifetimeTicks;
    private final int longLifetimeTicks;
    private final double longLivedFraction;

    public WorkloadConfig(WorkloadType type, int heapCapacity, int objectCount, long seed,
                          int collectionInterval, int shortLifetimeTicks,
                          int longLifetimeTicks, double longLivedFraction) {
        if (type == null) throw new IllegalArgumentException("type must not be null");
        if (heapCapacity < 0) throw new IllegalArgumentException("heapCapacity must not be negative");
        if (objectCount < 0) throw new IllegalArgumentException("objectCount must not be negative");
        if (objectCount > 0 && heapCapacity == 0)
            throw new IllegalArgumentException("non-empty workloads require heap capacity");
        if (collectionInterval <= 0)
            throw new IllegalArgumentException("collectionInterval must be positive");
        if (shortLifetimeTicks <= 0)
            throw new IllegalArgumentException("shortLifetimeTicks must be positive");
        if (longLifetimeTicks <= shortLifetimeTicks)
            throw new IllegalArgumentException("longLifetimeTicks must exceed shortLifetimeTicks");
        if (longLivedFraction < 0.0 || longLivedFraction > 1.0)
            throw new IllegalArgumentException("longLivedFraction must be between 0 and 1");
        this.type = type;
        this.heapCapacity = heapCapacity;
        this.objectCount = objectCount;
        this.seed = seed;
        this.collectionInterval = collectionInterval;
        this.shortLifetimeTicks = shortLifetimeTicks;
        this.longLifetimeTicks = longLifetimeTicks;
        this.longLivedFraction = longLivedFraction;
    }

    public static WorkloadConfig defaults(WorkloadType type, int heapCapacity,
                                          int objectCount, long seed) {
        int interval = Math.max(1, heapCapacity / 10);
        return new WorkloadConfig(type, heapCapacity, objectCount, seed, interval,
                3, 30, 0.20);
    }

    public WorkloadType getType() { return type; }
    public int getHeapCapacity() { return heapCapacity; }
    public int getObjectCount() { return objectCount; }
    public long getSeed() { return seed; }
    public int getCollectionInterval() { return collectionInterval; }
    public int getShortLifetimeTicks() { return shortLifetimeTicks; }
    public int getLongLifetimeTicks() { return longLifetimeTicks; }
    public double getLongLivedFraction() { return longLivedFraction; }
}

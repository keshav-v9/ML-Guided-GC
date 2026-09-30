package workload;

import core.Heap;

/** Result and reproducibility metadata for one generated workload. */
public final class WorkloadResult {
    private final WorkloadConfig config;
    private final Heap heap;
    private final int collectionCount;

    WorkloadResult(WorkloadConfig config, Heap heap, int collectionCount) {
        this.config = config;
        this.heap = heap;
        this.collectionCount = collectionCount;
    }

    public WorkloadConfig getConfig() { return config; }
    public Heap getHeap() { return heap; }
    public int getCollectionCount() { return collectionCount; }
}

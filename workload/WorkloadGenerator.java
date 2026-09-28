package workload;

import core.Heap;
import java.util.Random;

/** Creates deterministic object graphs for demos and benchmarks. */
public final class WorkloadGenerator {
    private WorkloadGenerator() {}

    public static Heap generate(int capacity, int objectCount, long seed) {
        if (objectCount < 0 || objectCount > capacity)
            throw new IllegalArgumentException("objectCount must be between 0 and capacity");
        Heap heap = new Heap(capacity);
        Random random = new Random(seed);
        for (int i = 0; i < objectCount; i++) heap.allocate("object-" + (i + 1));
        if (objectCount == 0) return heap;

        int rootCount = Math.max(1, objectCount / 10);
        for (int i = 0; i < rootCount; i++) heap.addRoot(random.nextInt(objectCount));
        for (int from = 0; from < objectCount; from++) {
            int referenceCount = random.nextInt(4);
            for (int j = 0; j < referenceCount; j++)
                heap.addReference(from, random.nextInt(objectCount));
        }
        return heap;
    }
}

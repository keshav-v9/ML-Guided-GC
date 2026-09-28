package bench;

import core.Heap;
import gc.CopyingCollector;
import gc.GarbageCollector;
import gc.MarkAndCompactCollector;
import gc.MarkAndSweepCollector;
import workload.WorkloadGenerator;

/** A lightweight comparison; timings are illustrative rather than a JMH benchmark. */
public final class Benchmark {
    private static final int CAPACITY = 100_000;
    private static final int OBJECTS = 80_000;
    private static final long SEED = 42L;

    private Benchmark() {}

    public static void main(String[] args) {
        run("Mark and sweep", new MarkAndSweepCollector());
        run("Mark and compact", new MarkAndCompactCollector());
        run("Copying", new CopyingCollector());
    }

    private static void run(String name, GarbageCollector collector) {
        Heap heap = WorkloadGenerator.generate(CAPACITY, OBJECTS, SEED);
        int before = heap.usedSlots();
        long start = System.nanoTime();
        collector.collect(heap);
        long elapsed = System.nanoTime() - start;
        System.out.printf("%-18s %7.2f ms | before: %,d | live: %,d | reclaimed: %,d%n",
                name, elapsed / 1_000_000.0, before, heap.usedSlots(), before - heap.usedSlots());
    }
}

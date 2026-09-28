package core;

import gc.GarbageCollector;

/** Allocation facade that triggers collection once when the heap is full. */
public class Allocator {
    private final Heap heap;
    private final GarbageCollector collector;

    public Allocator(Heap heap, GarbageCollector collector) {
        if (heap == null || collector == null)
            throw new IllegalArgumentException("heap and collector must not be null");
        this.heap = heap;
        this.collector = collector;
    }

    public int allocate(String name) {
        int address = heap.allocate(name);
        if (address >= 0) return address;
        collector.collect(heap);
        return heap.allocate(name);
    }

    public Heap getHeap() { return heap; }
}

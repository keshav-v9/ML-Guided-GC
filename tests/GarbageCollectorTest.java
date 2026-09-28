package tests;

import core.Allocator;
import core.Heap;
import gc.CopyingCollector;
import gc.GarbageCollector;
import gc.MarkAndCompactCollector;
import gc.MarkAndSweepCollector;

public final class GarbageCollectorTest {
    public static void main(String[] args) {
        testMarkAndSweep();
        testMovingCollector(new MarkAndCompactCollector());
        testMovingCollector(new CopyingCollector());
        testNoRoots(new MarkAndSweepCollector());
        testNoRoots(new MarkAndCompactCollector());
        testNoRoots(new CopyingCollector());
        testEmptyHeap();
        testAllocatorRetriesAfterCollection();
        System.out.println("All garbage collector tests passed.");
    }

    private static Heap graph() {
        Heap heap = new Heap(7);
        for (int i = 0; i < 6; i++) heap.allocate(String.valueOf((char) ('A' + i)));
        heap.addRoot(2);       // C -> E -> C (live cycle)
        heap.addReference(2, 4);
        heap.addReference(4, 2);
        heap.addReference(0, 1); // unreachable graph
        return heap;
    }

    private static void testMarkAndSweep() {
        Heap heap = graph();
        new MarkAndSweepCollector().collect(heap);
        check(heap.usedSlots() == 2, "sweep should retain exactly two objects");
        check(heap.get(2) != null && heap.get(4) != null, "sweep must not move live objects");
        check(heap.get(2).getReferences().get(0) == 4, "sweep must preserve references");
        check(!heap.get(2).isMarked(), "marks must be reset after collection");
    }

    private static void testMovingCollector(GarbageCollector collector) {
        Heap heap = graph();
        collector.collect(heap);
        check(heap.usedSlots() == 2, "moving collector should retain exactly two objects");
        check(heap.isRoot(0), "root should be updated to its new address");
        check("C".equals(heap.get(0).getName()), "root object should move to address zero");
        int target = heap.get(0).getReferences().get(0);
        check(target >= 0 && target < 2, "references should use moved addresses");
        check(heap.get(target).getReferences().contains(0), "cycle should survive relocation");
        check(heap.get(2) == null, "live region should be contiguous");
    }

    private static void testNoRoots(GarbageCollector collector) {
        Heap heap = new Heap(3);
        heap.allocate("garbage");
        collector.collect(heap);
        check(heap.usedSlots() == 0, "an unrooted heap should be reclaimed completely");
    }

    private static void testAllocatorRetriesAfterCollection() {
        Heap heap = new Heap(2);
        int live = heap.allocate("live");
        heap.addRoot(live);
        heap.allocate("garbage");
        int address = new Allocator(heap, new MarkAndSweepCollector()).allocate("replacement");
        check(address == 1, "allocator should retry in a reclaimed slot");
    }

    private static void testEmptyHeap() {
        Heap heap = new Heap(0);
        new CopyingCollector().collect(heap);
        check(heap.allocate("impossible") == -1, "a zero-capacity heap should remain full");
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}

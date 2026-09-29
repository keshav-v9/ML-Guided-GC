package tests;

import core.Allocator;
import core.Generation;
import core.Heap;
import core.HeapObject;
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
        testAllocationAndInvalidAddresses();
        testReferencesAndRoots();
        testCycleAndRecursiveMarking();
        testInstrumentation();
        testInstrumentationWithMovingCollectors();
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

    private static void testAllocationAndInvalidAddresses() {
        Heap heap = new Heap(1);
        check(heap.allocate("first") == 0, "first allocation should use address zero");
        check(heap.allocate("full") == -1, "allocation should fail when the heap is full");
        expectThrows(IndexOutOfBoundsException.class, () -> heap.get(-1),
                "negative addresses should be rejected");
        expectThrows(IndexOutOfBoundsException.class, () -> heap.get(1),
                "addresses at capacity should be rejected");
        expectThrows(IllegalArgumentException.class, () -> new Heap(-1),
                "negative capacity should be rejected");
        expectThrows(IllegalArgumentException.class, () -> heap.allocate("bad", 0, "test"),
                "non-positive object sizes should be rejected");
    }

    private static void testReferencesAndRoots() {
        Heap heap = new Heap(3);
        int from = heap.allocate("from");
        int to = heap.allocate("to");
        heap.addReference(from, to);
        heap.addReference(from, to);
        check(heap.get(from).getOutgoingReferenceCount() == 1,
                "duplicate references should not be added");
        check(heap.get(to).getIncomingReferenceCount() == 1,
                "incoming references should be tracked");
        heap.removeReference(from, to);
        check(heap.get(from).getOutgoingReferenceCount() == 0,
                "reference removal should update the source");
        check(heap.get(to).getIncomingReferenceCount() == 0,
                "reference removal should update the target");

        heap.addRoot(from);
        heap.addRoot(2);
        check(heap.isRoot(from), "allocated objects can become roots");
        check(!heap.isRoot(2), "empty slots cannot become roots");
        heap.removeRoot(from);
        check(!heap.isRoot(from), "roots should be removable");
    }

    private static void testCycleAndRecursiveMarking() {
        Heap heap = new Heap(4);
        for (int i = 0; i < 4; i++) heap.allocate("node-" + i);
        heap.addRoot(0);
        heap.addReference(0, 1);
        heap.addReference(1, 2);
        heap.addReference(2, 0);
        new MarkAndSweepCollector().collect(heap);
        check(heap.usedSlots() == 3, "reachable cycles should survive collection");
        check(heap.get(3) == null, "an isolated object should be swept");
    }

    private static void testInstrumentation() {
        Heap heap = new Heap(4, 5);
        int first = heap.allocate("first", 64, "parser");
        heap.advanceTick(3);
        int second = heap.allocate("second", 128, "cache");
        heap.addRoot(first);
        heap.addReference(first, first);

        HeapObject firstObject = heap.get(first);
        HeapObject secondObject = heap.get(second);
        check(firstObject.getId() != secondObject.getId(), "object identities must be unique");
        check(firstObject.getSizeBytes() == 64, "synthetic size should be retained");
        check(firstObject.getAllocationTick() == 0, "allocation tick should be captured");
        check(secondObject.getAllocationTick() == 3, "advanced tick should be captured");
        check("parser".equals(firstObject.getAllocationSite()),
                "allocation site should be captured");
        check(firstObject.getHeapUtilizationAtAllocation() == 0.25,
                "allocation utilization should include the new object");
        check(firstObject.getGeneration() == Generation.YOUNG,
                "new objects should start in the young generation");
        firstObject.setGeneration(Generation.OLD);
        check(firstObject.getGeneration() == Generation.OLD,
                "generation should be configurable by future policies");

        new MarkAndSweepCollector().collect(heap);
        check(firstObject.getAge() == 1 && firstObject.getGcCyclesSurvived() == 1,
                "live-object survival metadata should advance after collection");
        check(secondObject.getDeathTick() == 3,
                "reclaimed objects should capture a ground-truth death tick");
        check(heap.getReclaimedObjects().contains(secondObject),
                "reclaimed objects should remain available for later telemetry");
    }

    private static void testInstrumentationWithMovingCollectors() {
        assertMovingCollectorInstrumentation(new MarkAndCompactCollector());
        assertMovingCollectorInstrumentation(new CopyingCollector());
    }

    private static void assertMovingCollectorInstrumentation(GarbageCollector collector) {
        Heap heap = new Heap(3);
        HeapObject garbage = objectAt(heap, heap.allocate("garbage"));
        int live = heap.allocate("live");
        heap.addRoot(live);
        collector.collect(heap);
        check(garbage.getDeathTick() != null,
                "moving collectors should record ground-truth deaths");
        check(heap.get(0).getGcCyclesSurvived() == 1,
                "moving collectors should update survivor metadata");
    }

    private static HeapObject objectAt(Heap heap, int address) { return heap.get(address); }

    private static void expectThrows(Class<? extends Throwable> type, Runnable action,
                                     String message) {
        try {
            action.run();
        } catch (Throwable thrown) {
            if (type.isInstance(thrown)) return;
            throw new AssertionError(message + ": wrong exception " + thrown);
        }
        throw new AssertionError(message + ": no exception thrown");
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}

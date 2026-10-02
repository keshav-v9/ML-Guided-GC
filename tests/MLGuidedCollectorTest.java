package tests;

import core.Generation;
import core.Heap;
import core.HeapObject;
import gc.MLGuidedCollector;
import gc.MarkAndCompactCollector;
import gc.MarkAndSweepCollector;
import ml.FakeLifetimePredictor;
import ml.LifetimePredictor;

public final class MLGuidedCollectorTest {
    private MLGuidedCollectorTest() {}

    public static void main(String[] args) {
        testPredictionControlsPromotionOnly();
        testMovingCollectorCompatibility();
        testMinimumAge();
        testPredictionFailureFailsOpen();
        System.out.println("All ML-guided collector tests passed.");
    }

    private static void testPredictionControlsPromotionOnly() {
        Heap heap = new Heap(4);
        int small = heap.allocate("small", 32, "short-site");
        int large = heap.allocate("large", 256, "long-site");
        heap.allocate("unreachable", 512, "long-site");
        heap.addRoot(small);
        heap.addRoot(large);

        MLGuidedCollector collector = new MLGuidedCollector(new MarkAndSweepCollector(),
                new FakeLifetimePredictor(features ->
                        features.getSizeBytes() >= 128 ? 0.95 : 0.05, 0.70));
        collector.collect(heap);

        check(heap.usedSlots() == 2, "prediction must not retain unreachable objects");
        check(heap.get(small).getGeneration() == Generation.YOUNG,
                "short-lived predictions should remain young");
        check(heap.get(large).getGeneration() == Generation.OLD,
                "long-lived predictions should be promoted");
        check(collector.getPredictionCount() == 2,
                "only live young objects should be scored");
        check(collector.getPromotionCount() == 1, "one object should be promoted");
        check(collector.getPredictionFailureCount() == 0, "valid predictions should succeed");
    }

    private static void testMovingCollectorCompatibility() {
        Heap heap = new Heap(4);
        heap.allocate("garbage", 8, "short-site");
        int live = heap.allocate("live", 256, "long-site");
        HeapObject liveObject = heap.get(live);
        heap.addRoot(live);

        MLGuidedCollector collector = new MLGuidedCollector(new MarkAndCompactCollector(),
                new FakeLifetimePredictor(features -> 1.0, 0.70));
        collector.collect(heap);

        check(heap.get(0) == liveObject, "moving collectors should preserve object identity");
        check(heap.isRoot(0), "moving collectors should still rewrite roots");
        check(liveObject.getGeneration() == Generation.OLD,
                "promotion should apply after relocation");
    }

    private static void testMinimumAge() {
        Heap heap = new Heap(1);
        int address = heap.allocate("survivor", 64, "site");
        heap.addRoot(address);
        MLGuidedCollector collector = new MLGuidedCollector(new MarkAndSweepCollector(),
                new FakeLifetimePredictor(features -> 1.0, 0.70), 2);

        collector.collect(heap);
        check(heap.get(address).getGeneration() == Generation.YOUNG,
                "objects younger than the policy age should not be scored");
        check(collector.getPredictionCount() == 0, "young object should not be scored yet");
        collector.collect(heap);
        check(heap.get(address).getGeneration() == Generation.OLD,
                "an eligible long-lived object should be promoted");
    }

    private static void testPredictionFailureFailsOpen() {
        Heap heap = new Heap(1);
        int address = heap.allocate("survivor");
        heap.addRoot(address);
        LifetimePredictor failing = features -> { throw new IllegalStateException("model down"); };
        MLGuidedCollector collector = new MLGuidedCollector(new MarkAndSweepCollector(), failing);

        collector.collect(heap);
        check(heap.get(address) != null, "a model failure must not reclaim a live object");
        check(heap.get(address).getGeneration() == Generation.YOUNG,
                "a failed prediction should preserve the default generation");
        check(collector.getPredictionFailureCount() == 1,
                "model failures should be observable in policy statistics");
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}

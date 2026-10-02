package bench;

import gc.GarbageCollector;
import gc.MLGuidedCollector;
import gc.MarkAndSweepCollector;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Locale;
import ml.FakeLifetimePredictor;
import ml.LifetimePredictor;
import ml.OnnxLifetimePredictor;
import workload.WorkloadConfig;
import workload.WorkloadGenerator;
import workload.WorkloadResult;
import workload.WorkloadType;

/** Repeatable end-to-end comparison of baseline and ML-guided collection policy. */
public final class PolicyBenchmark {
    private static final int CAPACITY = 10_000;
    private static final int OBJECTS = 10_000;
    private static final int WARMUPS = 2;
    private static final int REPETITIONS = 7;
    private static final long SEED = 42L;

    private PolicyBenchmark() {}

    public static void main(String[] args) {
        if (args.length != 0 && args.length != 2)
            throw new IllegalArgumentException(
                    "usage: java bench.PolicyBenchmark [model.onnx model.properties]");
        Path model = args.length == 2 ? Paths.get(args[0]) : null;
        Path properties = args.length == 2 ? Paths.get(args[1]) : null;

        for (int index = 0; index < WARMUPS; index++) {
            runBaseline();
            runGuided(model, properties);
        }

        long[] baselineNanos = new long[REPETITIONS];
        long[] guidedNanos = new long[REPETITIONS];
        long predictions = 0L;
        long promotions = 0L;
        long failures = 0L;
        long inferenceNanos = 0L;
        for (int index = 0; index < REPETITIONS; index++) {
            ScenarioResult baseline = runBaseline();
            ScenarioResult guided = runGuided(model, properties);
            assertEquivalent(baseline, guided);
            baselineNanos[index] = baseline.elapsedNanos;
            guidedNanos[index] = guided.elapsedNanos;
            predictions += guided.predictions;
            promotions += guided.promotions;
            failures += guided.failures;
            inferenceNanos += guided.inferenceNanos;
        }

        System.out.printf(Locale.ROOT,
                "Policy benchmark: %,d objects, %d warmups, %d measured runs%n",
                OBJECTS, WARMUPS, REPETITIONS);
        printTiming("Exact tracing", baselineNanos);
        printTiming(model == null ? "ML policy (fake)" : "ML policy (ONNX)", guidedNanos);
        System.out.printf(Locale.ROOT,
                "ML totals: %,d predictions | %,d promotions | %,d failures | %.2f ms inference%n",
                predictions, promotions, failures, inferenceNanos / 1_000_000.0);
    }

    private static ScenarioResult runBaseline() {
        return run(new MarkAndSweepCollector(), null);
    }

    private static ScenarioResult runGuided(Path model, Path properties) {
        LifetimePredictor predictor = model == null
                ? new FakeLifetimePredictor(features -> {
                    double sizeSignal = Math.min(0.55, features.getSizeBytes() / 1024.0);
                    double survivalSignal = Math.min(0.35,
                            features.getGcCyclesSurvived() * 0.12);
                    return Math.min(1.0, 0.10 + sizeSignal + survivalSignal);
                }, 0.70)
                : new OnnxLifetimePredictor(model, properties);
        try {
            MLGuidedCollector collector = new MLGuidedCollector(
                    new MarkAndSweepCollector(), predictor);
            return run(collector, collector);
        } finally {
            if (predictor instanceof AutoCloseable) {
                try {
                    ((AutoCloseable) predictor).close();
                } catch (Exception exception) {
                    throw new IllegalStateException("could not close predictor", exception);
                }
            }
        }
    }

    private static ScenarioResult run(GarbageCollector collector, MLGuidedCollector guided) {
        WorkloadConfig config = new WorkloadConfig(WorkloadType.MIXED, CAPACITY, OBJECTS,
                SEED, 500, 3, 30, 0.20);
        long start = System.nanoTime();
        WorkloadResult result = WorkloadGenerator.run(config, collector);
        long elapsed = System.nanoTime() - start;
        return new ScenarioResult(elapsed, result.getHeap().usedSlots(),
                result.getHeap().getReclaimedObjects().size(), result.getCollectionCount(),
                guided == null ? 0L : guided.getPredictionCount(),
                guided == null ? 0L : guided.getPromotionCount(),
                guided == null ? 0L : guided.getPredictionFailureCount(),
                guided == null ? 0L : guided.getInferenceNanos());
    }

    private static void assertEquivalent(ScenarioResult baseline, ScenarioResult guided) {
        if (baseline.liveObjects != guided.liveObjects
                || baseline.reclaimedObjects != guided.reclaimedObjects
                || baseline.collections != guided.collections)
            throw new AssertionError("ML policy changed exact collection semantics");
    }

    private static void printTiming(String name, long[] values) {
        long[] sorted = values.clone();
        Arrays.sort(sorted);
        double median = sorted[sorted.length / 2] / 1_000_000.0;
        int p95Index = (int) Math.ceil(sorted.length * 0.95) - 1;
        double p95 = sorted[Math.max(0, p95Index)] / 1_000_000.0;
        System.out.printf(Locale.ROOT, "%-18s median %8.2f ms | p95 %8.2f ms%n",
                name, median, p95);
    }

    private static final class ScenarioResult {
        private final long elapsedNanos;
        private final int liveObjects;
        private final int reclaimedObjects;
        private final int collections;
        private final long predictions;
        private final long promotions;
        private final long failures;
        private final long inferenceNanos;

        private ScenarioResult(long elapsedNanos, int liveObjects, int reclaimedObjects,
                               int collections, long predictions, long promotions,
                               long failures, long inferenceNanos) {
            this.elapsedNanos = elapsedNanos;
            this.liveObjects = liveObjects;
            this.reclaimedObjects = reclaimedObjects;
            this.collections = collections;
            this.predictions = predictions;
            this.promotions = promotions;
            this.failures = failures;
            this.inferenceNanos = inferenceNanos;
        }
    }
}

package tests;

import core.HeapObject;
import gc.CopyingCollector;
import gc.MarkAndCompactCollector;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import ml.FakeLifetimePredictor;
import ml.ObjectFeatures;
import ml.Prediction;
import telemetry.CensoringPolicy;
import telemetry.LifetimeLabel;
import telemetry.ObjectTelemetry;
import telemetry.TelemetryRecorder;
import workload.WorkloadConfig;
import workload.WorkloadGenerator;
import workload.WorkloadResult;
import workload.WorkloadType;

public final class WorkloadTelemetryTest {
    private static final int CAPACITY = 64;
    private static final int OBJECTS = 40;
    private static final long SEED = 42L;

    private WorkloadTelemetryTest() {}

    public static void main(String[] args) throws Exception {
        testDeterministicWorkload();
        testWorkloadFamilies();
        testMovingCollectorWorkloads();
        testTelemetryLabelsAndCensoring();
        testCsvOutputAndLeakageGuard();
        testConfigurationValidation();
        testFakePredictor();
        System.out.println("All workload and telemetry tests passed.");
    }

    private static void testDeterministicWorkload() {
        WorkloadConfig config = config(WorkloadType.MIXED);
        WorkloadResult first = WorkloadGenerator.run(config);
        WorkloadResult second = WorkloadGenerator.run(config);
        check(fingerprint(first).equals(fingerprint(second)),
                "the same seed and configuration must reproduce the same workload");

        WorkloadResult different = WorkloadGenerator.run(new WorkloadConfig(
                WorkloadType.MIXED, CAPACITY, OBJECTS, SEED + 1, 10, 3, 30, 0.20));
        check(!fingerprint(first).equals(fingerprint(different)),
                "a different seed should change the generated workload");
    }

    private static void testWorkloadFamilies() {
        WorkloadResult shortLived = WorkloadGenerator.run(config(WorkloadType.SHORT_LIVED));
        check(shortLived.getHeap().getReclaimedObjects().size() == OBJECTS,
                "short-lived workloads should reclaim every object by the final collection");

        WorkloadResult longLived = WorkloadGenerator.run(config(WorkloadType.LONG_LIVED));
        check(longLived.getHeap().usedSlots() > 0,
                "long-lived workloads should retain a rooted graph at the horizon");

        WorkloadResult mixed = WorkloadGenerator.run(config(WorkloadType.MIXED));
        check(mixed.getHeap().usedSlots() > 0
                        && !mixed.getHeap().getReclaimedObjects().isEmpty(),
                "mixed workloads should contain live and reclaimed objects");

        WorkloadResult phased = WorkloadGenerator.run(config(WorkloadType.PHASE_CHANGING));
        boolean sawShortPhase = false;
        boolean sawLongPhase = false;
        for (HeapObject object : phased.getHeap().getAllocatedObjects()) {
            sawShortPhase |= "phase-short".equals(object.getAllocationSite());
            sawLongPhase |= "phase-long".equals(object.getAllocationSite());
        }
        check(sawShortPhase && sawLongPhase,
                "phase-changing workloads should expose both sequential phases");

        WorkloadResult graph = WorkloadGenerator.run(config(WorkloadType.GRAPH_STRESS));
        boolean fanOut = false;
        boolean cycle = false;
        Set<Integer> sizes = new HashSet<>();
        List<HeapObject> objects = graph.getHeap().getAllocatedObjects();
        for (HeapObject object : objects) {
            sizes.add(object.getSizeBytes());
            fanOut |= object.getOutgoingReferenceCount() > 1;
        }
        for (int sourceAddress = 0; sourceAddress < graph.getHeap().capacity(); sourceAddress++) {
            HeapObject source = graph.getHeap().get(sourceAddress);
            if (source == null) continue;
            for (int targetAddress : source.getReferences()) {
                if (targetAddress >= 0 && targetAddress < graph.getHeap().capacity()) {
                    HeapObject target = graph.getHeap().get(targetAddress);
                    if (target != null && target.getReferences().contains(sourceAddress)) cycle = true;
                }
            }
        }
        check(fanOut, "graph-stress workloads should contain fan-out");
        check(cycle, "graph-stress workloads should contain cycles");
        check(sizes.size() > 1, "graph-stress workloads should vary object sizes");
    }

    private static void testTelemetryLabelsAndCensoring() {
        WorkloadResult result = WorkloadGenerator.run(config(WorkloadType.MIXED));
        TelemetryRecorder recorder = new TelemetryRecorder();
        List<ObjectTelemetry> excluded = recorder.record(result, 12,
                CensoringPolicy.EXCLUDE);
        for (ObjectTelemetry row : excluded)
            check(!row.isCensored(), "EXCLUDE must omit every right-censored object");

        List<ObjectTelemetry> thresholdLabeled = recorder.record(result, 12,
                CensoringPolicy.LABEL_LONG_IF_THRESHOLD_EXCEEDED);
        check(thresholdLabeled.size() > excluded.size(),
                "eligible long-lived censored objects should be retained");
        boolean sawShort = false;
        boolean sawLong = false;
        boolean sawCensored = false;
        for (ObjectTelemetry row : thresholdLabeled) {
            sawShort |= row.getLabel() == LifetimeLabel.SHORT_LIVED;
            sawLong |= row.getLabel() == LifetimeLabel.LONG_LIVED;
            sawCensored |= row.isCensored();
            if (row.isCensored()) {
                check(row.getLabel() == LifetimeLabel.LONG_LIVED,
                        "a censored row may only be labeled after exceeding the threshold");
                check(row.getLifetimeTicks() > 12,
                        "censored long labels must exceed the configured threshold");
            }
        }
        check(sawShort && sawLong && sawCensored,
                "mixed telemetry should include both labels and explicit censoring");
    }

    private static void testMovingCollectorWorkloads() {
        WorkloadConfig config = new WorkloadConfig(
                WorkloadType.MIXED, 40, 25, 7, 5, 3, 30, 0.20);
        WorkloadResult compacted = WorkloadGenerator.run(config, new MarkAndCompactCollector());
        WorkloadResult copied = WorkloadGenerator.run(config, new CopyingCollector());
        check(compacted.getHeap().getAllocatedObjects().size() == 25,
                "mark-and-compact workloads should track stable object identities");
        check(copied.getHeap().getAllocatedObjects().size() == 25,
                "copying workloads should track stable object identities");
    }

    private static void testCsvOutputAndLeakageGuard() throws Exception {
        WorkloadResult result = WorkloadGenerator.run(config(WorkloadType.SHORT_LIVED));
        TelemetryRecorder recorder = new TelemetryRecorder();
        List<ObjectTelemetry> rows = recorder.record(result, 12, CensoringPolicy.EXCLUDE);
        Path output = Files.createTempFile("minigc-telemetry-", ".csv");
        try {
            recorder.writeCsv(output, rows);
            List<String> lines = Files.readAllLines(output, StandardCharsets.UTF_8);
            check(lines.size() == rows.size() + 1,
                    "CSV should contain one header and one line per labeled object");
            check(ObjectTelemetry.CSV_HEADER.equals(lines.get(0)),
                    "CSV header should be stable and complete");
            check(!ObjectTelemetry.MODEL_FEATURE_COLUMNS.contains("lifetime_ticks")
                            && !ObjectTelemetry.MODEL_FEATURE_COLUMNS.contains("death_tick")
                            && !ObjectTelemetry.MODEL_FEATURE_COLUMNS.contains("label"),
                    "post-death and target columns must be excluded from model inputs");
        } finally {
            Files.deleteIfExists(output);
        }
    }

    private static void testConfigurationValidation() {
        expectThrows(IllegalArgumentException.class, () -> new WorkloadConfig(
                WorkloadType.MIXED, 10, 10, 1, 0, 3, 30, 0.2));
        expectThrows(IllegalArgumentException.class, () -> new WorkloadConfig(
                WorkloadType.MIXED, 10, 10, 1, 2, 3, 30, 1.1));
    }

    private static void testFakePredictor() {
        FakeLifetimePredictor predictor = new FakeLifetimePredictor(
                features -> features.getSizeBytes() >= 128 ? 0.9 : 0.1, 0.7);
        Prediction prediction = predictor.predict(new ObjectFeatures(
                256, 1, 2, 0.5, 1.2, 1, 1, "test"));
        check(prediction.isLongLived() && prediction.getProbability() == 0.9,
                "the fake predictor should provide deterministic policy tests");
        check(prediction.getInferenceNanos() >= 0,
                "predictors should report inference overhead");
    }

    private static WorkloadConfig config(WorkloadType type) {
        return new WorkloadConfig(type, CAPACITY, OBJECTS, SEED, 10, 3, 30, 0.20);
    }

    private static String fingerprint(WorkloadResult result) {
        StringBuilder value = new StringBuilder();
        for (HeapObject object : result.getHeap().getAllocatedObjects()) {
            value.append(object.getId()).append(':').append(object.getSizeBytes()).append(':')
                    .append(object.getAllocationTick()).append(':')
                    .append(object.getDeathTick()).append(':')
                    .append(object.getAllocationSite()).append(':')
                    .append(object.getReferences()).append('|');
        }
        return value.toString();
    }

    private static void expectThrows(Class<? extends Throwable> type, Runnable action) {
        try {
            action.run();
        } catch (Throwable thrown) {
            if (type.isInstance(thrown)) return;
            throw new AssertionError("wrong exception: " + thrown);
        }
        throw new AssertionError("expected " + type.getSimpleName());
    }

    private static void check(boolean condition, String message) {
        if (!condition) throw new AssertionError(message);
    }
}

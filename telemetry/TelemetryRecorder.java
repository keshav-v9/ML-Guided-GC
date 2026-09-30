package telemetry;

import core.Heap;
import core.HeapObject;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import workload.WorkloadConfig;
import workload.WorkloadResult;

/** Builds a leakage-safe labeled dataset separately from workload generation. */
public final class TelemetryRecorder {
    public List<ObjectTelemetry> record(WorkloadResult result, long lifetimeThreshold,
                                        CensoringPolicy censoringPolicy) {
        if (result == null || censoringPolicy == null)
            throw new IllegalArgumentException("result and censoringPolicy must not be null");
        if (lifetimeThreshold < 0)
            throw new IllegalArgumentException("lifetimeThreshold must not be negative");

        Heap heap = result.getHeap();
        WorkloadConfig config = result.getConfig();
        List<ObjectTelemetry> rows = new ArrayList<>();
        for (HeapObject object : heap.getAllocatedObjects()) {
            Long deathTick = object.getDeathTick();
            boolean censored = deathTick == null;
            long observedLifetime = (censored ? heap.getCurrentTick() : deathTick)
                    - object.getAllocationTick();

            LifetimeLabel label;
            if (censored) {
                if (censoringPolicy == CensoringPolicy.EXCLUDE
                        || observedLifetime <= lifetimeThreshold) continue;
                label = LifetimeLabel.LONG_LIVED;
            } else {
                label = observedLifetime <= lifetimeThreshold
                        ? LifetimeLabel.SHORT_LIVED : LifetimeLabel.LONG_LIVED;
            }

            rows.add(new ObjectTelemetry(object.getId(),
                    config.getType().name().toLowerCase(), config.getSeed(),
                    object.getSizeBytes(), object.getIncomingReferenceCount(),
                    object.getOutgoingReferenceCount(), object.getHeapUtilizationAtAllocation(),
                    object.getAllocationRate(), object.getGcCyclesSurvived(), object.getAge(),
                    object.getAllocationSite(), observedLifetime, label, censored,
                    censoringPolicy));
        }
        return Collections.unmodifiableList(rows);
    }

    public void writeCsv(Path output, List<ObjectTelemetry> rows) throws IOException {
        if (output == null || rows == null)
            throw new IllegalArgumentException("output and rows must not be null");
        Path parent = output.toAbsolutePath().getParent();
        if (parent != null) Files.createDirectories(parent);
        try (BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            writer.write(ObjectTelemetry.CSV_HEADER);
            writer.newLine();
            for (ObjectTelemetry row : rows) {
                writer.write(row.toCsvRow());
                writer.newLine();
            }
        }
    }
}

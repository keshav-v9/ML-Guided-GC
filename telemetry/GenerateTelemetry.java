package telemetry;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import workload.WorkloadConfig;
import workload.WorkloadGenerator;
import workload.WorkloadResult;
import workload.WorkloadType;

/** Small dependency-free CLI for reproducible telemetry CSV generation. */
public final class GenerateTelemetry {
    private GenerateTelemetry() {}

    public static void main(String[] args) throws Exception {
        Map<String, String> options = parseOptions(args);
        int objects = integer(options, "objects", 1_000);
        int capacity = integer(options, "capacity", Math.max(1, objects));
        long threshold = longValue(options, "lifetime-threshold", 10L);
        CensoringPolicy policy = CensoringPolicy.valueOf(options
                .getOrDefault("censoring", "label-long-if-threshold-exceeded")
                .toUpperCase(Locale.ROOT).replace('-', '_'));
        Path output = Paths.get(options.getOrDefault("output", "data/raw/telemetry.csv"));

        TelemetryRecorder recorder = new TelemetryRecorder();
        List<ObjectTelemetry> rows = new ArrayList<>();
        int totalAllocations = 0;
        int totalCollections = 0;
        for (String workloadName : options.getOrDefault("workloads",
                options.getOrDefault("workload", "mixed")).split(",")) {
            WorkloadType type = WorkloadType.valueOf(workloadName.trim()
                    .toUpperCase(Locale.ROOT).replace('-', '_'));
            for (String seedText : options.getOrDefault("seeds",
                    options.getOrDefault("seed", "42")).split(",")) {
                long seed = Long.parseLong(seedText.trim());
                WorkloadConfig defaults = WorkloadConfig.defaults(type, capacity, objects, seed);
                WorkloadConfig config = new WorkloadConfig(type, capacity, objects, seed,
                        integer(options, "collection-interval", defaults.getCollectionInterval()),
                        integer(options, "short-lifetime", defaults.getShortLifetimeTicks()),
                        integer(options, "long-lifetime", defaults.getLongLifetimeTicks()),
                        decimal(options, "long-fraction", defaults.getLongLivedFraction()));
                WorkloadResult result = WorkloadGenerator.run(config);
                rows.addAll(recorder.record(result, threshold, policy));
                totalAllocations += objects;
                totalCollections += result.getCollectionCount();
            }
        }
        recorder.writeCsv(output, rows);
        System.out.printf(Locale.ROOT,
                "Wrote %,d labeled rows from %,d allocations to %s (%d collections).%n",
                rows.size(), totalAllocations, output, totalCollections);
    }

    private static Map<String, String> parseOptions(String[] args) {
        Map<String, String> options = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i += 2) {
            if (!args[i].startsWith("--") || i + 1 >= args.length)
                throw new IllegalArgumentException("expected --name value pairs");
            options.put(args[i].substring(2), args[i + 1]);
        }
        return options;
    }

    private static int integer(Map<String, String> options, String name, int fallback) {
        return Integer.parseInt(options.getOrDefault(name, Integer.toString(fallback)));
    }

    private static long longValue(Map<String, String> options, String name, long fallback) {
        return Long.parseLong(options.getOrDefault(name, Long.toString(fallback)));
    }

    private static double decimal(Map<String, String> options, String name, double fallback) {
        return Double.parseDouble(options.getOrDefault(name, Double.toString(fallback)));
    }
}

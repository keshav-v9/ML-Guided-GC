# JVM Garbage Collection Simulator

An educational Java simulation of three tracing garbage collectors:

- mark and sweep (non-moving)
- mark and compact (moving, address-order compaction)
- copying collection (moving, breadth-first traversal)

Objects live in an address-based `Heap`; roots and inter-object references are heap
addresses. Moving collectors update both roots and references.

The project currently covers the first two MiniGC-ML milestones: the collector
baseline and instrumentation, plus deterministic workloads and leakage-safe
telemetry generation. Objects record synthetic size, logical allocation/death
ticks, allocation context, generation, reference counts, and collection
survival counts. Reachability remains entirely graph-based; this milestone does
not use ML.

## Run

Requires JDK 8 or newer.

```sh
javac Main.java core/*.java gc/*.java workload/*.java telemetry/*.java bench/*.java tests/*.java
java Main
java tests.GarbageCollectorTest
java tests.WorkloadTelemetryTest
java bench.Benchmark
```

Generate a deterministic telemetry dataset:

```sh
java telemetry.GenerateTelemetry \
  --workload mixed --objects 1000 --capacity 1000 --seed 42 \
  --collection-interval 25 --lifetime-threshold 10 \
  --censoring label-long-if-threshold-exceeded \
  --output data/raw/telemetry.csv
```

Supported workloads are `short-lived`, `long-lived`, `mixed`,
`phase-changing`, and `graph-stress`. Workload configuration includes the seed,
heap capacity, collection interval, short/long lifetime settings, and mixed
long-lived fraction.

The CSV contains `lifetime_ticks` and `label` for analysis, plus `censored` and
`labeling_rule` for experiment auditing. `ObjectTelemetry.MODEL_FEATURE_COLUMNS`
is the authoritative training feature list and deliberately excludes all
post-death and target fields.

`Allocator` retries an allocation after invoking its configured collector when the
heap is full. `WorkloadGenerator` produces deterministic graphs and lifetime
simulations for comparisons.
See `ARCHITECTURE.md` for component responsibilities and instrumentation
semantics.

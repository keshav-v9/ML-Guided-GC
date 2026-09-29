# JVM Garbage Collection Simulator

An educational Java simulation of three tracing garbage collectors:

- mark and sweep (non-moving)
- mark and compact (moving, address-order compaction)
- copying collection (moving, breadth-first traversal)

Objects live in an address-based `Heap`; roots and inter-object references are heap
addresses. Moving collectors update both roots and references.

The project currently covers the first MiniGC-ML milestone: the collector
baseline plus instrumented object metadata. Objects record synthetic size,
logical allocation/death ticks, allocation context, generation, reference
counts, and collection survival counts. Reachability remains entirely
graph-based; this milestone does not use ML.

## Run

Requires JDK 8 or newer.

```sh
javac Main.java core/*.java gc/*.java workload/*.java bench/*.java tests/*.java
java Main
java tests.GarbageCollectorTest
java bench.Benchmark
```

`Allocator` retries an allocation after invoking its configured collector when the
heap is full. `WorkloadGenerator` produces deterministic graphs for comparisons.
See `ARCHITECTURE.md` for component responsibilities and instrumentation
semantics.

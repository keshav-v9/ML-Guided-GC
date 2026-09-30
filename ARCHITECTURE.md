# MiniGC Architecture

MiniGC is an address-based garbage-collection simulator. It models collector
algorithms; it does not replace or modify the JVM's own garbage collector.

## Current milestone

This repository implements Phases 0-3 of the MiniGC-ML specification: a stable
tracing-GC baseline, instrumented heap objects, deterministic workload families,
and leakage-safe telemetry CSV generation. ML does not affect reachability.
Later policies may use metadata only for safe placement, promotion, or
scheduling decisions.

## Components

- `core.Heap` owns fixed-capacity address slots, roots, logical simulation time,
  allocation-rate tracking, and the history of reclaimed objects.
- `core.HeapObject` owns stable object identity, outgoing references, mark state,
  and lifetime/policy metadata. References remain integer heap addresses.
- `core.Allocator` retries one failed allocation after invoking its configured
  collector.
- `gc.GarbageCollector` is the common collection interface.
- `gc.MarkAndSweepCollector` traces from roots without moving live objects.
- `gc.MarkAndCompactCollector` traces, compacts live objects in address order,
  and rewrites roots and references.
- `gc.CopyingCollector` discovers live objects breadth-first, copies them into a
  contiguous logical region, and rewrites roots and references.
- `workload.WorkloadGenerator` creates deterministic seeded short-lived,
  long-lived, mixed, phase-changing, and graph-stress simulations.
- `telemetry.TelemetryRecorder` labels completed lifetimes, applies an explicit
  right-censoring policy, and writes model-independent CSV data.
- `telemetry.GenerateTelemetry` is the dependency-free dataset-generation CLI.
- `bench.Benchmark` performs a lightweight, non-JMH collector comparison.

## Data flow

```text
Allocator / Workload
        |
        v
Heap.allocate -> HeapObject metadata + address
        |
        v
roots and address references
        |
        v
GarbageCollector.collect
        |-- graph tracing determines reachability
        |-- reachability transitions assign semantic deathTick
        |-- reclaimed objects enter retained telemetry history
        `-- survivors advance age and gcCyclesSurvived
```

## Instrumentation semantics

- `id` is stable for an object's lifetime; `address` may change under a moving
  collector.
- `allocationTick` uses `Heap`'s explicit logical clock. Workloads advance it
  with `advanceTick`.
- `deathTick` is assigned when a collector proves an object unreachable.
- Workload simulations additionally trace after root expiry, so `deathTick`
  records the reachability transition rather than the later collection pause.
- New objects are `YOUNG`; future generational policies may promote them to
  `OLD`.
- `heapUtilizationAtAllocation` is measured immediately after allocation.
- `allocationRate` is allocations per tick across the heap's configurable
  trailing window, including the current allocation.
- `incomingReferenceCount` is maintained during normal reference changes and
  recomputed after moving or sweeping collections.
- `age` and `gcCyclesSurvived` currently both count completed collections during
  which the object remained reachable. They are separate fields so a later
  prediction policy can define age independently without changing telemetry.

## Correctness invariants

1. Reachability is determined only by tracing from roots.
2. A reachable object is never reclaimed.
3. A moving collector updates roots and all live references to new addresses.
4. Marks are cleared before a collection completes.
5. Reclaimed-object history retains metadata needed to construct later labels,
   but reclaimed objects are no longer addressable from the heap.

## Deliberate limits of this milestone

There is no generational collector, Python training pipeline, ONNX runtime,
ML-guided policy, or production benchmark harness yet.
The repository keeps its existing direct `javac` workflow; adopting JUnit 5 and
a build tool belongs in the next infrastructure milestone so no unused external
dependency is introduced.

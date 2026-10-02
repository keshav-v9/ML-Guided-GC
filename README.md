# JVM Garbage Collection Simulator

An educational Java simulation of three tracing garbage collectors:

- mark and sweep (non-moving)
- mark and compact (moving, address-order compaction)
- copying collection (moving, breadth-first traversal)

Objects live in an address-based `Heap`; roots and inter-object references are heap
addresses. Moving collectors update both roots and references.

The project covers all five MiniGC-ML milestones: tracing-collector baselines,
instrumentation, deterministic workloads and leakage-safe telemetry, model
training and ONNX export/runtime inference, and a safe ML-guided promotion
policy with an end-to-end benchmark. Objects record synthetic size, logical
allocation/death ticks, allocation context, generation, reference counts, and
collection survival counts. Reachability always remains graph-based; ML can
change generation metadata but can never decide that an object is dead.

## ML results summary

The ML pipeline turns leakage-safe object telemetry into a deployable ONNX
lifetime classifier. On a reproducible synthetic dataset of 14,959 labeled
objects, the selected logistic-regression model achieved the following held-out
results:

| F1 | ROC AUC | Precision | Recall | Mean inference | Model size |
|---:|---:|---:|---:|---:|---:|
| 0.8578 | 0.9334 | 0.8323 | 0.8849 | 555 ns | 1,023 B |

Latency-aware selection preferred logistic regression over a 3.7 MB random
forest with similar validation F1. ONNX export preserved the Python model's
predictions to within 5.61e-8, and all 20 Java golden-test predictions matched.

The ML-guided collector safely promoted 1,344 predicted long-lived objects with
zero prediction failures while producing exactly the same liveness and
collection counts as the baseline collector. The benchmark now includes JVM
warmups, repeated measurements, median/p95 latency, real ONNX inference, and
prediction/promotion counters:

| 10,000-object policy benchmark | Median | p95 |
|---|---:|---:|
| Exact tracing baseline | 4.08 ms | 6.70 ms |
| ML-guided tracing | 6.15 ms | 8.64 ms |

This is a correctness and observability improvement, not yet a GC speedup: ML
currently updates `YOUNG`/`OLD` metadata while both generations are still traced
together, adding 2.07 ms median overhead in this reference run. The next
performance step is to use those promotions in separate young- and
old-generation collection paths. Results were measured on synthetic workloads
and will vary by machine.

## Run

Requires JDK 8 or newer.

```sh
javac Main.java core/*.java gc/*.java workload/*.java telemetry/*.java bench/*.java tests/*.java
java Main
java tests.GarbageCollectorTest
java tests.WorkloadTelemetryTest
java tests.MLGuidedCollectorTest
java bench.Benchmark
java bench.PolicyBenchmark
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

## Train and use the lifetime model

Install the Python dependencies in a virtual environment, then generate data
from at least three workload/seed runs so the train, validation, and test groups
remain disjoint:

```sh
python3 -m venv .venv
.venv/bin/pip install -r ml/requirements.txt
java telemetry.GenerateTelemetry \
  --workloads short-lived,long-lived,mixed,phase-changing,graph-stress \
  --seeds 41,42,43 --objects 1000 --capacity 1000 \
  --lifetime-threshold 10 \
  --censoring label-long-if-threshold-exceeded \
  --output data/raw/telemetry.csv
.venv/bin/python -m ml.train \
  --data data/raw/telemetry.csv --output ml/artifacts/latest
.venv/bin/python -m ml.export_onnx \
  --model-dir ml/artifacts/latest --golden-data data/raw/telemetry.csv \
  --output ml/models/lifetime.onnx
```

XGBoost is included as a candidate when its native OpenMP runtime is available;
otherwise training continues portably with logistic regression and random
forest.

`MLGuidedCollector` wraps any exact collector and promotes surviving young
objects predicted to be long-lived. Prediction failures fail open: the object
stays young and normal tracing continues. To benchmark a real exported model,
put the ONNX Runtime Java JAR on the classpath and run:

```sh
java -cp ".:path/to/onnxruntime.jar" bench.PolicyBenchmark \
  ml/models/lifetime.onnx ml/models/lifetime.properties
```

See `ARCHITECTURE.md` for component responsibilities and instrumentation
semantics.

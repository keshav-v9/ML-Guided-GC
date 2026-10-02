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

## Results

The following reference results were reproduced on October 2, 2026, on Darwin
arm64 with OpenJDK 21.0.12.1 and Python 3.9.6. Timings vary by machine; accuracy
results describe the simulator's synthetic workloads and should not be read as
measurements of production JVM applications.

### Correctness

| Check | Result |
|---|---:|
| Garbage collector tests | Passed |
| Workload and telemetry tests | Passed |
| ML-guided policy tests | Passed |
| Python pipeline tests | 6/6 passed |
| Java/ONNX golden predictions | 20/20 matched |
| Maximum Python vs. ONNX probability difference | 5.61e-8 |

### Dataset

The reference dataset used five workload families, seeds 41-43, 1,000 objects
per run, and the explicit right-censoring policy shown in the generation command
below.

| Measurement | Result |
|---|---:|
| Allocations | 15,000 |
| Labeled rows | 14,959 |
| Short-lived labels | 9,060 |
| Long-lived labels | 5,899 |
| Uncensored rows | 14,719 |
| Right-censored rows | 240 |
| Collections | 150 |
| Training rows | 8,973 |
| Validation rows | 2,996 |
| Test rows | 2,990 |

Splits are grouped by workload and seed, so rows from one simulation run cannot
appear in more than one split.

### Model selection and held-out quality

Validation candidates were ranked by F1 with a latency penalty. XGBoost was
skipped on this machine because its optional OpenMP runtime was unavailable.

| Validation candidate | Precision | Recall | F1 | ROC AUC | Mean inference | Size | Selection score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.9341 | 0.4570 | 0.6137 | 0.9012 | 480 ns | 1,023 B | 0.6098 |
| Random forest | 0.9709 | 0.4489 | 0.6140 | 0.8971 | 11,264 ns | 3,727,849 B | 0.5889 |

Logistic regression won the latency-aware selection score. After refitting on
the combined training and validation data, it produced these held-out test
results:

| Precision | Recall | F1 | ROC AUC | Mean inference | p95 inference | Model size |
|---:|---:|---:|---:|---:|---:|---:|
| 0.8323 | 0.8849 | 0.8578 | 0.9334 | 555 ns | 789 ns | 1,023 B |

The test confusion matrix was 1,536 true negatives, 220 false positives, 142
false negatives, and 1,092 true positives.

### Collector comparison

One illustrative collection over the same deterministic 80,000-object graph
retained 53,552 reachable objects and reclaimed 26,448:

| Collector | Collection time |
|---|---:|
| Mark and sweep | 27.77 ms |
| Mark and compact | 66.56 ms |
| Copying | 41.22 ms |

The end-to-end policy benchmark uses two warmups and seven measured 10,000-object
runs. With the exported ONNX model:

| Policy | Median | p95 |
|---|---:|---:|
| Exact tracing | 4.08 ms | 6.70 ms |
| ML-guided tracing | 6.15 ms | 8.64 ms |

Across the seven measured ML runs there were 1,344 predictions, 1,344
promotions, zero prediction failures, and 12.18 ms of measured inference time.
Both policies produced identical liveness and collection counts.

The current policy uses predictions only to update `YOUNG`/`OLD` metadata; both
generations are still traced together. Consequently, this benchmark quantifies
inference overhead and correctness, not a collection-speed improvement. A true
performance comparison requires a collector that performs separate young- and
old-generation collections.

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

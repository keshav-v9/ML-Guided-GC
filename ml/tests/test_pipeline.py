from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.data import group_keys, labels, load_dataset, split_by_run
from ml.features import FeaturePreprocessor, LEAKAGE_COLUMNS, MODEL_INPUT_COLUMNS
from ml.modeling import evaluate_model, selection_score


def sample_frame() -> pd.DataFrame:
    rows = []
    for seed in range(1, 7):
        for index in range(12):
            long_lived = index % 3 == 0
            rows.append(
                {
                    "object_id": seed * 100 + index,
                    "workload_type": "mixed" if seed % 2 else "phase_changing",
                    "seed": seed,
                    "size_bytes": 256 if long_lived else 32,
                    "incoming_reference_count": index % 2,
                    "outgoing_reference_count": index % 4,
                    "heap_utilization_at_allocation": index / 12,
                    "allocation_rate": 1.0 + index / 10,
                    "gc_cycles_survived": 3 if long_lived else 0,
                    "age_at_prediction": 3 if long_lived else 0,
                    "allocation_site": f"site-{index % 2}",
                    "lifetime_ticks": 30 if long_lived else 2,
                    "label": "LONG_LIVED" if long_lived else "SHORT_LIVED",
                    "censored": False,
                    "labeling_rule": "EXCLUDE",
                }
            )
    return pd.DataFrame(rows)


def test_split_is_group_disjoint_and_reproducible() -> None:
    frame = sample_frame()
    first = split_by_run(frame, 42)
    second = split_by_run(frame, 42)
    assert np.array_equal(first.train, second.train)
    assert set(first.train_groups).isdisjoint(first.validation_groups)
    assert set(first.train_groups).isdisjoint(first.test_groups)
    assert set(first.validation_groups).isdisjoint(first.test_groups)

    groups = group_keys(frame)
    assert set(groups[first.train]) == set(first.train_groups)


def test_preprocessing_fits_training_only_and_handles_unknown_sites() -> None:
    frame = sample_frame()
    split = split_by_run(frame, 7)
    preprocessor = FeaturePreprocessor.fit(frame.iloc[split.train])
    validation = frame.iloc[split.validation].copy()
    validation.loc[validation.index[0], "allocation_site"] = "never-seen-in-training"
    transformed = preprocessor.transform(validation)
    assert transformed.dtype == np.float32
    assert transformed.shape == (len(validation), len(preprocessor.feature_names))
    assert transformed[0, -1] == 1.0
    assert LEAKAGE_COLUMNS.isdisjoint(preprocessor.feature_names)
    assert MODEL_INPUT_COLUMNS[-1] == "allocation_site"


def test_dataset_validation_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.csv"
    sample_frame().to_csv(path, index=False)
    loaded = load_dataset(path)
    assert set(labels(loaded)) == {0, 1}

    loaded.drop(columns=["seed"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_selection_score_accounts_for_latency() -> None:
    fast = {"f1": 0.90, "mean_inference_nanos": 1_000.0}
    slow = {"f1": 0.90, "mean_inference_nanos": 1_000_000.0}
    assert selection_score(fast) > selection_score(slow)


def test_metrics_include_required_outputs() -> None:
    class FixedModel:
        def predict_proba(self, features: np.ndarray) -> np.ndarray:
            positive = np.where(features[:, 0] > 0, 0.9, 0.1)
            return np.column_stack((1.0 - positive, positive))

    features = np.asarray([[-1.0], [1.0], [-2.0], [2.0]], dtype=np.float32)
    metrics = evaluate_model(FixedModel(), features, np.asarray([0, 1, 0, 1]))
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]

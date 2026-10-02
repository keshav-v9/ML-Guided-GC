from __future__ import annotations

import io
import math
import time
import warnings
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def candidate_models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            max_iter=1_000, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
    }
    try:
        # XGBoost needs a platform OpenMP runtime. Keep it as an optional
        # candidate so the portable sklearn pipeline remains usable without it.
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=160,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=1,
        )
    except Exception as error:  # native loader errors vary by xgboost release
        warnings.warn(f"XGBoost candidate unavailable: {error}", RuntimeWarning)
    return models


def evaluate_model(model: Any, features: np.ndarray, targets: np.ndarray) -> dict[str, Any]:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    metrics: dict[str, Any] = {
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = (
        float(roc_auc_score(targets, probabilities)) if len(np.unique(targets)) == 2 else None
    )
    return metrics


def inference_latency(model: Any, features: np.ndarray, repetitions: int = 50) -> dict[str, float]:
    sample = features[: min(256, len(features))]
    if len(sample) == 0:
        raise ValueError("latency sample must not be empty")
    model.predict_proba(sample)
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        model.predict_proba(sample)
        timings.append((time.perf_counter_ns() - start) / len(sample))
    return {
        "mean_inference_nanos": float(np.mean(timings)),
        "p95_inference_nanos": float(np.percentile(timings, 95)),
    }


def serialized_size(model: Any) -> int:
    stream = io.BytesIO()
    joblib.dump(model, stream)
    return len(stream.getvalue())


def selection_score(metrics: dict[str, Any]) -> float:
    # Penalize each order-of-magnitude increase in per-row latency by one F1 point.
    latency_micros = metrics["mean_inference_nanos"] / 1_000.0
    return float(metrics["f1"] - 0.01 * math.log1p(latency_micros))

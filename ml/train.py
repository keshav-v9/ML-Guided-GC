from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from ml.data import dataset_sha256, labels, load_dataset, split_by_run
from ml.features import FeaturePreprocessor, MODEL_INPUT_COLUMNS
from ml.modeling import (
    candidate_models,
    evaluate_model,
    inference_latency,
    selection_score,
    serialized_size,
)


def train(args: argparse.Namespace) -> dict[str, Any]:
    data_path = Path(args.data).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(data_path)
    split = split_by_run(frame, args.seed)
    targets = labels(frame)

    preprocessor = FeaturePreprocessor.fit(frame.iloc[split.train])
    x_train = preprocessor.transform(frame.iloc[split.train])
    x_validation = preprocessor.transform(frame.iloc[split.validation])
    x_test = preprocessor.transform(frame.iloc[split.test])
    y_train = targets[split.train]
    y_validation = targets[split.validation]
    y_test = targets[split.test]
    if len(np.unique(y_train)) != 2:
        raise ValueError("training split must contain both lifetime classes")

    data_hash = dataset_sha256(data_path)
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)
    candidates: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Any] = {}

    for name, model in candidate_models(args.seed).items():
        model.fit(x_train, y_train)
        validation = evaluate_model(model, x_validation, y_validation)
        validation.update(inference_latency(model, x_validation, args.latency_repetitions))
        validation["model_size_bytes"] = serialized_size(model)
        validation["selection_score"] = selection_score(validation)
        candidates[name] = validation
        fitted_models[name] = model

        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model_type": name,
                "seed": args.seed,
                "dataset_sha256": data_hash,
                "training_rows": len(split.train),
                "validation_rows": len(split.validation),
                "feature_count": len(preprocessor.feature_names),
            })
            mlflow.log_dict({"features": preprocessor.feature_names}, "feature_list.json")
            mlflow.log_dict({"groups": split.train_groups}, "train_groups.json")
            _log_metrics("validation", validation)
            mlflow.sklearn.log_model(model, artifact_path="model")

    selected_name = max(candidates, key=lambda name: candidates[name]["selection_score"])
    selected_model = fitted_models[selected_name]
    x_train_validation = np.vstack((x_train, x_validation))
    y_train_validation = np.concatenate((y_train, y_validation))
    selected_model.fit(x_train_validation, y_train_validation)
    test_metrics = evaluate_model(selected_model, x_test, y_test)
    test_metrics.update(inference_latency(selected_model, x_test, args.latency_repetitions))
    test_metrics["model_size_bytes"] = serialized_size(selected_model)

    model_path = output / "selected_model.joblib"
    preprocessor_path = output / "preprocessor.json"
    joblib.dump(selected_model, model_path)
    preprocessor.save(preprocessor_path)
    split_metadata = {
        "seed": args.seed,
        "train_groups": split.train_groups,
        "validation_groups": split.validation_groups,
        "test_groups": split.test_groups,
        "train_rows": len(split.train),
        "validation_rows": len(split.validation),
        "test_rows": len(split.test),
    }
    summary = {
        "dataset": str(data_path),
        "dataset_sha256": data_hash,
        "input_columns": MODEL_INPUT_COLUMNS,
        "expanded_feature_order": preprocessor.feature_names,
        "selected_model": selected_name,
        "selection_rule": "validation_f1 - 0.01 * log1p(mean_inference_microseconds)",
        "candidates": candidates,
        "test_metrics": test_metrics,
        "split": split_metadata,
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with mlflow.start_run(run_name="selected-model"):
        mlflow.log_params({
            "model_type": selected_name,
            "seed": args.seed,
            "dataset_sha256": data_hash,
            "selection_rule": summary["selection_rule"],
        })
        _log_metrics("test", test_metrics)
        mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessing")
        mlflow.log_artifact(str(output / "training_summary.json"), artifact_path="evaluation")
        mlflow.sklearn.log_model(selected_model, artifact_path="model")
    return summary


def _log_metrics(prefix: str, metrics: dict[str, Any]) -> None:
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and value is not None:
            mlflow.log_metric(f"{prefix}_{name}", float(value))
    mlflow.log_dict(metrics, f"{prefix}_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and select MiniGC lifetime models")
    parser.add_argument("--data", required=True, help="Telemetry CSV containing multiple runs")
    parser.add_argument("--output", default="ml/artifacts/latest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-repetitions", type=int, default=50)
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="minigc-lifetime")
    return parser.parse_args()


if __name__ == "__main__":
    result = train(parse_args())
    print(json.dumps({"selected_model": result["selected_model"],
                      "test_metrics": result["test_metrics"]}, indent=2))

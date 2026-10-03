from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from ml.data import load_dataset
from ml.features import FeaturePreprocessor, MODEL_INPUT_COLUMNS


def export(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = joblib.load(model_dir / "selected_model.joblib")
    preprocessor = FeaturePreprocessor.load(model_dir / "preprocessor.json")
    vector_size = len(preprocessor.feature_names)

    if model.__class__.__module__.startswith("xgboost"):
        from onnxmltools import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType as XgbFloatTensorType

        converted = convert_xgboost(
            model,
            initial_types=[("features", XgbFloatTensorType([None, vector_size]))],
            target_opset=args.opset,
        )
    else:
        converted = convert_sklearn(
            model,
            "ML-Guided-GC lifetime classifier",
            initial_types=[("features", FloatTensorType([None, vector_size]))],
            target_opset=args.opset,
            options={id(model): {"zipmap": False}},
        )
    output.write_bytes(converted.SerializeToString())

    frame = load_dataset(Path(args.golden_data)).head(args.golden_rows)
    vectors = preprocessor.transform(frame)
    sklearn_probabilities = model.predict_proba(vectors)[:, 1]
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_outputs = session.run(None, {session.get_inputs()[0].name: vectors})
    probability_index, probability_output, onnx_probabilities = _find_probabilities(
        session, onnx_outputs
    )
    max_delta = float(np.max(np.abs(sklearn_probabilities - onnx_probabilities)))
    if max_delta > args.tolerance:
        raise AssertionError(
            f"ONNX predictions differ from Python by {max_delta}, tolerance {args.tolerance}"
        )

    properties = output.with_suffix(".properties")
    preprocessor.write_java_properties(
        properties,
        input_name=session.get_inputs()[0].name,
        probability_output=probability_output,
        probability_output_index=probability_index,
        threshold=args.threshold,
        tolerance=args.tolerance,
    )
    golden = output.with_name(output.stem + "_golden.csv")
    _write_golden(golden, frame, onnx_probabilities)
    metadata = {
        "model": str(output),
        "properties": str(properties),
        "golden_data": str(golden),
        "input_name": session.get_inputs()[0].name,
        "probability_output": probability_output,
        "probability_index": probability_index,
        "threshold": args.threshold,
        "tolerance": args.tolerance,
        "max_python_onnx_delta": max_delta,
        "feature_order": preprocessor.feature_names,
    }
    metadata_path = output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name="onnx-export"):
        mlflow.log_params({
            "model_type": model.__class__.__name__,
            "opset": args.opset,
            "feature_count": vector_size,
            "prediction_tolerance": args.tolerance,
        })
        mlflow.log_metric("max_python_onnx_delta", max_delta)
        mlflow.log_artifact(str(output), artifact_path="onnx")
        mlflow.log_artifact(str(properties), artifact_path="onnx")
        mlflow.log_artifact(str(golden), artifact_path="validation")
        mlflow.log_artifact(str(metadata_path), artifact_path="validation")
    return metadata


def _find_probabilities(
    session: ort.InferenceSession, outputs: list[Any]
) -> tuple[int, str, np.ndarray]:
    for index, value in enumerate(outputs):
        array = np.asarray(value)
        if array.ndim == 2 and array.shape[1] >= 2 and np.issubdtype(array.dtype, np.floating):
            return index, session.get_outputs()[index].name, array[:, 1].astype(np.float64)
    raise ValueError("exported model did not produce a dense two-class probability tensor")


def _write_golden(path: Path, frame: Any, probabilities: np.ndarray) -> None:
    columns = [*MODEL_INPUT_COLUMNS, "expected_probability"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for (_, row), probability in zip(frame.iterrows(), probabilities):
            output = {column: row[column] for column in MODEL_INPUT_COLUMNS}
            output["expected_probability"] = repr(float(probability))
            writer.writerow(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the selected ML-Guided-GC model to ONNX"
    )
    parser.add_argument("--model-dir", default="ml/artifacts/latest")
    parser.add_argument("--golden-data", required=True)
    parser.add_argument("--output", default="ml/models/lifetime.onnx")
    parser.add_argument("--golden-rows", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--mlflow-tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiment", default="minigc-lifetime")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(export(parse_args()), indent=2))

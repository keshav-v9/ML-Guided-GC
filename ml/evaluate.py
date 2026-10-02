from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from ml.data import labels, load_dataset
from ml.features import FeaturePreprocessor
from ml.modeling import evaluate_model, inference_latency


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MiniGC model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-dir", default="ml/artifacts/latest")
    parser.add_argument("--latency-repetitions", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    frame = load_dataset(Path(args.data))
    model = joblib.load(model_dir / "selected_model.joblib")
    preprocessor = FeaturePreprocessor.load(model_dir / "preprocessor.json")
    features = preprocessor.transform(frame)
    metrics = evaluate_model(model, features, labels(frame))
    metrics.update(inference_latency(model, features, args.latency_repetitions))
    rendered = json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

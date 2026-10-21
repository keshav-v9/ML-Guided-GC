"""Compare logistic regression, random forest, XGBoost and the PyTorch MLP on the held-out runs."""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ml.data import labels, load_dataset, split_by_run
from ml.features import FeaturePreprocessor
from ml.modeling import candidate_models, inference_latency, serialized_size
from ml.numpy_report import confusion, expected_calibration_error, calibration_bins, precision_recall_f1, roc_auc, roc_curve
from ml.torch_model import fit, predict_proba

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {
    "PyTorch MLP": "#2a78d6",
    "Logistic regression": "#eb6834",
    "Random forest": "#1baf7a",
    "XGBoost": "#e87ba4",
}
DISPLAY = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost"}


def torch_latency(model: torch.nn.Module, features: np.ndarray, repetitions: int = 50) -> float:
    sample = torch.from_numpy(features[: min(256, len(features))])
    with torch.no_grad():
        model(sample)
        timings = []
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            model(sample)
            timings.append((time.perf_counter_ns() - start) / len(sample))
    return float(np.mean(timings))


def torch_size(model: torch.nn.Module) -> int:
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return len(stream.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="docs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_dataset(Path(args.data))
    split = split_by_run(frame, args.seed)
    targets = labels(frame)
    pre = FeaturePreprocessor.fit(frame.iloc[split.train])
    x_train, x_val, x_test = (pre.transform(frame.iloc[i]) for i in (split.train, split.validation, split.test))
    y_train, y_val, y_test = targets[split.train], targets[split.validation], targets[split.test]

    scores: dict[str, np.ndarray] = {}
    cost: dict[str, tuple[float, int]] = {}

    # sklearn/XGBoost candidates follow ml.train: refit on train+validation, score on test.
    x_fit, y_fit = np.vstack((x_train, x_val)), np.concatenate((y_train, y_val))
    for key, model in candidate_models(args.seed).items():
        model.fit(x_fit, y_fit)
        name = DISPLAY[key]
        scores[name] = model.predict_proba(x_test)[:, 1]
        cost[name] = (inference_latency(model, x_test)["mean_inference_nanos"], serialized_size(model))

    # The MLP early-stops on validation loss, so it trains on train only.
    mlp, _ = fit(x_train, y_train, x_val, y_val, epochs=60, batch_size=128, learning_rate=3e-3, seed=args.seed)
    scores["PyTorch MLP"] = predict_proba(mlp, x_test)
    cost["PyTorch MLP"] = (torch_latency(mlp, x_test), torch_size(mlp))

    results = {}
    for name in COLORS:
        if name not in scores:
            continue
        p = scores[name]
        precision, recall, f1 = precision_recall_f1(confusion(y_test, p))
        results[name] = {
            "f1": f1, "precision": precision, "recall": recall, "roc_auc": roc_auc(y_test, p),
            "ece": expected_calibration_error(calibration_bins(y_test, p)),
            "mean_inference_nanos": cost[name][0], "model_size_bytes": cost[name][1],
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_comparison.json").write_text(
        json.dumps({"test_rows": int(len(y_test)), "seed": args.seed, "models": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    draw(results, scores, y_test, out / "model_comparison.png")
    print(f"{'model':<20}{'F1':>7}{'AUC':>8}{'ECE':>8}{'ns/row':>10}{'bytes':>12}")
    for name, r in results.items():
        print(f"{name:<20}{r['f1']:>7.4f}{r['roc_auc']:>8.4f}{r['ece']:>8.4f}"
              f"{r['mean_inference_nanos']:>10.0f}{r['model_size_bytes']:>12,}")


def draw(results: dict, scores: dict, y_test: np.ndarray, path: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax = axes[0]
    for name in results:
        fpr, tpr = roc_curve(y_test, scores[name])
        ax.plot(fpr, tpr, color=COLORS[name], linewidth=2, label=f"{name} ({results[name]['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    ax.set(title="ROC curve (AUC)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    names = list(results)
    positions = np.arange(len(names))
    f1s = [results[n]["f1"] for n in names]
    ax.bar(positions, f1s, color=[COLORS[n] for n in names], width=0.6)
    for x, value in zip(positions, f1s):
        ax.text(x, value + 0.01, f"{value:.3f}", ha="center", color=INK)
    ax.set(title="Test F1", ylim=(0, 1), xticks=positions)
    ax.set_xticklabels([n.replace(" ", "\n") for n in names])

    ax = axes[2]
    for name, r in results.items():
        ax.scatter(r["mean_inference_nanos"], r["f1"], s=40 + 60 * np.log10(r["model_size_bytes"]) ** 1.5 / 4,
                   color=COLORS[name], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"{name}\n{r['model_size_bytes'] / 1024:,.1f} KB", (r["mean_inference_nanos"], r["f1"]),
                    textcoords="offset points", xytext=(10, 8), color=INK, fontsize=9)
    ax.set_xscale("log")
    ax.margins(x=0.35, y=0.25)
    ax.set(title="Accuracy vs latency (bubble area ~ model size)",
           xlabel="Mean inference time per row (ns, log scale)", ylabel="Test F1")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)


if __name__ == "__main__":
    main()

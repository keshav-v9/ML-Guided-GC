"""Pure-NumPy evaluation metrics and a text report for lifetime classifiers."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def confusion(targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    predicted = probabilities >= threshold
    actual = targets == 1
    return np.array([
        [np.sum(~actual & ~predicted), np.sum(~actual & predicted)],
        [np.sum(actual & ~predicted), np.sum(actual & predicted)],
    ])


def precision_recall_f1(matrix: np.ndarray) -> tuple[float, float, float]:
    tn, fp, fn, tp = matrix.ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(precision), float(recall), float(f1)


def roc_curve(targets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-probabilities, kind="stable")
    sorted_targets = targets[order]
    sorted_scores = probabilities[order]
    tps = np.cumsum(sorted_targets == 1)
    fps = np.cumsum(sorted_targets == 0)
    # Keep only the last point of each tied-score run.
    keep = np.r_[np.diff(sorted_scores) != 0, True]
    tpr = np.r_[0.0, tps[keep] / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps[keep] / max(fps[-1], 1)]
    return fpr, tpr


def roc_auc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr = roc_curve(targets, probabilities)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def calibration_bins(targets: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            rows.append({
                "lower": float(edges[b]), "upper": float(edges[b + 1]), "count": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "observed_rate": float(targets[mask].mean()),
            })
    return rows


def expected_calibration_error(rows: Sequence[dict[str, float]]) -> float:
    total = sum(r["count"] for r in rows)
    return float(sum(r["count"] * abs(r["mean_predicted"] - r["observed_rate"]) for r in rows) / total)


def best_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    best = (0.5, -1.0)
    for threshold in np.unique(np.round(probabilities, 3)):
        f1 = precision_recall_f1(confusion(targets, probabilities, threshold))[2]
        if f1 > best[1]:
            best = (float(threshold), f1)
    return best


def build_report(
    targets: np.ndarray,
    probabilities: np.ndarray,
    history: dict[str, list[float]],
    feature_names: Sequence[str],
    model: Any,
) -> dict[str, Any]:
    matrix = confusion(targets, probabilities)
    precision, recall, f1 = precision_recall_f1(matrix)
    bins = calibration_bins(targets, probabilities)
    threshold, threshold_f1 = best_f1_threshold(targets, probabilities)
    first_layer = model.net[0].weight.detach().numpy()
    importance = np.abs(first_layer).mean(axis=0)
    ranked = np.argsort(-importance)[:5]
    return {
        "test_rows": int(len(targets)),
        "long_lived_fraction": float(targets.mean()),
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc(targets, probabilities),
        "confusion_matrix": matrix.tolist(),
        "expected_calibration_error": expected_calibration_error(bins),
        "best_threshold": threshold, "best_threshold_f1": threshold_f1,
        "calibration": bins,
        "epochs_run": len(history["train_loss"]),
        "best_validation_loss": float(np.min(history["validation_loss"])),
        "final_train_loss": float(history["train_loss"][-1]),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "top_features_by_first_layer_weight": [
            {"feature": feature_names[i], "mean_abs_weight": float(importance[i])} for i in ranked
        ],
    }


def render_report(report: dict[str, Any]) -> str:
    (tn, fp), (fn, tp) = report["confusion_matrix"]
    lines = [
        "PyTorch LifetimeNet - NumPy test report",
        f"  rows={report['test_rows']}  long-lived={report['long_lived_fraction']:.1%}  params={report['parameter_count']}",
        f"  precision={report['precision']:.4f}  recall={report['recall']:.4f}  "
        f"f1={report['f1']:.4f}  roc_auc={report['roc_auc']:.4f}",
        f"  best-F1 threshold={report['best_threshold']:.3f} (f1={report['best_threshold_f1']:.4f})  "
        f"ECE={report['expected_calibration_error']:.4f}",
        "",
        "  Confusion matrix        pred short  pred long",
        f"    actual short          {tn:>10}  {fp:>9}",
        f"    actual long           {fn:>10}  {tp:>9}",
        "",
        "  Calibration (predicted vs observed long-lived rate)",
    ]
    for row in report["calibration"]:
        bar = "#" * int(round(row["observed_rate"] * 20))
        lines.append(
            f"    [{row['lower']:.1f},{row['upper']:.1f}) n={row['count']:<5} "
            f"pred={row['mean_predicted']:.2f} obs={row['observed_rate']:.2f} {bar}"
        )
    lines += ["", "  Top features by mean |first-layer weight|"]
    lines += [f"    {f['feature']:<40} {f['mean_abs_weight']:.3f}" for f in report["top_features_by_first_layer_weight"]]
    return "\n".join(lines)

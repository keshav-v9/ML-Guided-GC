"""Render the PyTorch-vs-logistic-regression results (metrics computed in NumPy) to a PNG."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.data import labels, load_dataset, split_by_run
from ml.features import FeaturePreprocessor
from ml.numpy_report import calibration_bins, confusion, precision_recall_f1, roc_auc, roc_curve
from ml.torch_model import fit, predict_proba

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MLP_COLOR, LR_COLOR = "#2a78d6", "#eb6834"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="docs/ml_results.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = load_dataset(Path(args.data))
    split = split_by_run(frame, args.seed)
    targets = labels(frame)
    pre = FeaturePreprocessor.fit(frame.iloc[split.train])
    x_train, x_val, x_test = (pre.transform(frame.iloc[i]) for i in (split.train, split.validation, split.test))
    y_train, y_val, y_test = targets[split.train], targets[split.validation], targets[split.test]

    mlp, history = fit(x_train, y_train, x_val, y_val, epochs=60, batch_size=128,
                       learning_rate=3e-3, seed=args.seed)
    p_mlp = predict_proba(mlp, x_test)
    lr = LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=args.seed)
    lr.fit(np.vstack((x_train, x_val)), np.concatenate((y_train, y_val)))
    p_lr = lr.predict_proba(x_test)[:, 1]

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.2), facecolor=SURFACE)
    for ax in axes.flat:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # ROC
    ax = axes[0, 0]
    for name, p, color in (("PyTorch MLP", p_mlp, MLP_COLOR), ("Logistic regression", p_lr, LR_COLOR)):
        fpr, tpr = roc_curve(y_test, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {roc_auc(y_test, p):.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    ax.set(title="ROC curve (held-out runs)", xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(frameon=False, loc="lower right")

    # Calibration
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    for name, p, color in (("PyTorch MLP", p_mlp, MLP_COLOR), ("Logistic regression", p_lr, LR_COLOR)):
        rows = calibration_bins(y_test, p)
        ax.plot([r["mean_predicted"] for r in rows], [r["observed_rate"] for r in rows],
                color=color, linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE, label=name)
    ax.set(title="Calibration", xlabel="Mean predicted P(long-lived)", ylabel="Observed long-lived rate")
    ax.legend(frameon=False, loc="upper left")

    # Confusion matrix
    ax = axes[1, 0]
    matrix = confusion(y_test, p_mlp)
    ax.grid(False)
    ax.imshow(matrix / matrix.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (row, col), count in np.ndenumerate(matrix):
        share = count / matrix[row].sum()
        ax.text(col, row, f"{count}\n{share:.1%}", ha="center", va="center",
                color="white" if share > 0.5 else INK, fontsize=11)
    precision, recall, f1 = precision_recall_f1(matrix)
    ax.set(title=f"MLP confusion matrix  (P {precision:.3f} · R {recall:.3f} · F1 {f1:.3f})",
           xlabel="Predicted", ylabel="Actual", xticks=[0, 1], yticks=[0, 1],
           xticklabels=["short", "long"], yticklabels=["short", "long"])

    # Loss curves
    ax = axes[1, 1]
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color=MLP_COLOR, linewidth=2, label="Train")
    ax.plot(epochs, history["validation_loss"], color=LR_COLOR, linewidth=2, label="Validation")
    best = int(np.argmin(history["validation_loss"]))
    ax.axvline(best + 1, color=MUTED, linewidth=1, linestyle=":")
    ax.set(title=f"MLP training loss (best epoch {best + 1})", xlabel="Epoch", ylabel="Weighted BCE loss")
    ax.legend(frameon=False, loc="center right")

    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor=SURFACE)
    print(f"wrote {output}  (LR AUC {roc_auc(y_test, p_lr):.4f}, MLP AUC {roc_auc(y_test, p_mlp):.4f})")


if __name__ == "__main__":
    main()

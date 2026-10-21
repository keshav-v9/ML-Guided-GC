from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ml.data import labels, load_dataset, split_by_run
from ml.features import FeaturePreprocessor


class LifetimeNet(nn.Module):
    """Small MLP that outputs a single long-lived logit per object."""

    def __init__(self, input_size: int, hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[LifetimeNet, dict[str, list[float]]]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LifetimeNet(x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    positives = max(int(y_train.sum()), 1)
    pos_weight = torch.tensor((len(y_train) - positives) / positives, dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train.astype(np.float32))
    xv = torch.from_numpy(x_validation)
    yv = torch.from_numpy(y_validation.astype(np.float32))

    history: dict[str, list[float]] = {"train_loss": [], "validation_loss": []}
    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        order = rng.permutation(len(xt))
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = torch.from_numpy(order[start:start + batch_size])
            optimizer.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(xv), yv).item()
        history["train_loss"].append(total / len(xt))
        history["validation_loss"].append(validation_loss)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def predict_proba(model: LifetimeNet, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(features))).numpy().astype(np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch lifetime classifier")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="ml/artifacts/torch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    args = parser.parse_args()

    from ml.numpy_report import build_report, render_report

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(Path(args.data))
    split = split_by_run(frame, args.seed)
    targets = labels(frame)

    # Same leakage-safe protocol as ml.train: fit preprocessing on train only.
    preprocessor = FeaturePreprocessor.fit(frame.iloc[split.train])
    x_train, x_val, x_test = (
        preprocessor.transform(frame.iloc[idx]) for idx in (split.train, split.validation, split.test)
    )
    y_train, y_val, y_test = targets[split.train], targets[split.validation], targets[split.test]

    model, history = fit(
        x_train, y_train, x_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed,
    )
    probabilities = predict_proba(model, x_test)

    torch.save(model.state_dict(), output / "lifetime_net.pt")
    preprocessor.save(output / "preprocessor.json")
    report: dict[str, Any] = build_report(y_test, probabilities, history, preprocessor.feature_names, model)
    (output / "torch_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    np.savez(output / "test_predictions.npz", probabilities=probabilities, targets=y_test)
    print(render_report(report))


if __name__ == "__main__":
    main()

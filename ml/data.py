from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ml.features import MODEL_INPUT_COLUMNS


LABELS = {"SHORT_LIVED": 0, "LONG_LIVED": 1}
REQUIRED_COLUMNS = ["workload_type", "seed", "label", *MODEL_INPUT_COLUMNS]


@dataclass(frozen=True)
class DatasetSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_groups: list[str]
    validation_groups: list[str]
    test_groups: list[str]


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")
    unknown_labels = sorted(set(frame["label"].dropna()) - set(LABELS))
    if unknown_labels:
        raise ValueError(f"unknown labels: {unknown_labels}")
    if frame.empty:
        raise ValueError("dataset must contain at least one row")
    return frame.reset_index(drop=True)


def labels(frame: pd.DataFrame) -> np.ndarray:
    return frame["label"].map(LABELS).to_numpy(dtype=np.int64)


def group_keys(frame: pd.DataFrame) -> np.ndarray:
    return (frame["workload_type"].astype(str) + "::" + frame["seed"].astype(str)).to_numpy()


def split_by_run(frame: pd.DataFrame, seed: int) -> DatasetSplits:
    groups = group_keys(frame)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("at least three workload_type/seed runs are required for leakage-safe splits")

    train_validation, test = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed).split(frame, groups=groups)
    )
    remaining_groups = groups[train_validation]
    train_relative, validation_relative = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1).split(
            frame.iloc[train_validation], groups=remaining_groups
        )
    )
    train = train_validation[train_relative]
    validation = train_validation[validation_relative]
    split = DatasetSplits(
        train=train,
        validation=validation,
        test=test,
        train_groups=sorted(set(groups[train])),
        validation_groups=sorted(set(groups[validation])),
        test_groups=sorted(set(groups[test])),
    )
    _validate_disjoint(split)
    return split


def dataset_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_disjoint(split: DatasetSplits) -> None:
    train = set(split.train_groups)
    validation = set(split.validation_groups)
    test = set(split.test_groups)
    if train & validation or train & test or validation & test:
        raise AssertionError("workload runs leaked across dataset splits")

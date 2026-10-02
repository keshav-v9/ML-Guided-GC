from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "size_bytes",
    "incoming_reference_count",
    "outgoing_reference_count",
    "heap_utilization_at_allocation",
    "allocation_rate",
    "gc_cycles_survived",
    "age_at_prediction",
]
CATEGORICAL_COLUMN = "allocation_site"
MODEL_INPUT_COLUMNS = [*NUMERIC_COLUMNS, CATEGORICAL_COLUMN]
LEAKAGE_COLUMNS = {"death_tick", "lifetime_ticks", "label", "censored", "labeling_rule"}


@dataclass(frozen=True)
class FeaturePreprocessor:
    numeric_columns: list[str]
    medians: list[float]
    means: list[float]
    scales: list[float]
    allocation_sites: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "FeaturePreprocessor":
        _require_columns(frame, MODEL_INPUT_COLUMNS)
        numeric = frame[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        medians = numeric.median(axis=0).fillna(0.0)
        filled = numeric.fillna(medians)
        means = filled.mean(axis=0)
        scales = filled.std(axis=0, ddof=0)
        # Numerically constant columns can produce tiny non-zero deviations;
        # treating those as real scales creates enormous or infinite vectors.
        scales = scales.mask(~np.isfinite(scales) | (scales.abs() < 1e-12), 1.0)
        sites = sorted(frame[CATEGORICAL_COLUMN].fillna("__MISSING__").astype(str).unique())
        return cls(
            numeric_columns=list(NUMERIC_COLUMNS),
            medians=[float(medians[column]) for column in NUMERIC_COLUMNS],
            means=[float(means[column]) for column in NUMERIC_COLUMNS],
            scales=[float(scales[column]) for column in NUMERIC_COLUMNS],
            allocation_sites=sites,
        )

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        _require_columns(frame, MODEL_INPUT_COLUMNS)
        numeric = frame[self.numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
            dtype=np.float64
        )
        medians = np.asarray(self.medians, dtype=np.float64)
        missing_rows, missing_columns = np.where(np.isnan(numeric))
        numeric[missing_rows, missing_columns] = medians[missing_columns]
        numeric = (numeric - np.asarray(self.means)) / np.asarray(self.scales)

        site_to_index = {site: index for index, site in enumerate(self.allocation_sites)}
        one_hot = np.zeros((len(frame), len(self.allocation_sites) + 1), dtype=np.float64)
        for row, site in enumerate(frame[CATEGORICAL_COLUMN].fillna("__MISSING__").astype(str)):
            one_hot[row, site_to_index.get(site, len(self.allocation_sites))] = 1.0
        return np.hstack((numeric, one_hot)).astype(np.float32)

    @property
    def feature_names(self) -> list[str]:
        return [
            *self.numeric_columns,
            *(f"allocation_site={site}" for site in self.allocation_sites),
            "allocation_site=__UNKNOWN__",
        ]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeaturePreprocessor":
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    def write_java_properties(
        self,
        path: Path,
        *,
        input_name: str,
        probability_output: str,
        probability_output_index: int,
        threshold: float,
        tolerance: float,
    ) -> None:
        properties = {
            "input.name": input_name,
            "probability.output": probability_output,
            "probability.output.index": str(probability_output_index),
            "probability.class.index": "1",
            "long_lived.threshold": repr(threshold),
            "prediction.tolerance": repr(tolerance),
            "numeric.columns": ",".join(self.numeric_columns),
            "numeric.medians": _join_numbers(self.medians),
            "numeric.means": _join_numbers(self.means),
            "numeric.scales": _join_numbers(self.scales),
            "allocation.site.count": str(len(self.allocation_sites)),
            "vector.size": str(len(self.feature_names)),
        }
        for index, site in enumerate(self.allocation_sites):
            properties[f"allocation.site.{index}"] = site
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(f"{_escape_property(key)}={_escape_property(value)}\n" for key, value in properties.items()),
            encoding="utf-8",
        )


def _join_numbers(values: Sequence[float]) -> str:
    return ",".join(repr(float(value)) for value in values)


def _escape_property(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace("=", "\\=")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

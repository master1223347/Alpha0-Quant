"""Feature normalization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RESERVED_COLUMNS = {
    "timestamp",
    "ticker",
    "label",
    "close",
    "next_close",
    "next_log_return",
    "vol_target",
    "vol_target_clipped",
    "vol_threshold",
    "z_return",
    "threshold_up",
    "threshold_down",
    "threshold_no_move",
    "threshold_label",
    "vol_direction_up",
    "vol_direction_down",
    "vol_direction_neutral",
    "vol_direction_label",
    "cross_sectional_rank",
    "event_threshold",
    "event_label",
    "event_direction_label",
    "event_signed_label",
    "event_magnitude",
}


@dataclass(slots=True)
class FeatureNormalizer:
    feature_columns: list[str]
    means: dict[str, float]
    stds: dict[str, float]


def infer_feature_columns(rows: list[dict[str, Any]], *, exclude: set[str] | None = None) -> list[str]:
    """Infer numeric feature columns from row dictionaries."""
    if not rows:
        return []
    excluded = RESERVED_COLUMNS.copy()
    if exclude is not None:
        excluded.update(exclude)

    columns: list[str] = []
    for key, value in rows[0].items():
        if key in excluded:
            continue
        if isinstance(value, (int, float)):
            columns.append(key)
    return sorted(columns)


def fit_feature_normalizer(
    rows: list[dict[str, Any]],
    *,
    feature_columns: list[str] | None = None,
    min_std: float = 1e-8,
) -> FeatureNormalizer:
    """Fit standard-score normalizer from training rows."""
    if not rows:
        raise ValueError("Cannot fit normalizer on empty rows")

    columns = feature_columns if feature_columns is not None else infer_feature_columns(rows)
    if not columns:
        raise ValueError("No numeric feature columns found for normalization")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}

    for column in columns:
        values = [float(row[column]) for row in rows]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = variance**0.5
        means[column] = mean_value
        stds[column] = max(std_value, min_std)

    return FeatureNormalizer(feature_columns=columns, means=means, stds=stds)


def transform_feature_rows(rows: list[dict[str, Any]], normalizer: FeatureNormalizer) -> list[dict[str, Any]]:
    """Apply fitted normalization to feature columns."""
    transformed: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        for column in normalizer.feature_columns:
            new_row[column] = (float(new_row[column]) - normalizer.means[column]) / normalizer.stds[column]
        transformed.append(new_row)
    return transformed

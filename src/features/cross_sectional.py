"""Cross-sectional feature generation utilities."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence


RESERVED_COLUMNS = {"timestamp", "ticker", "close", "next_close"}


def cross_sectional_feature_columns(feature_columns: Sequence[str]) -> list[str]:
    """Return derived feature names for same-timestamp cross-sectional stats."""
    columns: list[str] = []
    for column in feature_columns:
        columns.append(f"cs_{column}_rank")
        columns.append(f"cs_{column}_zscore")
    return columns


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return mean_value, math.sqrt(variance)


def _timestamp_sort_key(timestamp: Any) -> tuple[int, Any]:
    return (0, timestamp)


def apply_cross_sectional_features(
    rows: list[dict[str, Any]],
    *,
    feature_columns: Sequence[str],
) -> None:
    """Attach leakage-safe same-timestamp cross-sectional features in place."""
    if not rows:
        return
    if not feature_columns:
        raise ValueError("feature_columns must not be empty when building cross-sectional features")

    grouped_rows: dict[Any, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        timestamp = row.get("timestamp")
        if timestamp is None:
            raise ValueError("Cross-sectional feature generation requires timestamp values")
        grouped_rows[timestamp].append((index, row))

    for timestamp in sorted(grouped_rows.keys(), key=_timestamp_sort_key):
        indexed_rows = grouped_rows[timestamp]
        sorted_rows = sorted(
            indexed_rows,
            key=lambda item: (
                str(item[1].get("ticker", "")),
                item[0],
            ),
        )

        for column in feature_columns:
            values: list[float] = []
            for _, row in sorted_rows:
                if column not in row:
                    raise ValueError(f"Missing feature column {column!r} in cross-sectional batch")
                value = row[column]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Feature column {column!r} must be numeric for cross-sectional features")
                values.append(float(value))

            mean_value, std_value = _mean_and_std(values)
            if std_value <= 0 or all(value == values[0] for value in values):
                for _, row in sorted_rows:
                    row[f"cs_{column}_rank"] = 0.5
                    row[f"cs_{column}_zscore"] = 0.0
                continue

            ranked_rows = sorted(
                ((float(row[column]), str(row.get("ticker", "")), original_index, row) for original_index, row in sorted_rows),
                key=lambda item: (item[0], item[1], item[2]),
            )
            denominator = float(max(len(ranked_rows) - 1, 1))

            for position, (value, _, _, row) in enumerate(ranked_rows):
                row[f"cs_{column}_rank"] = position / denominator if len(ranked_rows) > 1 else 0.5
                row[f"cs_{column}_zscore"] = (value - mean_value) / std_value if std_value > 0 else 0.0

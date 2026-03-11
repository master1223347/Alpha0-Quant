"""Validation helpers for OHLCV and feature rows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from src.data.cleaner import validate_ohlcv_row
from src.data.loader import OhlcvRow


@dataclass(slots=True)
class ValidationSummary:
    stage: str
    row_count: int
    invalid_rows: int
    duplicate_timestamps: int
    out_of_order_pairs: int
    missing_field_counts: dict[str, int]

    @property
    def valid(self) -> bool:
        return self.invalid_rows == 0 and self.duplicate_timestamps == 0 and self.out_of_order_pairs == 0


def _count_order_and_duplicates(rows: list[dict[str, Any]]) -> tuple[int, int]:
    duplicate_timestamps = 0
    out_of_order_pairs = 0
    seen: set[Any] = set()
    previous_timestamp = None

    for row in rows:
        timestamp = row.get("timestamp")
        if timestamp in seen:
            duplicate_timestamps += 1
        seen.add(timestamp)

        if previous_timestamp is not None and timestamp is not None and timestamp < previous_timestamp:
            out_of_order_pairs += 1
        previous_timestamp = timestamp

    return duplicate_timestamps, out_of_order_pairs


def validate_ohlcv_rows(rows: list[OhlcvRow], *, stage: str = "ohlcv", raise_on_error: bool = False) -> ValidationSummary:
    """Validate canonical OHLCV rows and return a summary."""
    missing_field_counts: dict[str, int] = {}
    invalid_rows = 0

    for row in rows:
        reason = validate_ohlcv_row(row)
        if reason is not None:
            invalid_rows += 1
            missing_field_counts[reason] = missing_field_counts.get(reason, 0) + 1

    duplicate_timestamps, out_of_order_pairs = _count_order_and_duplicates(rows)
    summary = ValidationSummary(
        stage=stage,
        row_count=len(rows),
        invalid_rows=invalid_rows,
        duplicate_timestamps=duplicate_timestamps,
        out_of_order_pairs=out_of_order_pairs,
        missing_field_counts=missing_field_counts,
    )

    if raise_on_error and not summary.valid:
        raise ValueError(f"{stage} validation failed: {summary}")
    return summary


def validate_feature_rows(
    rows: list[dict[str, Any]],
    *,
    feature_columns: list[str],
    stage: str = "features",
    raise_on_error: bool = False,
) -> ValidationSummary:
    """Validate derived feature rows for shape, NaNs, and ordering."""
    missing_field_counts: dict[str, int] = {}
    invalid_rows = 0

    for row in rows:
        if "timestamp" not in row or row["timestamp"] is None:
            invalid_rows += 1
            missing_field_counts["missing_timestamp"] = missing_field_counts.get("missing_timestamp", 0) + 1
            continue

        for column in feature_columns:
            if column not in row:
                invalid_rows += 1
                key = f"missing_{column}"
                missing_field_counts[key] = missing_field_counts.get(key, 0) + 1
                continue

            value = row[column]
            if not isinstance(value, (int, float)) or math.isnan(value) or math.isinf(value):
                invalid_rows += 1
                key = f"invalid_{column}"
                missing_field_counts[key] = missing_field_counts.get(key, 0) + 1

    duplicate_timestamps, out_of_order_pairs = _count_order_and_duplicates(rows)
    summary = ValidationSummary(
        stage=stage,
        row_count=len(rows),
        invalid_rows=invalid_rows,
        duplicate_timestamps=duplicate_timestamps,
        out_of_order_pairs=out_of_order_pairs,
        missing_field_counts=missing_field_counts,
    )

    if raise_on_error and not summary.valid:
        raise ValueError(f"{stage} validation failed: {summary}")
    return summary

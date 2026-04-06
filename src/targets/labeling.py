"""Target labeling helpers for feature sequences."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence


TARGET_COLUMNS = (
    "next_log_return",
    "vol_target",
    "vol_target_clipped",
    "vol_threshold",
    "z_return",
    "label",
    "threshold_up",
    "threshold_down",
    "threshold_no_move",
    "threshold_label",
    "vol_direction_up",
    "vol_direction_down",
    "vol_direction_neutral",
    "vol_direction_label",
    "cross_sectional_rank",
)


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return mean_value, math.sqrt(variance)


def _trailing_values(sequence: Sequence[dict[str, Any]], index: int, window: int, key: str) -> list[float]:
    if window <= 0:
        raise ValueError(f"{key} window must be > 0")
    start_index = max(0, index - window + 1)
    values: list[float] = []
    for position in range(start_index, index + 1):
        if key not in sequence[position]:
            raise ValueError(f"Missing {key!r} in feature row at index {position}")
        values.append(float(sequence[position][key]))
    return values


def label_sequence(
    sequence: list[dict[str, Any]],
    *,
    horizon: int = 1,
    threshold: float = 0.001,
    volatility_window: int = 20,
    zscore_window: int = 20,
    volatility_label_k: float = 0.25,
    regression_clip: float = 3.0,
    epsilon: float = 1e-8,
) -> list[dict[str, Any]]:
    """Attach forward-return targets to a single feature sequence."""
    if horizon <= 0:
        raise ValueError("horizon must be > 0")
    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    if volatility_window <= 0:
        raise ValueError("volatility_window must be > 0")
    if zscore_window <= 0:
        raise ValueError("zscore_window must be > 0")
    if volatility_label_k < 0:
        raise ValueError("volatility_label_k must be >= 0")
    if regression_clip <= 0:
        raise ValueError("regression_clip must be > 0")

    if len(sequence) <= horizon:
        return []

    labeled: list[dict[str, Any]] = []
    for index in range(len(sequence) - horizon):
        current = dict(sequence[index])
        next_row = sequence[index + horizon]

        if "close" not in current or "close" not in next_row:
            raise ValueError("Feature rows must contain close prices before labeling")

        current_close = float(current["close"])
        next_close = float(next_row["close"])
        if current_close <= 0 or next_close <= 0:
            raise ValueError("close prices must be positive before labeling")

        next_log_return = math.log(next_close / current_close)
        volatility_values = _trailing_values(sequence, index, volatility_window, "log_return")
        _, vol_std = _mean_and_std(volatility_values)
        vol_threshold = float(volatility_label_k) * max(vol_std, epsilon)

        z_values = _trailing_values(sequence, index, zscore_window, "log_return")
        z_mean, z_std = _mean_and_std(z_values)

        if next_log_return > threshold:
            threshold_label = 2
            threshold_up = 1
            threshold_down = 0
            threshold_no_move = 0
        elif next_log_return < -threshold:
            threshold_label = 0
            threshold_up = 0
            threshold_down = 1
            threshold_no_move = 0
        else:
            threshold_label = 1
            threshold_up = 0
            threshold_down = 0
            threshold_no_move = 1

        if next_log_return > vol_threshold:
            vol_direction_label = 2
            vol_direction_up = 1
            vol_direction_down = 0
            vol_direction_neutral = 0
        elif next_log_return < -vol_threshold:
            vol_direction_label = 0
            vol_direction_up = 0
            vol_direction_down = 1
            vol_direction_neutral = 0
        else:
            vol_direction_label = 1
            vol_direction_up = 0
            vol_direction_down = 0
            vol_direction_neutral = 1

        current["next_close"] = next_close
        current["next_log_return"] = next_log_return
        current["vol_target"] = next_log_return / max(vol_std, epsilon)
        current["vol_target_clipped"] = max(min(current["vol_target"], float(regression_clip)), -float(regression_clip))
        current["vol_threshold"] = vol_threshold
        current["z_return"] = (next_log_return - z_mean) / max(z_std, epsilon)
        current["label"] = 1 if next_log_return > 0 else 0
        current["threshold_up"] = threshold_up
        current["threshold_down"] = threshold_down
        current["threshold_no_move"] = threshold_no_move
        current["threshold_label"] = threshold_label
        current["vol_direction_up"] = vol_direction_up
        current["vol_direction_down"] = vol_direction_down
        current["vol_direction_neutral"] = vol_direction_neutral
        current["vol_direction_label"] = vol_direction_label
        current["cross_sectional_rank"] = 0.0
        labeled.append(current)

    return labeled


def label_ticker_sequences(
    ticker_sequences: dict[str, list[list[dict[str, Any]]]],
    *,
    horizon: int = 1,
    threshold: float = 0.001,
    volatility_window: int = 20,
    zscore_window: int = 20,
    volatility_label_k: float = 0.25,
    regression_clip: float = 3.0,
    epsilon: float = 1e-8,
) -> dict[str, list[list[dict[str, Any]]]]:
    """Label all sequences for each ticker without mutating the input mapping."""
    labeled: dict[str, list[list[dict[str, Any]]]] = {}
    for ticker, sequences in ticker_sequences.items():
        labeled_sequences: list[list[dict[str, Any]]] = []
        for sequence in sequences:
            labeled_sequence = label_sequence(
                sequence,
                horizon=horizon,
                threshold=threshold,
                volatility_window=volatility_window,
                zscore_window=zscore_window,
                volatility_label_k=volatility_label_k,
                regression_clip=regression_clip,
                epsilon=epsilon,
            )
            if labeled_sequence:
                labeled_sequences.append(labeled_sequence)
        if labeled_sequences:
            labeled[ticker] = labeled_sequences
    return labeled


def assign_cross_sectional_rank(
    ticker_sequences: dict[str, list[list[dict[str, Any]]]],
    *,
    target_column: str = "next_log_return",
) -> None:
    """Assign same-timestamp cross-sectional rank targets in place."""
    grouped_rows: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for sequences in ticker_sequences.values():
        for sequence in sequences:
            for row in sequence:
                grouped_rows[row["timestamp"]].append(row)

    for timestamp in sorted(grouped_rows.keys()):
        rows = grouped_rows[timestamp]
        if not rows:
            continue

        if len(rows) == 1:
            rows[0]["cross_sectional_rank"] = 0.5
            continue

        ordered_rows = sorted(
            rows,
            key=lambda row: (
                float(row[target_column]),
                str(row.get("ticker", "")),
                row.get("timestamp"),
            ),
        )
        denominator = float(len(ordered_rows) - 1)
        for index, row in enumerate(ordered_rows):
            row["cross_sectional_rank"] = index / denominator

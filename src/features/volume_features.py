"""Volume-based feature generation."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TypedDict

from src.data.loader import OhlcvRow


class VolumeFeatureRow(TypedDict):
    timestamp: datetime
    volume_change: float
    volume_zscore: float
    relative_volume_long: float


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def build_volume_features(rows: list[OhlcvRow], *, window: int = 10) -> list[VolumeFeatureRow]:
    """Generate rolling volume features for a single contiguous sequence."""
    if window <= 1:
        raise ValueError("window must be > 1")
    if len(rows) < window:
        return []

    features: list[VolumeFeatureRow] = []
    for index in range(window - 1, len(rows)):
        row = rows[index]
        previous_volume = rows[index - 1]["volume"] if index > 0 else row["volume"]
        volume_change = (row["volume"] / previous_volume) - 1.0 if previous_volume > 0 else 0.0

        window_volumes = [float(item["volume"]) for item in rows[index - window + 1 : index + 1]]
        volume_mean = sum(window_volumes) / len(window_volumes)
        volume_std = _std(window_volumes)
        volume_zscore = (row["volume"] - volume_mean) / volume_std if volume_std > 0 else 0.0

        long_window = min(window * 2, index + 1)
        long_volumes = [float(item["volume"]) for item in rows[index - long_window + 1 : index + 1]]
        long_mean = sum(long_volumes) / len(long_volumes)
        relative_volume_long = row["volume"] / long_mean if long_mean > 0 else 0.0

        features.append(
            {
                "timestamp": row["timestamp"],
                "volume_change": volume_change,
                "volume_zscore": volume_zscore,
                "relative_volume_long": relative_volume_long,
            }
        )
    return features

"""Time-derived feature generation."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TypedDict

from src.data.loader import OhlcvRow


class TimeFeatureRow(TypedDict):
    timestamp: datetime
    session_pos_sin: float
    session_pos_cos: float
    day_of_week_sin: float
    day_of_week_cos: float


SESSION_OPEN_MINUTE = 9 * 60 + 30
SESSION_LENGTH_MINUTES = 390


def _session_minute(timestamp: datetime) -> int:
    return (timestamp.hour * 60 + timestamp.minute) - SESSION_OPEN_MINUTE


def build_time_features(rows: list[OhlcvRow]) -> list[TimeFeatureRow]:
    """Generate cyclical time features for each OHLCV row."""
    features: list[TimeFeatureRow] = []
    for row in rows:
        timestamp = row["timestamp"]
        session_minute = max(0, min(_session_minute(timestamp), SESSION_LENGTH_MINUTES))
        session_ratio = session_minute / SESSION_LENGTH_MINUTES
        session_angle = 2.0 * math.pi * session_ratio

        weekday_ratio = timestamp.weekday() / 5.0
        weekday_angle = 2.0 * math.pi * weekday_ratio

        features.append(
            {
                "timestamp": timestamp,
                "session_pos_sin": math.sin(session_angle),
                "session_pos_cos": math.cos(session_angle),
                "day_of_week_sin": math.sin(weekday_angle),
                "day_of_week_cos": math.cos(weekday_angle),
            }
        )
    return features

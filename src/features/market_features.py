"""Price-action market feature generation."""

from __future__ import annotations

from datetime import datetime
from typing import TypedDict

from src.data.loader import OhlcvRow


class MarketFeatureRow(TypedDict):
    timestamp: datetime
    gap_return: float
    intrabar_return: float
    close_position: float
    range_expansion: float


def _close_position(row: OhlcvRow) -> float:
    spread = row["high"] - row["low"]
    if spread <= 0:
        return 0.5
    return (row["close"] - row["low"]) / spread


def build_market_features(rows: list[OhlcvRow]) -> list[MarketFeatureRow]:
    """Generate simple price-action features from OHLC bars."""
    if len(rows) < 2:
        return []

    features: list[MarketFeatureRow] = []
    for index in range(1, len(rows)):
        previous = rows[index - 1]
        current = rows[index]

        gap_return = (current["open"] / previous["close"]) - 1.0
        intrabar_return = (current["close"] / current["open"]) - 1.0
        close_position = _close_position(current)

        previous_range = max(previous["high"] - previous["low"], 1e-12)
        current_range = current["high"] - current["low"]
        range_expansion = (current_range / previous_range) - 1.0

        features.append(
            {
                "timestamp": current["timestamp"],
                "gap_return": gap_return,
                "intrabar_return": intrabar_return,
                "close_position": close_position,
                "range_expansion": range_expansion,
            }
        )
    return features

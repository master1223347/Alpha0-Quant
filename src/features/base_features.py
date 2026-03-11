"""Base feature generation from cleaned, aligned OHLCV rows."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

from src.data.loader import OhlcvRow

if TYPE_CHECKING:
    import pandas as pd


class BaseFeatureRow(TypedDict):
    timestamp: datetime
    log_return: float
    candle_range: float
    candle_body: float
    upper_wick: float
    lower_wick: float
    short_term_momentum: float
    rolling_volatility: float
    relative_volume: float


def _price_scale(row: OhlcvRow) -> float:
    # Cleaner enforces positive prices, but keep a defensive fallback.
    open_price = row["open"]
    return open_price if open_price > 0 else 1.0


def _log_return(current_close: float, previous_close: float) -> float:
    return math.log(current_close / previous_close)


def _rolling_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def build_base_features(
    rows: list[OhlcvRow],
    *,
    momentum_lookback: int = 3,
    volatility_window: int = 5,
    relative_volume_window: int = 5,
) -> list[BaseFeatureRow]:
    """Generate base features for a single contiguous OHLCV sequence."""
    if momentum_lookback <= 0:
        raise ValueError("momentum_lookback must be > 0")
    if volatility_window <= 1:
        raise ValueError("volatility_window must be > 1")
    if relative_volume_window <= 0:
        raise ValueError("relative_volume_window must be > 0")

    if len(rows) < 2:
        return []

    log_returns: list[float] = [0.0]
    for index in range(1, len(rows)):
        previous_close = rows[index - 1]["close"]
        current_close = rows[index]["close"]
        log_returns.append(_log_return(current_close, previous_close))

    min_index = max(
        1,
        momentum_lookback,
        volatility_window,
        relative_volume_window - 1,
    )

    features: list[BaseFeatureRow] = []
    for index in range(min_index, len(rows)):
        row = rows[index]
        scale = _price_scale(row)

        high_price = row["high"]
        low_price = row["low"]
        open_price = row["open"]
        close_price = row["close"]

        candle_range = (high_price - low_price) / scale
        candle_body = (close_price - open_price) / scale
        upper_wick = (high_price - max(open_price, close_price)) / scale
        lower_wick = (min(open_price, close_price) - low_price) / scale

        momentum_base = rows[index - momentum_lookback]["close"]
        short_term_momentum = (close_price / momentum_base) - 1.0

        vol_start = index - volatility_window + 1
        rolling_volatility = _rolling_std(log_returns[vol_start : index + 1])

        volume_start = index - relative_volume_window + 1
        rolling_volume = rows[volume_start : index + 1]
        average_volume = sum(item["volume"] for item in rolling_volume) / len(rolling_volume)
        relative_volume = row["volume"] / average_volume if average_volume > 0 else 0.0

        features.append(
            {
                "timestamp": row["timestamp"],
                "log_return": log_returns[index],
                "candle_range": candle_range,
                "candle_body": candle_body,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "short_term_momentum": short_term_momentum,
                "rolling_volatility": rolling_volatility,
                "relative_volume": relative_volume,
            }
        )

    return features


def build_base_features_for_sequences(
    sequences: list[list[OhlcvRow]],
    *,
    momentum_lookback: int = 3,
    volatility_window: int = 5,
    relative_volume_window: int = 5,
) -> list[list[BaseFeatureRow]]:
    """Generate base features for multiple contiguous OHLCV sequences."""
    feature_sequences: list[list[BaseFeatureRow]] = []
    for rows in sequences:
        feature_rows = build_base_features(
            rows,
            momentum_lookback=momentum_lookback,
            volatility_window=volatility_window,
            relative_volume_window=relative_volume_window,
        )
        if feature_rows:
            feature_sequences.append(feature_rows)
    return feature_sequences


def build_base_feature_frame(
    rows: list[OhlcvRow],
    *,
    momentum_lookback: int = 3,
    volatility_window: int = 5,
    relative_volume_window: int = 5,
) -> "pd.DataFrame":
    """Generate base features and return a pandas DataFrame if pandas is installed."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for build_base_feature_frame(); use build_base_features() otherwise."
        ) from exc

    feature_rows = build_base_features(
        rows,
        momentum_lookback=momentum_lookback,
        volatility_window=volatility_window,
        relative_volume_window=relative_volume_window,
    )
    columns = [
        "timestamp",
        "log_return",
        "candle_range",
        "candle_body",
        "upper_wick",
        "lower_wick",
        "short_term_momentum",
        "rolling_volatility",
        "relative_volume",
    ]
    return pd.DataFrame(feature_rows, columns=columns)

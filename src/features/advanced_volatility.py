"""Advanced realized-volatility feature estimators from OHLC bars."""

from __future__ import annotations

import math
from typing import TypedDict

from src.data.loader import OhlcvRow


class RealizedVolFeatureRow(TypedDict):
    timestamp: object
    rv_close_to_close: float
    rv_parkinson: float
    rv_garman_klass: float


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    return math.log(max(numerator, 1e-8) / max(denominator, 1e-8))


def _rolling_mean(values: list[float], start: int, end: int) -> float:
    window = values[start:end]
    if not window:
        return 0.0
    return sum(window) / len(window)


def build_realized_volatility_features(
    rows: list[OhlcvRow],
    *,
    window: int = 20,
) -> list[RealizedVolFeatureRow]:
    """Build range-aware realized-volatility estimates aligned to row timestamps."""
    if window <= 1:
        raise ValueError("window must be > 1")
    if len(rows) < 2:
        return []

    close_to_close_sq: list[float] = [0.0]
    parkinson_var: list[float] = [0.0]
    garman_klass_var: list[float] = [0.0]
    log2 = math.log(2.0)

    for index in range(1, len(rows)):
        prev_close = float(rows[index - 1]["close"])
        row = rows[index]
        open_price = float(row["open"])
        high_price = float(row["high"])
        low_price = float(row["low"])
        close_price = float(row["close"])

        ret = _safe_log_ratio(close_price, prev_close)
        close_to_close_sq.append(ret * ret)

        hl = _safe_log_ratio(high_price, low_price)
        p_var = (hl * hl) / (4.0 * log2)
        parkinson_var.append(max(0.0, p_var))

        co = _safe_log_ratio(close_price, open_price)
        gk_var = (0.5 * hl * hl) - ((2.0 * log2 - 1.0) * co * co)
        garman_klass_var.append(max(0.0, gk_var))

    start_index = max(1, window - 1)
    output: list[RealizedVolFeatureRow] = []
    for index in range(start_index, len(rows)):
        window_start = index - window + 1
        cc = math.sqrt(max(0.0, _rolling_mean(close_to_close_sq, window_start, index + 1)))
        pk = math.sqrt(max(0.0, _rolling_mean(parkinson_var, window_start, index + 1)))
        gk = math.sqrt(max(0.0, _rolling_mean(garman_klass_var, window_start, index + 1)))
        output.append(
            {
                "timestamp": rows[index]["timestamp"],
                "rv_close_to_close": float(cc),
                "rv_parkinson": float(pk),
                "rv_garman_klass": float(gk),
            }
        )
    return output

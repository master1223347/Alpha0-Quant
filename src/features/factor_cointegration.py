"""Factor-style and cointegration-style bar-only proxy features."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float], mean_value: float | None = None) -> float:
    if not values:
        return 0.0
    baseline = _mean(values) if mean_value is None else float(mean_value)
    variance = sum((value - baseline) ** 2 for value in values) / len(values)
    return math.sqrt(max(0.0, variance))


def _ols_line(x: list[float], y: list[float]) -> tuple[float, float]:
    if not x or not y or len(x) != len(y):
        return 0.0, 0.0
    x_mean = _mean(x)
    y_mean = _mean(y)
    var_x = sum((value - x_mean) ** 2 for value in x)
    if var_x <= 1e-12:
        return 0.0, y_mean
    cov_xy = sum((xv - x_mean) * (yv - y_mean) for xv, yv in zip(x, y))
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    return beta, alpha


def _ar1_half_life(values: list[float], *, clip: float) -> float:
    if len(values) < 4:
        return float(clip)
    lag = values[:-1]
    curr = values[1:]
    lag_mean = _mean(lag)
    curr_mean = _mean(curr)
    var_lag = sum((value - lag_mean) ** 2 for value in lag)
    if var_lag <= 1e-12:
        return float(clip)
    cov = sum((lv - lag_mean) * (cv - curr_mean) for lv, cv in zip(lag, curr))
    phi = cov / var_lag
    if phi <= 0.0 or phi >= 0.999:
        return float(clip)
    half_life = -math.log(2.0) / math.log(phi)
    if not math.isfinite(half_life):
        return float(clip)
    return float(min(max(1.0, half_life), float(clip)))


def apply_factor_cointegration_features(
    rows: list[dict[str, Any]],
    *,
    use_factor_features: bool = True,
    use_cointegration_features: bool = True,
    factor_window: int = 78,
    cointegration_window: int = 78,
    min_samples: int = 40,
    half_life_clip: float = 200.0,
) -> None:
    """Attach market-factor residual and cointegration proxy features in place."""
    if not rows:
        return
    if not use_factor_features and not use_cointegration_features:
        return
    if factor_window <= 1:
        raise ValueError("factor_window must be > 1")
    if cointegration_window <= 1:
        raise ValueError("cointegration_window must be > 1")

    by_timestamp: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        timestamp = row.get("timestamp")
        if timestamp is None:
            raise ValueError("factor/cointegration features require timestamp")
        by_timestamp[timestamp].append(row)

    ordered_timestamps = sorted(by_timestamp.keys())
    market_proxy_close: dict[Any, float] = {}
    for timestamp in ordered_timestamps:
        closes = [float(entry.get("close", 0.0)) for entry in by_timestamp[timestamp]]
        valid_closes = [value for value in closes if value > 0.0 and math.isfinite(value)]
        market_proxy_close[timestamp] = _mean(valid_closes) if valid_closes else 1.0

    market_log_return: dict[Any, float] = {}
    market_log_price: dict[Any, float] = {}
    prev_close: float | None = None
    running_log_price = 0.0
    for timestamp in ordered_timestamps:
        close_value = max(float(market_proxy_close[timestamp]), 1e-8)
        if prev_close is None:
            ret = 0.0
        else:
            ret = math.log(close_value / max(prev_close, 1e-8))
        running_log_price += ret
        market_log_return[timestamp] = float(ret)
        market_log_price[timestamp] = float(running_log_price)
        prev_close = close_value

    by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_ticker[str(row.get("ticker", ""))].append(row)

    for ticker_rows in by_ticker.values():
        ticker_rows.sort(key=lambda item: item["timestamp"])
        own_returns: list[float] = []
        mkt_returns: list[float] = []
        residual_history: list[float] = []

        for row in ticker_rows:
            timestamp = row["timestamp"]
            row_return = float(row.get("log_return", 0.0))
            mkt_return = float(market_log_return.get(timestamp, 0.0))
            if use_factor_features:
                row["market_log_return"] = mkt_return

            own_returns.append(row_return)
            mkt_returns.append(mkt_return)

            factor_start = max(0, len(own_returns) - factor_window)
            own_factor_window = own_returns[factor_start:]
            mkt_factor_window = mkt_returns[factor_start:]
            if use_factor_features:
                if len(own_factor_window) >= min_samples:
                    beta, alpha = _ols_line(mkt_factor_window, own_factor_window)
                else:
                    beta, alpha = 0.0, _mean(own_factor_window)
                predicted = alpha + beta * mkt_return
                residual = row_return - predicted
                row["rolling_beta_market"] = float(beta)
                row["residual_return_market"] = float(residual)

            own_log_price = math.log(max(float(row.get("close", 1e-8)), 1e-8))
            mkt_log_price = float(market_log_price.get(timestamp, 0.0))
            residual_history.append(own_log_price)
            ctg_start = max(0, len(residual_history) - cointegration_window)
            own_price_window = [math.log(max(float(entry.get("close", 1e-8)), 1e-8)) for entry in ticker_rows[ctg_start : len(residual_history)]]
            mkt_price_window = [float(market_log_price.get(entry["timestamp"], 0.0)) for entry in ticker_rows[ctg_start : len(residual_history)]]
            if len(own_price_window) >= min_samples:
                hedge_beta, hedge_alpha = _ols_line(mkt_price_window, own_price_window)
            else:
                hedge_beta, hedge_alpha = 1.0, 0.0

            coint_residual = own_log_price - (hedge_alpha + hedge_beta * mkt_log_price)
            recent_residuals: list[float] = []
            for entry in ticker_rows[ctg_start : len(residual_history)]:
                entry_own_log = math.log(max(float(entry.get("close", 1e-8)), 1e-8))
                entry_mkt_log = float(market_log_price.get(entry["timestamp"], 0.0))
                recent_residuals.append(entry_own_log - (hedge_alpha + hedge_beta * entry_mkt_log))
            residual_mean = _mean(recent_residuals)
            residual_std = _std(recent_residuals, residual_mean)
            residual_z = (coint_residual - residual_mean) / residual_std if residual_std > 1e-8 else 0.0
            half_life = _ar1_half_life(recent_residuals, clip=half_life_clip)

            if use_cointegration_features:
                row["cointegration_residual"] = float(coint_residual)
                row["cointegration_zscore"] = float(residual_z)
                row["cointegration_half_life"] = float(half_life)

"""Simple probability-threshold backtest."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class BacktestReport:
    pnl: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    trade_count: int
    long_count: int
    short_count: int
    flat_count: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return variance**0.5


def run_backtest(
    probabilities: list[float],
    close: list[float],
    next_close: list[float],
    *,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    periods_per_year: int = 252 * 78,
) -> BacktestReport:
    """Run a threshold-based long/short strategy."""
    if not (len(probabilities) == len(close) == len(next_close)):
        raise ValueError("probabilities, close, and next_close must have equal lengths")
    if not probabilities:
        raise ValueError("Cannot backtest empty arrays")

    positions: list[int] = []
    period_returns: list[float] = []

    long_count = short_count = flat_count = 0
    for probability, current_close, future_close in zip(probabilities, close, next_close):
        if probability > long_threshold:
            position = 1
            long_count += 1
        elif probability < short_threshold:
            position = -1
            short_count += 1
        else:
            position = 0
            flat_count += 1

        raw_return = (future_close - current_close) / current_close if current_close > 0 else 0.0
        strategy_return = position * raw_return

        positions.append(position)
        period_returns.append(strategy_return)

    pnl = sum(period_returns)

    return_std = _std(period_returns)
    sharpe = 0.0
    if return_std > 0:
        sharpe = (sum(period_returns) / len(period_returns)) / return_std
        sharpe *= math.sqrt(periods_per_year)

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for strategy_return in period_returns:
        equity *= 1.0 + strategy_return
        if equity > peak:
            peak = equity
        drawdown = (equity - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    active_indices = [index for index, position in enumerate(positions) if position != 0]
    winning_trades = 0
    for index in active_indices:
        if period_returns[index] > 0:
            winning_trades += 1
    hit_rate = (winning_trades / len(active_indices)) if active_indices else 0.0

    return BacktestReport(
        pnl=pnl,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        trade_count=len(active_indices),
        long_count=long_count,
        short_count=short_count,
        flat_count=flat_count,
    )

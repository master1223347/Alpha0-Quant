"""Simple probability-threshold backtest."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class BacktestReport:
    pnl: float
    gross_pnl: float
    transaction_cost_pnl: float
    slippage_pnl: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    trade_count: int
    long_count: int
    short_count: int
    flat_count: int
    nan_signal_count: int = 0
    cost_bps_per_trade: float = 0.0
    slippage_bps: float = 0.0

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
    cost_bps_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
) -> BacktestReport:
    """Run a threshold-based long/short strategy."""
    if not (len(probabilities) == len(close) == len(next_close)):
        raise ValueError("probabilities, close, and next_close must have equal lengths")
    if not probabilities:
        raise ValueError("Cannot backtest empty arrays")
    if short_threshold >= long_threshold:
        raise ValueError("short_threshold must be < long_threshold")

    positions: list[int] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []

    long_count = short_count = flat_count = 0
    nan_signal_count = 0
    trade_count = 0
    previous_position = 0
    cost_fraction = (float(cost_bps_per_trade) + float(slippage_bps)) / 10000.0
    transaction_cost_pnl = 0.0
    slippage_pnl = 0.0

    for probability, current_close, future_close in zip(probabilities, close, next_close):
        if not math.isfinite(float(probability)):
            position = 0
            nan_signal_count += 1
            flat_count += 1
        elif probability >= long_threshold:
            position = 1
            long_count += 1
        elif probability <= short_threshold:
            position = -1
            short_count += 1
        else:
            position = 0
            flat_count += 1

        turnover = abs(position - previous_position)
        if turnover > 0:
            trade_count += 1

        raw_return = (future_close - current_close) / current_close if current_close > 0 and math.isfinite(current_close) else 0.0
        gross_strategy_return = position * raw_return
        trade_cost = cost_fraction * turnover
        net_strategy_return = gross_strategy_return - trade_cost

        positions.append(position)
        gross_returns.append(gross_strategy_return)
        net_returns.append(net_strategy_return)
        transaction_cost_pnl -= (float(cost_bps_per_trade) / 10000.0) * turnover
        slippage_pnl -= (float(slippage_bps) / 10000.0) * turnover
        previous_position = position

    gross_pnl = sum(gross_returns)
    pnl = sum(net_returns)

    return_std = _std(net_returns)
    sharpe = 0.0
    if return_std > 0:
        sharpe = (sum(net_returns) / len(net_returns)) / return_std
        sharpe *= math.sqrt(periods_per_year)

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for strategy_return in net_returns:
        equity *= 1.0 + strategy_return
        if equity > peak:
            peak = equity
        drawdown = (equity - peak) / peak
        if drawdown < max_drawdown:
            max_drawdown = drawdown

    active_indices = [index for index, position in enumerate(positions) if position != 0]
    winning_trades = 0
    for index in active_indices:
        if net_returns[index] > 0:
            winning_trades += 1
    hit_rate = (winning_trades / len(active_indices)) if active_indices else 0.0

    return BacktestReport(
        pnl=pnl,
        gross_pnl=gross_pnl,
        transaction_cost_pnl=transaction_cost_pnl,
        slippage_pnl=slippage_pnl,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        hit_rate=hit_rate,
        trade_count=trade_count,
        long_count=long_count,
        short_count=short_count,
        flat_count=flat_count,
        nan_signal_count=nan_signal_count,
        cost_bps_per_trade=float(cost_bps_per_trade),
        slippage_bps=float(slippage_bps),
    )

"""Simple probability-threshold backtest."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.regime import adapt_position_to_regime, detect_market_regimes


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
    flip_positions: bool = False
    nan_signal_count: int = 0
    cost_bps_per_trade: float = 0.0
    slippage_bps: float = 0.0
    confidence_threshold: float | None = None
    top_percentile: float | None = None
    selection_mode: str | None = None
    long_short_percentile: float | None = None
    selected_bars: int = 0
    equity_curve: list[float] | None = None
    regime_adaptation_enabled: bool = False
    regime_counts: dict[str, int] | None = None
    regime_transition_matrix: list[list[float]] | None = None
    regime_state_mapping: dict[str, str] | None = None
    regime_policies: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return variance**0.5


def _normalize_timestamp(value: Any) -> float | str | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if hasattr(value, "to_pydatetime"):
        try:
            value = value.to_pydatetime()
        except Exception:
            pass
    if hasattr(value, "timestamp"):
        try:
            return float(value.timestamp())
        except Exception:
            pass
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return value
    return str(value)


def _build_market_proxy_series(
    *,
    close: list[float],
    next_close: list[float],
    timestamps: list[Any] | None,
) -> tuple[list[float | str], list[float], list[float | str]] | None:
    if timestamps is None or len(timestamps) != len(close) or len(next_close) != len(close):
        return None

    grouped_returns: dict[float | str, list[float]] = {}
    normalized_per_sample: list[float | str] = []
    for index, raw_timestamp in enumerate(timestamps):
        timestamp = _normalize_timestamp(raw_timestamp)
        if timestamp is None:
            normalized_per_sample.append("__missing__")
            continue
        normalized_per_sample.append(timestamp)
        current_close = float(close[index])
        future_close = float(next_close[index])
        if current_close > 0 and math.isfinite(current_close) and math.isfinite(future_close):
            market_return = (future_close - current_close) / current_close
        else:
            market_return = 0.0
        grouped_returns.setdefault(timestamp, []).append(float(market_return))

    if not grouped_returns:
        return None

    ordered_timestamps = sorted(grouped_returns.keys())
    proxy_close: list[float] = []
    level = 1.0
    for timestamp in ordered_timestamps:
        proxy_close.append(level)
        returns = grouped_returns[timestamp]
        average_return = sum(returns) / len(returns) if returns else 0.0
        if not math.isfinite(average_return):
            average_return = 0.0
        average_return = max(-0.95, float(average_return))
        level = max(1e-8, level * (1.0 + average_return))

    return ordered_timestamps, proxy_close, normalized_per_sample


def run_backtest(
    probabilities: list[float] | None,
    close: list[float],
    next_close: list[float],
    *,
    up_probabilities: list[float] | None = None,
    down_probabilities: list[float] | None = None,
    timestamps: list[Any] | None = None,
    tickers: list[str] | None = None,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    confidence_threshold: float | None = None,
    top_percentile: float | None = None,
    selection_mode: str | None = None,
    long_short_percentile: float | None = None,
    periods_per_year: int = 252 * 78,
    execution_lag_bars: int = 1,
    flip_positions: bool = False,
    cost_bps_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    signal_scores: list[float] | None = None,
    signal_source: str = "classification_prob",
    mu_values: list[float] | None = None,
    sigma_values: list[float] | None = None,
    require_directional_agreement: bool = False,
    confidence_mu_agreement_weight: float = 0.50,
    signal_mu_sigma_floor: float = 0.05,
    enable_regime_adaptation: bool = False,
    regime_states: int = 3,
    regime_feature_window: int = 32,
    regime_random_state: int = 7,
    trending_policy: str = "follow",
    mean_reverting_policy: str = "flip",
    volatile_policy: str = "flat",
    volatile_confidence_threshold: float = 0.70,
    regime_fit_close: list[float] | None = None,
    regime_fit_next_close: list[float] | None = None,
    regime_fit_timestamps: list[Any] | None = None,
    regime_fit_tickers: list[str] | None = None,
) -> BacktestReport:
    """Run a threshold-based long/short strategy."""
    if probabilities is None and (up_probabilities is None or down_probabilities is None):
        raise ValueError("Provide either probabilities or both up_probabilities/down_probabilities")

    if up_probabilities is None or down_probabilities is None:
        resolved_probabilities = probabilities or []
        up_probabilities = [float(value) for value in resolved_probabilities]
        down_probabilities = [1.0 - float(value) for value in resolved_probabilities]
    else:
        up_probabilities = [float(value) for value in up_probabilities]
        down_probabilities = [float(value) for value in down_probabilities]

    if not (len(up_probabilities) == len(down_probabilities) == len(close) == len(next_close)):
        raise ValueError("probability and price arrays must have equal lengths")
    if not up_probabilities:
        raise ValueError("Cannot backtest empty arrays")
    if confidence_threshold is None and short_threshold >= long_threshold:
        raise ValueError("short_threshold must be < long_threshold when confidence_threshold is not set")
    lag_bars = int(execution_lag_bars)
    if lag_bars < 0:
        raise ValueError("execution_lag_bars must be >= 0")
    if top_percentile is not None and not (0.0 < float(top_percentile) <= 1.0):
        raise ValueError("top_percentile must be in (0, 1]")
    if long_short_percentile is not None and not (0.0 < float(long_short_percentile) <= 0.5):
        raise ValueError("long_short_percentile must be in (0, 0.5]")

    confidence_scores = [max(float(up), float(down)) for up, down in zip(up_probabilities, down_probabilities)]
    blended_scores = signal_scores
    source = str(signal_source).strip().lower()
    if blended_scores is None and source != "classification_prob":
        mu = [float(value) for value in (mu_values or [0.0 for _ in up_probabilities])]
        if len(mu) != len(up_probabilities):
            mu = [0.0 for _ in up_probabilities]
        if sigma_values is None:
            sigma = [1.0 for _ in up_probabilities]
        else:
            sigma = [max(float(signal_mu_sigma_floor), float(value)) for value in sigma_values]
            if len(sigma) != len(up_probabilities):
                sigma = [1.0 for _ in up_probabilities]
        directional_confidence = [
            float(up - down) for up, down in zip(up_probabilities, down_probabilities)
        ]
        if source == "mu":
            blended_scores = list(mu)
        elif source == "mu_over_sigma":
            blended_scores = [float(m) / float(s) for m, s in zip(mu, sigma)]
        elif source == "confidence_plus_mu":
            weight = float(confidence_mu_agreement_weight)
            blended_scores = [
                (weight * directional_confidence[index]) + ((1.0 - weight) * mu[index])
                for index in range(len(up_probabilities))
            ]
        else:
            blended_scores = directional_confidence
    if blended_scores is not None and len(blended_scores) != len(up_probabilities):
        blended_scores = None

    if blended_scores is None:
        scores_for_selection = [float(up - down) for up, down in zip(up_probabilities, down_probabilities)]
    else:
        scores_for_selection = [float(value) for value in blended_scores]

    score_nan_count = 0
    sanitized_scores: list[float] = []
    for value in scores_for_selection:
        if math.isfinite(value):
            sanitized_scores.append(float(value))
        else:
            score_nan_count += 1
            sanitized_scores.append(0.0)
    scores_for_selection = sanitized_scores

    selection = str(selection_mode or "global_abs").strip().lower()
    long_indices: set[int] = set()
    short_indices: set[int] = set()
    selected_indices = set(range(len(up_probabilities)))
    if selection == "separate_long_short" and long_short_percentile is not None:
        keep_count = max(1, int(len(up_probabilities) * float(long_short_percentile)))
        ranked_long = sorted(range(len(up_probabilities)), key=lambda index: scores_for_selection[index], reverse=True)
        ranked_short = sorted(range(len(up_probabilities)), key=lambda index: scores_for_selection[index])
        long_indices = set(ranked_long[:keep_count])
        short_indices = set(ranked_short[:keep_count])
        selected_indices = long_indices | short_indices
    elif top_percentile is not None:
        keep_count = max(1, int(len(up_probabilities) * float(top_percentile)))
        if selection == "global_abs":
            ranked_indices = sorted(
                range(len(up_probabilities)),
                key=lambda index: abs(scores_for_selection[index]),
                reverse=True,
            )
        else:
            ranked_indices = sorted(range(len(up_probabilities)), key=lambda index: confidence_scores[index], reverse=True)
        selected_indices = set(ranked_indices[:keep_count])

    regime_sequence = ["trending" for _ in up_probabilities]
    regime_counts: dict[str, int] | None = None
    regime_transition_matrix: list[list[float]] | None = None
    regime_state_mapping: dict[str, str] | None = None
    regime_enabled = bool(enable_regime_adaptation)
    if regime_enabled and not regime_fit_close:
        # Avoid fitting regimes on the same stream being traded (forward-look leakage).
        regime_enabled = False
    regime_policies = {
        "trending": str(trending_policy),
        "mean_reverting": str(mean_reverting_policy),
        "volatile": str(volatile_policy),
        "volatile_confidence_threshold": f"{float(volatile_confidence_threshold):.4f}",
    }
    if regime_enabled:
        try:
            sample_proxy = _build_market_proxy_series(
                close=[float(value) for value in close],
                next_close=[float(value) for value in next_close],
                timestamps=timestamps,
            )
            fit_proxy = _build_market_proxy_series(
                close=[float(value) for value in (regime_fit_close or [])],
                next_close=[float(value) for value in (regime_fit_next_close or [])],
                timestamps=regime_fit_timestamps,
            )
            if sample_proxy is not None:
                ordered_timestamps, proxy_close, sample_timestamp_keys = sample_proxy
                fit_close_series = fit_proxy[1] if fit_proxy is not None else (regime_fit_close or None)
                regime_result = detect_market_regimes(
                    close=proxy_close,
                    fit_close=[float(value) for value in fit_close_series] if fit_close_series else None,
                    n_states=int(regime_states),
                    feature_window=int(regime_feature_window),
                    random_state=int(regime_random_state),
                )
                if len(regime_result.regime_sequence) == len(ordered_timestamps):
                    regime_by_timestamp = {
                        ordered_timestamps[index]: regime_name
                        for index, regime_name in enumerate(regime_result.regime_sequence)
                    }
                    regime_sequence = [
                        regime_by_timestamp.get(timestamp_key, "trending")
                        for timestamp_key in sample_timestamp_keys
                    ]
            else:
                regime_result = detect_market_regimes(
                    close=[float(value) for value in close],
                    fit_close=[float(value) for value in regime_fit_close] if regime_fit_close else None,
                    n_states=int(regime_states),
                    feature_window=int(regime_feature_window),
                    random_state=int(regime_random_state),
                )
                if len(regime_result.regime_sequence) == len(up_probabilities):
                    regime_sequence = list(regime_result.regime_sequence)
            regime_transition_matrix = regime_result.transition_matrix
            regime_state_mapping = {
                str(int(state)): str(label)
                for state, label in regime_result.state_to_regime.items()
            }
            regime_counts = {
                "trending": 0,
                "mean_reverting": 0,
                "volatile": 0,
            }
            for regime_name in regime_sequence:
                key = str(regime_name).strip().lower()
                if key not in regime_counts:
                    regime_counts[key] = 0
                regime_counts[key] += 1
        except Exception:
            regime_enabled = False
            regime_counts = None
            regime_transition_matrix = None
            regime_state_mapping = None

    signal_positions: list[int] = []
    positions: list[int] = []
    gross_returns: list[float] = []
    net_returns: list[float] = []

    long_count = short_count = flat_count = 0
    nan_signal_count = score_nan_count
    trade_count = 0
    previous_position = 0
    cost_fraction = (float(cost_bps_per_trade) + float(slippage_bps)) / 10000.0
    transaction_cost_pnl = 0.0
    slippage_pnl = 0.0

    for index, (up_probability, down_probability) in enumerate(zip(up_probabilities, down_probabilities)):
        raw_position = 0
        if index not in selected_indices:
            raw_position = 0
        elif not math.isfinite(float(up_probability)) or not math.isfinite(float(down_probability)):
            nan_signal_count += 1
        elif selection == "separate_long_short" and long_short_percentile is not None:
            if index in long_indices and index in short_indices:
                raw_position = 1 if scores_for_selection[index] >= 0 else -1
            elif index in long_indices:
                raw_position = 1
            elif index in short_indices:
                raw_position = -1
        else:
            score = float(scores_for_selection[index])
            if confidence_threshold is not None:
                if up_probability >= float(confidence_threshold) and up_probability > down_probability:
                    raw_position = 1
                elif down_probability >= float(confidence_threshold) and down_probability > up_probability:
                    raw_position = -1
            else:
                if score > 0:
                    raw_position = 1
                elif score < 0:
                    raw_position = -1
                elif up_probability >= long_threshold and up_probability > down_probability:
                    raw_position = 1
                elif down_probability >= short_threshold and down_probability > up_probability:
                    raw_position = -1
            if require_directional_agreement and blended_scores is not None:
                directional_sign = 1 if up_probability >= down_probability else -1
                if raw_position != 0 and (1 if raw_position > 0 else -1) != directional_sign:
                    raw_position = 0
        if regime_enabled:
            raw_position = adapt_position_to_regime(
                raw_position=raw_position,
                regime=regime_sequence[index],
                confidence=confidence_scores[index],
                trending_policy=trending_policy,
                mean_reverting_policy=mean_reverting_policy,
                volatile_policy=volatile_policy,
                volatile_confidence_threshold=float(volatile_confidence_threshold),
            )

        position = -raw_position if flip_positions else raw_position
        signal_positions.append(position)

    executed_positions = [0] * len(signal_positions)
    for index, position in enumerate(signal_positions):
        execution_index = index + lag_bars
        if execution_index < len(executed_positions):
            executed_positions[execution_index] = position

    for position, current_close, future_close in zip(executed_positions, close, next_close):
        if position > 0:
            long_count += 1
        elif position < 0:
            short_count += 1
        else:
            flat_count += 1

        turnover = abs(position - previous_position)
        if turnover > 0:
            trade_count += 1

        if (
            current_close > 0
            and math.isfinite(current_close)
            and math.isfinite(future_close)
        ):
            raw_return = (future_close - current_close) / current_close
        else:
            raw_return = 0.0
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
    equity_curve: list[float] = [equity]
    for strategy_return in net_returns:
        equity *= 1.0 + strategy_return
        equity_curve.append(equity)
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
        flip_positions=bool(flip_positions),
        nan_signal_count=nan_signal_count,
        cost_bps_per_trade=float(cost_bps_per_trade),
        slippage_bps=float(slippage_bps),
        confidence_threshold=float(confidence_threshold) if confidence_threshold is not None else None,
        top_percentile=float(top_percentile) if top_percentile is not None else None,
        selection_mode=selection,
        long_short_percentile=float(long_short_percentile) if long_short_percentile is not None else None,
        selected_bars=len(selected_indices),
        equity_curve=equity_curve,
        regime_adaptation_enabled=regime_enabled,
        regime_counts=regime_counts,
        regime_transition_matrix=regime_transition_matrix,
        regime_state_mapping=regime_state_mapping,
        regime_policies=regime_policies if regime_enabled else None,
    )

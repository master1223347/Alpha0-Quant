"""Market regime detection via clustering-assisted Gaussian HMM smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class RegimeDetectionResult:
    state_sequence: list[int]
    regime_sequence: list[str]
    transition_matrix: list[list[float]]
    state_to_regime: dict[int, str]
    state_stats: dict[int, dict[str, float]]


def _safe_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(max(variance, 0.0))


def _zscore_column(values: list[float]) -> list[float]:
    if not values:
        return []
    mean_value = sum(values) / len(values)
    std_value = _safe_std(values)
    if std_value <= 1e-12:
        return [0.0 for _ in values]
    return [(float(value) - mean_value) / std_value for value in values]


def _rolling_std(values: list[float], window: int) -> list[float]:
    window = max(2, int(window))
    if not values:
        return []
    output: list[float] = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        output.append(_safe_std(values[start : index + 1]))
    return output


def _lag_product_signal(values: list[float], window: int) -> list[float]:
    window = max(2, int(window))
    if not values:
        return []
    products = [0.0]
    for index in range(1, len(values)):
        products.append(float(values[index]) * float(values[index - 1]))
    output: list[float] = []
    for index in range(len(products)):
        start = max(0, index - window + 1)
        segment = products[start : index + 1]
        output.append(sum(segment) / len(segment))
    return output


def _state_transition_matrix(states: list[int], n_states: int, *, smoothing: float = 1.0) -> list[list[float]]:
    counts = [[float(smoothing) for _ in range(n_states)] for _ in range(n_states)]
    for prev_state, next_state in zip(states[:-1], states[1:]):
        counts[int(prev_state)][int(next_state)] += 1.0
    matrix: list[list[float]] = []
    for row in counts:
        row_total = sum(row)
        if row_total <= 0:
            matrix.append([1.0 / n_states for _ in range(n_states)])
        else:
            matrix.append([float(value) / row_total for value in row])
    return matrix


def _log_gaussian_diag(point: list[float], mean: list[float], var: list[float]) -> float:
    total = 0.0
    for value, mu, sigma2 in zip(point, mean, var):
        sigma2 = max(float(sigma2), 1e-8)
        diff = float(value) - float(mu)
        total += math.log(2.0 * math.pi * sigma2) + ((diff * diff) / sigma2)
    return -0.5 * total


def _viterbi_decode(
    *,
    features: list[list[float]],
    priors: list[float],
    transition: list[list[float]],
    means: list[list[float]],
    variances: list[list[float]],
) -> list[int]:
    if not features:
        return []
    n_states = len(priors)
    t_count = len(features)
    log_priors = [math.log(max(float(value), 1e-12)) for value in priors]
    log_transition = [
        [math.log(max(float(transition[s0][s1]), 1e-12)) for s1 in range(n_states)]
        for s0 in range(n_states)
    ]
    emission = [
        [
            _log_gaussian_diag(features[t], means[state], variances[state])
            for state in range(n_states)
        ]
        for t in range(t_count)
    ]

    dp = [[-float("inf") for _ in range(n_states)] for _ in range(t_count)]
    parent = [[0 for _ in range(n_states)] for _ in range(t_count)]

    for state in range(n_states):
        dp[0][state] = log_priors[state] + emission[0][state]

    for t in range(1, t_count):
        for state in range(n_states):
            best_score = -float("inf")
            best_prev = 0
            for prev_state in range(n_states):
                score = dp[t - 1][prev_state] + log_transition[prev_state][state]
                if score > best_score:
                    best_score = score
                    best_prev = prev_state
            dp[t][state] = best_score + emission[t][state]
            parent[t][state] = best_prev

    final_state = max(range(n_states), key=lambda state: dp[-1][state])
    states = [0 for _ in range(t_count)]
    states[-1] = final_state
    for t in range(t_count - 1, 0, -1):
        states[t - 1] = parent[t][states[t]]
    return states


def _state_autocorrelation(returns: list[float], states: list[int], state_id: int) -> float:
    x_values: list[float] = []
    y_values: list[float] = []
    for index in range(1, len(returns)):
        if states[index] != state_id:
            continue
        x_values.append(float(returns[index - 1]))
        y_values.append(float(returns[index]))
    if len(x_values) < 2:
        return 0.0
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / len(x_values)
    x_std = _safe_std(x_values)
    y_std = _safe_std(y_values)
    if x_std <= 1e-12 or y_std <= 1e-12:
        return 0.0
    return float(cov / (x_std * y_std))


def _build_state_stats(returns: list[float], states: list[int], n_states: int) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for state_id in range(n_states):
        state_returns = [float(returns[index]) for index, value in enumerate(states) if value == state_id]
        state_mean = sum(state_returns) / len(state_returns) if state_returns else 0.0
        state_vol = _safe_std(state_returns)
        autocorr = _state_autocorrelation(returns, states, state_id)
        trend_score = (abs(state_mean) / max(state_vol, 1e-6)) + max(0.0, autocorr)
        reversion_score = max(0.0, -autocorr) + (state_vol / max(abs(state_mean), 1e-6)) * 0.05
        stats[state_id] = {
            "count": float(len(state_returns)),
            "mean_return": float(state_mean),
            "volatility": float(state_vol),
            "autocorr_lag1": float(autocorr),
            "trend_score": float(trend_score),
            "reversion_score": float(reversion_score),
        }
    return stats


def _map_states_to_regimes(state_stats: dict[int, dict[str, float]]) -> dict[int, str]:
    if not state_stats:
        return {}
    state_ids = list(state_stats.keys())
    if len(state_ids) == 1:
        return {state_ids[0]: "trending"}

    by_vol = sorted(state_ids, key=lambda state: state_stats[state]["volatility"], reverse=True)
    volatile_state = by_vol[0]
    mapping: dict[int, str] = {volatile_state: "volatile"}

    remaining = [state for state in state_ids if state != volatile_state]
    if not remaining:
        return mapping
    if len(remaining) == 1:
        only_state = remaining[0]
        if state_stats[only_state]["autocorr_lag1"] < 0:
            mapping[only_state] = "mean_reverting"
        else:
            mapping[only_state] = "trending"
        return mapping

    trending_state = max(remaining, key=lambda state: state_stats[state]["trend_score"])
    mapping[trending_state] = "trending"

    leftover = [state for state in remaining if state != trending_state]
    for state in leftover:
        mapping[state] = "mean_reverting"
    return mapping


def detect_market_regimes(
    *,
    close: list[float],
    n_states: int = 3,
    feature_window: int = 32,
    random_state: int = 7,
) -> RegimeDetectionResult:
    """Detect per-bar market regimes using KMeans + Gaussian-HMM smoothing."""
    if not close:
        raise ValueError("close must be non-empty")

    sample_count = len(close)
    if sample_count == 1:
        return RegimeDetectionResult(
            state_sequence=[0],
            regime_sequence=["trending"],
            transition_matrix=[[1.0]],
            state_to_regime={0: "trending"},
            state_stats={0: {"count": 1.0, "mean_return": 0.0, "volatility": 0.0, "autocorr_lag1": 0.0, "trend_score": 0.0, "reversion_score": 0.0}},
        )

    returns = [0.0]
    for index in range(1, sample_count):
        prev_close = float(close[index - 1])
        curr_close = float(close[index])
        if prev_close > 0 and math.isfinite(prev_close) and math.isfinite(curr_close):
            bar_return = (curr_close - prev_close) / prev_close
            # Guard against cross-symbol boundary jumps in flattened panels.
            if abs(bar_return) > 0.20:
                bar_return = 0.0
            returns.append(bar_return)
        else:
            returns.append(0.0)

    rolling_vol = _rolling_std(returns, window=max(4, int(feature_window)))
    lag_signal = _lag_product_signal(returns, window=max(4, int(feature_window)))

    feature_columns = [
        _zscore_column(returns),
        _zscore_column(rolling_vol),
        _zscore_column(lag_signal),
    ]
    features = [
        [feature_columns[0][index], feature_columns[1][index], feature_columns[2][index]]
        for index in range(sample_count)
    ]

    resolved_states = max(1, min(int(n_states), sample_count))
    if resolved_states == 1:
        state_sequence = [0 for _ in range(sample_count)]
        state_stats = _build_state_stats(returns, state_sequence, 1)
        state_to_regime = {0: "trending"}
        return RegimeDetectionResult(
            state_sequence=state_sequence,
            regime_sequence=[state_to_regime[0] for _ in state_sequence],
            transition_matrix=[[1.0]],
            state_to_regime=state_to_regime,
            state_stats=state_stats,
        )

    try:
        import numpy as np
        from sklearn.cluster import KMeans
    except ModuleNotFoundError:
        # Conservative fallback: volatility tertiles as pseudo states.
        ordered = sorted(range(sample_count), key=lambda index: rolling_vol[index])
        labels = [0 for _ in range(sample_count)]
        for rank, index in enumerate(ordered):
            state_id = min(resolved_states - 1, (rank * resolved_states) // sample_count)
            labels[index] = state_id
        cluster_labels = labels
    else:
        matrix = np.asarray(features, dtype=float)
        kmeans = KMeans(n_clusters=resolved_states, random_state=int(random_state), n_init=10)
        cluster_labels = [int(value) for value in kmeans.fit_predict(matrix).tolist()]

    transition = _state_transition_matrix(cluster_labels, resolved_states, smoothing=1.0)
    state_counts = [0 for _ in range(resolved_states)]
    for value in cluster_labels:
        state_counts[int(value)] += 1
    total = float(sum(state_counts))
    priors = [max(1e-6, float(count) / max(total, 1.0)) for count in state_counts]
    prior_total = sum(priors)
    priors = [value / prior_total for value in priors]

    means: list[list[float]] = []
    variances: list[list[float]] = []
    dim = len(features[0])
    for state_id in range(resolved_states):
        state_points = [features[index] for index, label in enumerate(cluster_labels) if label == state_id]
        if not state_points:
            means.append([0.0 for _ in range(dim)])
            variances.append([1.0 for _ in range(dim)])
            continue
        mean_row = []
        var_row = []
        for column in range(dim):
            values = [row[column] for row in state_points]
            col_mean = sum(values) / len(values)
            col_var = sum((value - col_mean) ** 2 for value in values) / len(values)
            mean_row.append(float(col_mean))
            var_row.append(max(float(col_var), 1e-4))
        means.append(mean_row)
        variances.append(var_row)

    state_sequence = _viterbi_decode(
        features=features,
        priors=priors,
        transition=transition,
        means=means,
        variances=variances,
    )
    state_stats = _build_state_stats(returns, state_sequence, resolved_states)
    state_to_regime = _map_states_to_regimes(state_stats)
    regime_sequence = [state_to_regime.get(int(state), "trending") for state in state_sequence]
    return RegimeDetectionResult(
        state_sequence=state_sequence,
        regime_sequence=regime_sequence,
        transition_matrix=transition,
        state_to_regime=state_to_regime,
        state_stats=state_stats,
    )


def adapt_position_to_regime(
    *,
    raw_position: int,
    regime: str,
    confidence: float,
    trending_policy: str = "follow",
    mean_reverting_policy: str = "flip",
    volatile_policy: str = "flat",
    volatile_confidence_threshold: float = 0.70,
) -> int:
    """Apply policy by regime and return adjusted {-1,0,1} position."""
    regime_name = str(regime).strip().lower()
    position = int(raw_position)

    def _apply_policy(value: int, policy: str) -> int:
        mode = str(policy).strip().lower()
        if mode == "follow":
            return int(value)
        if mode == "flip":
            return -int(value)
        if mode == "flat":
            return 0
        if mode == "high_confidence":
            return int(value) if float(confidence) >= float(volatile_confidence_threshold) else 0
        return int(value)

    if regime_name == "trending":
        return _apply_policy(position, trending_policy)
    if regime_name == "mean_reverting":
        return _apply_policy(position, mean_reverting_policy)
    if regime_name == "volatile":
        return _apply_policy(position, volatile_policy)
    return position

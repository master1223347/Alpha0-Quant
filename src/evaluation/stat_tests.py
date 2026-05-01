"""Selection-bias and data-snooping statistical controls."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any


_NORMAL = NormalDist()
_EULER_GAMMA = 0.5772156649015329


@dataclass(slots=True)
class DeflatedSharpeResult:
    observed_sharpe: float
    benchmark_max_sharpe: float
    deflated_sharpe_z: float
    deflated_sharpe_pvalue: float
    trial_count: int
    sample_length: int


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float], *, ddof: int = 0) -> float:
    if not values:
        return 0.0
    if len(values) <= ddof:
        return 0.0
    mu = _mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - ddof)
    return math.sqrt(max(0.0, variance))


def _skew(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    mu = _mean(values)
    sigma = _std(values)
    if sigma <= 1e-12:
        return 0.0
    return sum(((value - mu) / sigma) ** 3 for value in values) / len(values)


def _kurtosis(values: list[float]) -> float:
    if len(values) < 4:
        return 3.0
    mu = _mean(values)
    sigma = _std(values)
    if sigma <= 1e-12:
        return 3.0
    return sum(((value - mu) / sigma) ** 4 for value in values) / len(values)


def _sharpe(values: list[float]) -> float:
    sigma = _std(values)
    if sigma <= 1e-12:
        return 0.0
    return _mean(values) / sigma


def benjamini_hochberg(
    p_values: list[float],
    *,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Benjamini-Hochberg FDR control with selected indices output."""
    clipped = [min(max(float(value), 0.0), 1.0) for value in p_values]
    m = len(clipped)
    if m == 0:
        return {"alpha": float(alpha), "rejections": [], "adjusted_threshold": 0.0}

    indexed = sorted(enumerate(clipped), key=lambda item: item[1])
    last_rank = 0
    threshold = 0.0
    for rank, (_, p_value) in enumerate(indexed, start=1):
        candidate = (rank / m) * float(alpha)
        if p_value <= candidate:
            last_rank = rank
            threshold = candidate

    rejected: list[int] = []
    if last_rank > 0:
        rejected = [index for index, _ in indexed[:last_rank]]
    return {
        "alpha": float(alpha),
        "rejections": sorted(rejected),
        "adjusted_threshold": float(threshold),
    }


def deflated_sharpe_ratio(
    *,
    strategy_returns: list[float],
    trial_sharpes: list[float],
) -> DeflatedSharpeResult:
    """Approximate deflated Sharpe ratio (Bailey-Lopez de Prado style)."""
    if not strategy_returns:
        raise ValueError("strategy_returns must not be empty")
    if not trial_sharpes:
        raise ValueError("trial_sharpes must not be empty")

    observed_sharpe = _sharpe(strategy_returns)
    trial_count = max(1, len(trial_sharpes))
    sample_length = max(2, len(strategy_returns))

    mu_trials = _mean(trial_sharpes)
    sigma_trials = max(1e-8, _std(trial_sharpes))
    q1 = _NORMAL.inv_cdf(1.0 - (1.0 / trial_count))
    q2 = _NORMAL.inv_cdf(1.0 - (1.0 / (trial_count * math.e)))
    benchmark_max = mu_trials + sigma_trials * ((1.0 - _EULER_GAMMA) * q1 + _EULER_GAMMA * q2)

    skew = _skew(strategy_returns)
    kurt = _kurtosis(strategy_returns)
    adjustment = max(1e-8, 1.0 - skew * observed_sharpe + ((kurt - 1.0) * (observed_sharpe**2) / 4.0))
    z_score = ((observed_sharpe - benchmark_max) * math.sqrt(sample_length - 1.0)) / math.sqrt(adjustment)
    p_value = 1.0 - _NORMAL.cdf(z_score)

    return DeflatedSharpeResult(
        observed_sharpe=float(observed_sharpe),
        benchmark_max_sharpe=float(benchmark_max),
        deflated_sharpe_z=float(z_score),
        deflated_sharpe_pvalue=float(p_value),
        trial_count=int(trial_count),
        sample_length=int(sample_length),
    )


def _bootstrap_indices(length: int, *, rng: random.Random) -> list[int]:
    return [rng.randrange(length) for _ in range(length)]


def white_reality_check(
    *,
    strategy_returns: list[list[float]],
    bootstrap: int = 500,
    seed: int = 7,
) -> dict[str, float]:
    """White's Reality Check p-value for best-mean-return among strategies."""
    if not strategy_returns:
        return {"p_value": float("nan"), "observed_best_mean": float("nan")}
    lengths = {len(values) for values in strategy_returns}
    if len(lengths) != 1:
        raise ValueError("all strategy return series must have equal length")
    length = lengths.pop()
    if length <= 1:
        return {"p_value": float("nan"), "observed_best_mean": float("nan")}

    centered = []
    means = []
    for values in strategy_returns:
        mu = _mean(values)
        means.append(mu)
        centered.append([value - mu for value in values])
    observed_best = max(means)

    rng = random.Random(seed)
    exceed = 0
    for _ in range(max(10, int(bootstrap))):
        idx = _bootstrap_indices(length, rng=rng)
        boot_best = max(_mean([series[i] for i in idx]) for series in centered)
        if boot_best >= observed_best:
            exceed += 1
    p_value = exceed / max(10, int(bootstrap))
    return {"p_value": float(p_value), "observed_best_mean": float(observed_best)}


def hansen_spa_test(
    *,
    strategy_returns: list[list[float]],
    benchmark_returns: list[float] | None = None,
    bootstrap: int = 500,
    seed: int = 11,
) -> dict[str, float]:
    """Hansen SPA-style bootstrap test against benchmark mean return."""
    if not strategy_returns:
        return {"p_value": float("nan"), "observed_statistic": float("nan")}
    lengths = {len(values) for values in strategy_returns}
    if len(lengths) != 1:
        raise ValueError("all strategy return series must have equal length")
    length = lengths.pop()
    if length <= 1:
        return {"p_value": float("nan"), "observed_statistic": float("nan")}

    benchmark = benchmark_returns if benchmark_returns is not None else [0.0 for _ in range(length)]
    if len(benchmark) != length:
        raise ValueError("benchmark_returns length must match strategy series length")

    differential = [[series[index] - benchmark[index] for index in range(length)] for series in strategy_returns]
    t_stats = []
    centered = []
    for values in differential:
        mu = _mean(values)
        sigma = _std(values)
        t_stats.append((mu / max(1e-8, sigma)) * math.sqrt(length))
        centered.append([value - mu for value in values])
    observed = max(0.0, max(t_stats))

    rng = random.Random(seed)
    exceed = 0
    runs = max(10, int(bootstrap))
    for _ in range(runs):
        idx = _bootstrap_indices(length, rng=rng)
        boot_stats = []
        for values in centered:
            sampled = [values[i] for i in idx]
            sigma = _std(sampled)
            boot_stats.append((_mean(sampled) / max(1e-8, sigma)) * math.sqrt(length))
        if max(0.0, max(boot_stats)) >= observed:
            exceed += 1
    return {"p_value": float(exceed / runs), "observed_statistic": float(observed)}

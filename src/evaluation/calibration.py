"""Post-hoc probability calibration helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(slots=True)
class TemperatureScalingResult:
    temperature: float
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float
    calibrated_probabilities: list[float]


def _clip_probability(value: float) -> float:
    return min(max(float(value), 1e-6), 1.0 - 1e-6)


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _binary_nll(labels: list[int], probabilities: list[float]) -> float:
    total = 0.0
    for label, probability in zip(labels, probabilities):
        p = _clip_probability(probability)
        total += -math.log(p) if int(label) == 1 else -math.log(1.0 - p)
    return total / max(1, len(labels))


def _brier(labels: list[int], probabilities: list[float]) -> float:
    if not labels:
        return float("nan")
    return sum((float(probability) - float(label)) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)


def fit_temperature_scaling(
    *,
    labels: list[int],
    probabilities: list[float],
    min_temperature: float = 0.25,
    max_temperature: float = 4.0,
    grid_size: int = 151,
) -> TemperatureScalingResult:
    """Fit scalar temperature by minimizing NLL over a 1D grid."""
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have equal lengths")
    if not labels:
        raise ValueError("cannot fit temperature on empty arrays")

    logits = [_logit(value) for value in probabilities]
    baseline_nll = _binary_nll(labels, probabilities)
    baseline_brier = _brier(labels, probabilities)

    lower = float(min_temperature)
    upper = float(max_temperature)
    if lower <= 0.0 or upper <= lower:
        raise ValueError("temperature bounds must satisfy 0 < min_temperature < max_temperature")
    steps = max(2, int(grid_size))

    best_temperature = 1.0
    best_probs = list(probabilities)
    best_nll = baseline_nll

    for index in range(steps):
        alpha = index / (steps - 1)
        temperature = lower + (upper - lower) * alpha
        scaled = [_sigmoid(logit / temperature) for logit in logits]
        nll = _binary_nll(labels, scaled)
        if nll < best_nll:
            best_nll = nll
            best_temperature = float(temperature)
            best_probs = scaled

    return TemperatureScalingResult(
        temperature=float(best_temperature),
        nll_before=float(baseline_nll),
        nll_after=float(best_nll),
        brier_before=float(baseline_brier),
        brier_after=float(_brier(labels, best_probs)),
        calibrated_probabilities=[float(value) for value in best_probs],
    )

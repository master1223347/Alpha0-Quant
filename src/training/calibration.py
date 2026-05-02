"""Bucketed post-hoc calibration utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.calibration import fit_temperature_scaling


@dataclass(slots=True)
class BucketCalibrationResult:
    bucket: str
    count: int
    method: str
    parameters: dict[str, float]
    nll_before: float
    nll_after: float
    brier_before: float
    brier_after: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bucket_key(value: Any) -> str:
    if value is None:
        return "__missing__"
    return str(value)


class TemperatureScalerByBucket:
    """Fit one scalar temperature per regime/liquidity/event bucket."""

    def __init__(self, *, min_bucket_count: int = 50) -> None:
        self.min_bucket_count = int(min_bucket_count)
        self.bucket_temperatures: dict[str, float] = {}
        self.default_temperature: float = 1.0
        self.results: dict[str, BucketCalibrationResult] = {}

    def fit(self, *, labels: list[int], probabilities: list[float], buckets: list[Any]) -> "TemperatureScalerByBucket":
        if not (len(labels) == len(probabilities) == len(buckets)):
            raise ValueError("labels, probabilities, and buckets must have equal lengths")
        if not labels:
            raise ValueError("cannot fit calibration on empty arrays")

        default = fit_temperature_scaling(labels=labels, probabilities=probabilities)
        self.default_temperature = float(default.temperature)
        grouped: dict[str, list[int]] = {}
        for index, bucket in enumerate(buckets):
            grouped.setdefault(_bucket_key(bucket), []).append(index)

        for bucket, indices in grouped.items():
            if len(indices) < self.min_bucket_count:
                self.bucket_temperatures[bucket] = self.default_temperature
                continue
            bucket_labels = [labels[index] for index in indices]
            bucket_probs = [probabilities[index] for index in indices]
            result = fit_temperature_scaling(labels=bucket_labels, probabilities=bucket_probs)
            self.bucket_temperatures[bucket] = float(result.temperature)
            self.results[bucket] = BucketCalibrationResult(
                bucket=bucket,
                count=len(indices),
                method="temperature",
                parameters={"temperature": float(result.temperature)},
                nll_before=float(result.nll_before),
                nll_after=float(result.nll_after),
                brier_before=float(result.brier_before),
                brier_after=float(result.brier_after),
            )
        return self

    def transform(self, *, probabilities: list[float], buckets: list[Any]) -> list[float]:
        if len(probabilities) != len(buckets):
            raise ValueError("probabilities and buckets must have equal lengths")
        import math

        output: list[float] = []
        for probability, bucket in zip(probabilities, buckets):
            p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
            logit = math.log(p / (1.0 - p))
            temperature = self.bucket_temperatures.get(_bucket_key(bucket), self.default_temperature)
            scaled = logit / max(float(temperature), 1e-6)
            if scaled >= 0:
                z = math.exp(-scaled)
                output.append(1.0 / (1.0 + z))
            else:
                z = math.exp(scaled)
                output.append(z / (1.0 + z))
        return output


class IsotonicByBucket:
    """Monotone empirical calibration fallback without sklearn dependency."""

    def __init__(self, *, bins: int = 10, min_bucket_count: int = 50) -> None:
        self.bins = max(2, int(bins))
        self.min_bucket_count = int(min_bucket_count)
        self.tables: dict[str, list[tuple[float, float]]] = {}

    def fit(self, *, labels: list[int], probabilities: list[float], buckets: list[Any]) -> "IsotonicByBucket":
        if not (len(labels) == len(probabilities) == len(buckets)):
            raise ValueError("labels, probabilities, and buckets must have equal lengths")
        grouped: dict[str, list[int]] = {}
        for index, bucket in enumerate(buckets):
            grouped.setdefault(_bucket_key(bucket), []).append(index)
        for bucket, indices in grouped.items():
            if len(indices) < self.min_bucket_count:
                continue
            pairs = sorted((float(probabilities[index]), int(labels[index])) for index in indices)
            bin_size = max(1, len(pairs) // self.bins)
            table: list[tuple[float, float]] = []
            for start in range(0, len(pairs), bin_size):
                chunk = pairs[start : start + bin_size]
                if not chunk:
                    continue
                avg_prob = sum(prob for prob, _ in chunk) / len(chunk)
                empirical = sum(label for _, label in chunk) / len(chunk)
                if table and empirical < table[-1][1]:
                    empirical = table[-1][1]
                table.append((avg_prob, empirical))
            self.tables[bucket] = table
        return self

    def transform(self, *, probabilities: list[float], buckets: list[Any]) -> list[float]:
        if len(probabilities) != len(buckets):
            raise ValueError("probabilities and buckets must have equal lengths")
        output: list[float] = []
        for probability, bucket in zip(probabilities, buckets):
            table = self.tables.get(_bucket_key(bucket))
            if not table:
                output.append(float(probability))
                continue
            candidate = table[0][1]
            for cutoff, value in table:
                if float(probability) >= cutoff:
                    candidate = value
                else:
                    break
            output.append(float(candidate))
        return output

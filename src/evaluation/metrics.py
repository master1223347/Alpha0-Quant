"""Classification metric helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    auc: float
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def logits_to_probabilities(logits: list[float]) -> list[float]:
    """Convert raw logits to probabilities."""
    return [sigmoid(float(logit)) for logit in logits]


def _binary_auc(y_true: list[int], y_prob: list[float]) -> float:
    positives = sum(1 for label in y_true if label == 1)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    sorted_pairs = sorted(zip(y_prob, y_true), key=lambda pair: pair[0])

    rank_sum_positive = 0.0
    rank = 1
    index = 0
    while index < len(sorted_pairs):
        tie_start = index
        tie_probability = sorted_pairs[index][0]
        while index < len(sorted_pairs) and sorted_pairs[index][0] == tie_probability:
            index += 1
        tie_end = index
        average_rank = (rank + (rank + (tie_end - tie_start) - 1)) / 2.0
        positives_in_tie = sum(1 for _, label in sorted_pairs[tie_start:tie_end] if label == 1)
        rank_sum_positive += positives_in_tie * average_rank
        rank += tie_end - tie_start

    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def compute_classification_metrics(
    y_true: list[int],
    y_prob: list[float],
    *,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute binary classification metrics."""
    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have equal lengths")
    if not y_true:
        raise ValueError("Cannot compute metrics on empty arrays")

    tp = fp = tn = fn = 0
    for label, probability in zip(y_true, y_prob):
        prediction = 1 if probability >= threshold else 0
        if prediction == 1 and label == 1:
            tp += 1
        elif prediction == 1 and label == 0:
            fp += 1
        elif prediction == 0 and label == 0:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    auc = _binary_auc(y_true, y_prob)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        auc=auc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
    )

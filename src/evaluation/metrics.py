"""Classification and probabilistic metric helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


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
    average_precision: float = float("nan")
    balanced_accuracy: float = float("nan")
    mcc: float = float("nan")
    macro_f1: float = float("nan")
    brier_score: float = float("nan")
    ece: float = float("nan")
    return_mae: float = float("nan")
    return_rmse: float = float("nan")
    return_nll: float = float("nan")
    avg_predicted_return: float = float("nan")
    avg_actual_return: float = float("nan")

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


def _average_precision(y_true: list[int], y_prob: list[float]) -> float:
    positives = sum(1 for label in y_true if label == 1)
    if positives == 0:
        return float("nan")

    sorted_pairs = sorted(zip(y_prob, y_true), key=lambda pair: pair[0], reverse=True)
    true_positive_count = 0
    false_positive_count = 0
    precision_sum = 0.0
    for _, label in sorted_pairs:
        if label == 1:
            true_positive_count += 1
            precision_sum += true_positive_count / max(1, true_positive_count + false_positive_count)
        else:
            false_positive_count += 1
    return precision_sum / positives


def _brier_score(y_true: list[int], y_prob: list[float]) -> float:
    return sum((float(prob) - float(label)) ** 2 for label, prob in zip(y_true, y_prob)) / len(y_true)


def _expected_calibration_error(y_true: list[int], y_prob: list[float], *, num_bins: int = 10) -> float:
    if not y_true:
        return float("nan")
    num_bins = max(1, int(num_bins))
    bin_totals = [0 for _ in range(num_bins)]
    bin_confidence = [0.0 for _ in range(num_bins)]
    bin_correct = [0.0 for _ in range(num_bins)]

    for label, probability in zip(y_true, y_prob):
        bin_index = min(num_bins - 1, int(float(probability) * num_bins))
        bin_totals[bin_index] += 1
        bin_confidence[bin_index] += float(probability)
        bin_correct[bin_index] += float(label)

    total = len(y_true)
    ece = 0.0
    for index in range(num_bins):
        count = bin_totals[index]
        if count == 0:
            continue
        avg_confidence = bin_confidence[index] / count
        accuracy = bin_correct[index] / count
        ece += (count / total) * abs(accuracy - avg_confidence)
    return ece


def _gaussian_nll(y_true: list[float], mean: list[float], log_scale: list[float]) -> float:
    total = 0.0
    for actual, predicted_mean, predicted_log_scale in zip(y_true, mean, log_scale):
        scale = math.log1p(math.exp(float(predicted_log_scale))) + 1e-6
        z = (float(actual) - float(predicted_mean)) / scale
        total += 0.5 * z * z + math.log(scale) + 0.5 * math.log(2.0 * math.pi)
    return total / len(y_true)


def _student_t_nll(y_true: list[float], mean: list[float], log_scale: list[float], *, df: float = 3.0) -> float:
    total = 0.0
    df = float(df)
    for actual, predicted_mean, predicted_log_scale in zip(y_true, mean, log_scale):
        scale = math.log1p(math.exp(float(predicted_log_scale))) + 1e-6
        z = (float(actual) - float(predicted_mean)) / scale
        total += (
            math.lgamma((df + 1.0) / 2.0)
            - math.lgamma(df / 2.0)
            + 0.5 * (math.log(df) + math.log(math.pi))
            + math.log(scale)
            + 0.5 * (df + 1.0) * math.log1p((z * z) / df)
        )
    return total / len(y_true)


def compute_classification_metrics(
    y_true: list[int],
    y_prob: list[float],
    *,
    threshold: float = 0.5,
    y_return: list[float] | None = None,
    mean_return: list[float] | None = None,
    log_scale: list[float] | None = None,
    distribution: str | None = None,
    calibration_bins: int = 10,
) -> ClassificationMetrics:
    """Compute classification, calibration, and optional distributional metrics."""
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
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    negative_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    positive_f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    negative_f1 = (
        2.0 * negative_precision * specificity / (negative_precision + specificity)
        if (negative_precision + specificity) > 0
        else 0.0
    )
    balanced_accuracy = 0.5 * (recall + specificity)
    mcc_denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator > 0 else 0.0
    macro_f1 = 0.5 * (positive_f1 + negative_f1)
    auc = _binary_auc(y_true, y_prob)
    average_precision = _average_precision(y_true, y_prob)
    brier_score = _brier_score(y_true, y_prob)
    ece = _expected_calibration_error(y_true, y_prob, num_bins=calibration_bins)

    return_mae = float("nan")
    return_rmse = float("nan")
    return_nll = float("nan")
    avg_predicted_return = float("nan")
    avg_actual_return = float("nan")

    if y_return is not None and mean_return is not None:
        if len(y_return) != len(mean_return):
            raise ValueError("y_return and mean_return must have equal lengths")
        avg_predicted_return = sum(float(value) for value in mean_return) / len(mean_return)
        avg_actual_return = sum(float(value) for value in y_return) / len(y_return)
        errors = [float(actual) - float(predicted) for actual, predicted in zip(y_return, mean_return)]
        return_mae = sum(abs(error) for error in errors) / len(errors)
        return_rmse = math.sqrt(sum(error * error for error in errors) / len(errors))

        if log_scale is not None:
            if len(log_scale) != len(mean_return):
                raise ValueError("log_scale and mean_return must have equal lengths")
            chosen_distribution = (distribution or "gaussian").lower().strip()
            if chosen_distribution == "student_t":
                return_nll = _student_t_nll(y_return, mean_return, log_scale)
            else:
                return_nll = _gaussian_nll(y_return, mean_return, log_scale)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        auc=auc,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        average_precision=average_precision,
        balanced_accuracy=balanced_accuracy,
        mcc=mcc,
        macro_f1=macro_f1,
        brier_score=brier_score,
        ece=ece,
        return_mae=return_mae,
        return_rmse=return_rmse,
        return_nll=return_nll,
        avg_predicted_return=avg_predicted_return,
        avg_actual_return=avg_actual_return,
    )

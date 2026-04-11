"""Validation loop helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics, logits_to_probabilities
from src.models.losses_prob import (
    extract_direction_target,
    extract_direction_logit,
    extract_forward_return,
    extract_log_scale,
    extract_mean_return,
    extract_threshold_logits,
    extract_threshold_target,
)


@dataclass(slots=True)
class ValidationResult:
    loss: float
    metrics: ClassificationMetrics
    probabilities: list[float]
    labels: list[int]
    up_probabilities: list[float]
    down_probabilities: list[float]
    neutral_probabilities: list[float]
    confidence_scores: list[float]
    three_class_labels: list[int]
    close: list[float]
    next_close: list[float]
    forward_returns: list[float]
    mean_returns: list[float]
    log_scales: list[float]
    sigma_values: list[float]

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["metrics"] = self.metrics.to_dict()
        return output


def _empty_metrics() -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=0.0,
        precision=0.0,
        recall=0.0,
        auc=float("nan"),
        tp=0,
        fp=0,
        tn=0,
        fn=0,
    )


def _softplus_scalar(value: float) -> float:
    if value > 20.0:
        return value
    if value < -20.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


def validate_epoch(model: Any, dataloader: Any, loss_fn: Any, *, device: Any) -> ValidationResult:
    """Run one validation pass and return aggregate metrics."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for validation") from exc

    model.eval()
    total_loss = 0.0
    batch_count = 0

    logits_list: list[float] = []
    labels: list[int] = []
    three_class_labels: list[int] = []
    up_probabilities: list[float] = []
    down_probabilities: list[float] = []
    neutral_probabilities: list[float] = []
    confidence_scores: list[float] = []
    close_values: list[float] = []
    next_close_values: list[float] = []
    forward_returns: list[float] = []
    mean_returns: list[float] = []
    log_scales: list[float] = []
    sigma_values: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["X"].to(device)
            targets = batch["y"].to(device)

            outputs = model(features)
            loss = loss_fn(outputs, targets, batch=batch)
            total_loss += float(loss.item())
            batch_count += 1

            direction_logit = extract_direction_logit(outputs)
            if torch.is_tensor(direction_logit):
                logits_list.extend(direction_logit.detach().cpu().reshape(-1).tolist())
            else:
                logits_list.extend(torch.as_tensor(direction_logit).reshape(-1).tolist())

            threshold_logits = extract_threshold_logits(outputs)
            if threshold_logits is not None:
                probs = torch.softmax(threshold_logits, dim=-1).detach().cpu()
                down_probabilities.extend(float(value) for value in probs[:, 0].reshape(-1).tolist())
                neutral_probabilities.extend(float(value) for value in probs[:, 1].reshape(-1).tolist())
                up_probabilities.extend(float(value) for value in probs[:, 2].reshape(-1).tolist())
                confidence_scores.extend(float(max(row[0], row[2])) for row in probs.tolist())

            forward_return = extract_forward_return(batch, device=device)
            if forward_return is not None:
                forward_returns.extend(float(value) for value in forward_return.detach().cpu().reshape(-1).tolist())

            direction_target = extract_direction_target(targets, batch, device=device)
            threshold_target = extract_threshold_target(batch, device=device)
            if threshold_target is not None:
                batch_three_class = [int(value) for value in threshold_target.detach().cpu().reshape(-1).tolist()]
                three_class_labels.extend(batch_three_class)
                labels.extend(1 if value == 2 else 0 for value in batch_three_class)
            elif direction_target is not None:
                binary_labels = [int(round(float(value))) for value in direction_target.detach().cpu().reshape(-1).tolist()]
                labels.extend(binary_labels)
                three_class_labels.extend(2 if value == 1 else 0 for value in binary_labels)
            elif forward_return is not None:
                derived_labels = [int(float(value) > 0.0) for value in forward_return.detach().cpu().reshape(-1).tolist()]
                labels.extend(derived_labels)
                three_class_labels.extend(2 if value == 1 else 0 for value in derived_labels)
            else:
                derived_labels = [int(float(value) > 0.0) for value in targets.detach().cpu().reshape(-1).tolist()]
                labels.extend(derived_labels)
                three_class_labels.extend(2 if value == 1 else 0 for value in derived_labels)

            close_values.extend(float(value) for value in batch["close"].detach().cpu().reshape(-1).tolist())
            next_close_values.extend(float(value) for value in batch["next_close"].detach().cpu().reshape(-1).tolist())

            mean_return = extract_mean_return(outputs)
            if mean_return is not None:
                mean_returns.extend(float(value) for value in mean_return.detach().cpu().reshape(-1).tolist())

            log_scale = extract_log_scale(outputs)
            if log_scale is not None:
                log_values = [float(value) for value in log_scale.detach().cpu().reshape(-1).tolist()]
                log_scales.extend(log_values)
                sigma_values.extend(float(_softplus_scalar(value) + 1e-6) for value in log_values)

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    if up_probabilities and down_probabilities:
        probabilities = []
        for up_probability, down_probability in zip(up_probabilities, down_probabilities):
            denominator = float(up_probability + down_probability)
            probabilities.append(float(up_probability / denominator) if denominator > 1e-8 else 0.5)
    else:
        probabilities = logits_to_probabilities(logits_list)
        up_probabilities = list(probabilities)
        down_probabilities = [1.0 - probability for probability in probabilities]
        neutral_probabilities = [0.0 for _ in probabilities]
        confidence_scores = [max(up_probability, down_probability) for up_probability, down_probability in zip(up_probabilities, down_probabilities)]

    if three_class_labels and len(three_class_labels) == len(probabilities):
        directional_indices = [index for index, label in enumerate(three_class_labels) if label != 1]
        directional_labels = [1 if three_class_labels[index] == 2 else 0 for index in directional_indices]
        directional_probabilities = [probabilities[index] for index in directional_indices]
        directional_returns = [forward_returns[index] for index in directional_indices] if forward_returns else None
        directional_mean_returns = [mean_returns[index] for index in directional_indices] if mean_returns else None
        directional_log_scales = [log_scales[index] for index in directional_indices] if log_scales else None
    else:
        directional_labels = labels
        directional_probabilities = probabilities
        directional_returns = forward_returns if forward_returns else None
        directional_mean_returns = mean_returns if mean_returns else None
        directional_log_scales = log_scales if log_scales else None

    if directional_labels and len(directional_labels) == len(directional_probabilities):
        metrics = compute_classification_metrics(
            directional_labels,
            directional_probabilities,
            y_return=directional_returns,
            mean_return=directional_mean_returns,
            log_scale=directional_log_scales,
            distribution=str(getattr(model, "distribution", "gaussian")),
        )
    else:
        metrics = _empty_metrics()

    return ValidationResult(
        loss=avg_loss,
        metrics=metrics,
        probabilities=probabilities,
        labels=labels,
        up_probabilities=up_probabilities,
        down_probabilities=down_probabilities,
        neutral_probabilities=neutral_probabilities,
        confidence_scores=confidence_scores,
        three_class_labels=three_class_labels,
        close=close_values,
        next_close=next_close_values,
        forward_returns=forward_returns,
        mean_returns=mean_returns,
        log_scales=log_scales,
        sigma_values=sigma_values,
    )

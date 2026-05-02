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
from src.models.losses_event import (
    extract_event_direction_logit,
    extract_event_direction_target,
    extract_event_logit,
    extract_event_target,
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
    action_scores: list[float]
    three_class_labels: list[int]
    close: list[float]
    next_close: list[float]
    forward_returns: list[float]
    mean_returns: list[float]
    log_scales: list[float]
    sigma_values: list[float]
    timestamps: list[Any]
    tickers: list[str]
    directional_sample_count: int
    total_sample_count: int
    event_probabilities: list[float] | None = None
    event_labels: list[int] | None = None
    event_direction_probabilities: list[float] | None = None
    event_direction_labels: list[int] | None = None
    up_event_probabilities: list[float] | None = None
    down_event_probabilities: list[float] | None = None
    no_event_probabilities: list[float] | None = None
    event_metrics: ClassificationMetrics | None = None
    event_direction_metrics: ClassificationMetrics | None = None

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["metrics"] = self.metrics.to_dict()
        if self.event_metrics is not None:
            output["event_metrics"] = self.event_metrics.to_dict()
        if self.event_direction_metrics is not None:
            output["event_direction_metrics"] = self.event_direction_metrics.to_dict()
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
    action_scores: list[float] = []
    close_values: list[float] = []
    next_close_values: list[float] = []
    forward_returns: list[float] = []
    mean_returns: list[float] = []
    log_scales: list[float] = []
    sigma_values: list[float] = []
    timestamp_values: list[Any] = []
    ticker_values: list[str] = []
    event_probabilities: list[float] = []
    event_labels: list[int] = []
    event_direction_probabilities: list[float] = []
    event_direction_labels: list[int] = []

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

            event_logit = extract_event_logit(outputs)
            if event_logit is not None:
                event_probs = torch.sigmoid(event_logit.reshape(-1)).detach().cpu().tolist()
                event_probabilities.extend(float(value) for value in event_probs)
                event_target = extract_event_target(batch, device=device)
                if event_target is not None:
                    event_labels.extend(int(round(float(value))) for value in event_target.detach().cpu().reshape(-1).tolist())

            event_direction_logit = extract_event_direction_logit(outputs)
            if event_direction_logit is not None:
                event_dir_probs = torch.sigmoid(event_direction_logit.reshape(-1)).detach().cpu().tolist()
                event_direction_probabilities.extend(float(value) for value in event_dir_probs)
                event_direction_target = extract_event_direction_target(batch, device=device)
                if event_direction_target is not None:
                    event_direction_labels.extend(
                        int(round(float(value))) for value in event_direction_target.detach().cpu().reshape(-1).tolist()
                    )

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
            if "timestamp" in batch:
                batch_timestamps = batch["timestamp"]
                if hasattr(batch_timestamps, "detach"):
                    timestamp_values.extend(batch_timestamps.detach().cpu().reshape(-1).tolist())
                elif isinstance(batch_timestamps, (list, tuple)):
                    timestamp_values.extend(list(batch_timestamps))
                else:
                    timestamp_values.append(batch_timestamps)
            if "ticker" in batch:
                batch_tickers = batch["ticker"]
                if hasattr(batch_tickers, "detach"):
                    ticker_values.extend(str(value) for value in batch_tickers.detach().cpu().reshape(-1).tolist())
                elif isinstance(batch_tickers, (list, tuple)):
                    ticker_values.extend(str(value) for value in batch_tickers)
                else:
                    ticker_values.append(str(batch_tickers))

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

    up_event_probabilities: list[float] = []
    down_event_probabilities: list[float] = []
    no_event_probabilities: list[float] = []
    if (
        event_probabilities
        and event_direction_probabilities
        and len(event_probabilities) == len(event_direction_probabilities)
        and len(event_probabilities) == len(close_values)
    ):
        up_event_probabilities = [
            max(0.0, min(1.0, float(event_probabilities[index]) * float(event_direction_probabilities[index])))
            for index in range(len(event_probabilities))
        ]
        down_event_probabilities = [
            max(0.0, min(1.0, float(event_probabilities[index]) * (1.0 - float(event_direction_probabilities[index]))))
            for index in range(len(event_probabilities))
        ]
        no_event_probabilities = [
            max(0.0, min(1.0, 1.0 - float(value))) for value in event_probabilities
        ]
        probabilities = []
        for up_probability, down_probability in zip(up_event_probabilities, down_event_probabilities):
            denominator = float(up_probability + down_probability)
            probabilities.append(float(up_probability / denominator) if denominator > 1e-8 else 0.5)
        up_probabilities = list(up_event_probabilities)
        down_probabilities = list(down_event_probabilities)
        neutral_probabilities = list(no_event_probabilities)
        confidence_scores = [
            max(float(up_probability), float(down_probability))
            for up_probability, down_probability in zip(up_probabilities, down_probabilities)
        ]

    if mean_returns and sigma_values and len(mean_returns) == len(sigma_values) == len(logits_list):
        floor = 0.10
        raw_scores = []
        for mu_value, sigma_value, logit_value in zip(mean_returns, sigma_values, logits_list):
            direction_sign = 1.0 if float(logit_value) >= 0.0 else -1.0
            safe_sigma = max(float(sigma_value), floor)
            raw_scores.append(direction_sign * (float(mu_value) / safe_sigma))
    elif probabilities:
        raw_scores = [(float(probability) - 0.5) * 2.0 for probability in probabilities]
    else:
        raw_scores = [0.0 for _ in logits_list]

    if raw_scores:
        mean_score = sum(raw_scores) / len(raw_scores)
        variance = sum((value - mean_score) ** 2 for value in raw_scores) / len(raw_scores)
        std_score = math.sqrt(variance)
        if std_score > 1e-8:
            action_scores = [(value - mean_score) / std_score for value in raw_scores]
        else:
            action_scores = [0.0 for _ in raw_scores]

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

    event_metrics: ClassificationMetrics | None = None
    if event_labels and len(event_labels) == len(event_probabilities):
        event_metrics = compute_classification_metrics(
            event_labels,
            event_probabilities,
        )

    event_direction_metrics: ClassificationMetrics | None = None
    if (
        event_labels
        and event_direction_labels
        and event_direction_probabilities
        and len(event_labels) == len(event_direction_labels) == len(event_direction_probabilities)
    ):
        conditional_indices = [index for index, label in enumerate(event_labels) if int(label) == 1]
        conditional_labels = [event_direction_labels[index] for index in conditional_indices]
        conditional_probs = [event_direction_probabilities[index] for index in conditional_indices]
        if conditional_labels and len(set(conditional_labels)) > 1:
            event_direction_metrics = compute_classification_metrics(
                conditional_labels,
                conditional_probs,
            )

    return ValidationResult(
        loss=avg_loss,
        metrics=metrics,
        probabilities=probabilities,
        labels=labels,
        up_probabilities=up_probabilities,
        down_probabilities=down_probabilities,
        neutral_probabilities=neutral_probabilities,
        confidence_scores=confidence_scores,
        action_scores=action_scores,
        three_class_labels=three_class_labels,
        close=close_values,
        next_close=next_close_values,
        forward_returns=forward_returns,
        mean_returns=mean_returns,
        log_scales=log_scales,
        sigma_values=sigma_values,
        timestamps=timestamp_values,
        tickers=ticker_values,
        directional_sample_count=int(len(directional_labels)),
        total_sample_count=int(len(probabilities)),
        event_probabilities=event_probabilities or None,
        event_labels=event_labels or None,
        event_direction_probabilities=event_direction_probabilities or None,
        event_direction_labels=event_direction_labels or None,
        up_event_probabilities=up_event_probabilities or None,
        down_event_probabilities=down_event_probabilities or None,
        no_event_probabilities=no_event_probabilities or None,
        event_metrics=event_metrics,
        event_direction_metrics=event_direction_metrics,
    )

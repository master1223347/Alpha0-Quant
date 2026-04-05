"""Validation loop helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics, logits_to_probabilities
from src.models.losses_prob import (
    extract_direction_target,
    extract_direction_logit,
    extract_forward_return,
    extract_log_scale,
    extract_mean_return,
)


@dataclass(slots=True)
class ValidationResult:
    loss: float
    metrics: ClassificationMetrics
    probabilities: list[float]
    labels: list[int]
    close: list[float]
    next_close: list[float]
    forward_returns: list[float]
    mean_returns: list[float]
    log_scales: list[float]

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
    close_values: list[float] = []
    next_close_values: list[float] = []
    forward_returns: list[float] = []
    mean_returns: list[float] = []
    log_scales: list[float] = []

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

            forward_return = extract_forward_return(batch, device=device)
            if forward_return is not None:
                forward_returns.extend(float(value) for value in forward_return.detach().cpu().reshape(-1).tolist())

            direction_target = extract_direction_target(targets, batch, device=device)
            if direction_target is not None:
                labels.extend(int(round(float(value))) for value in direction_target.detach().cpu().reshape(-1).tolist())
            elif forward_return is not None:
                labels.extend(int(float(value) > 0.0) for value in forward_return.detach().cpu().reshape(-1).tolist())
            else:
                labels.extend(int(float(value) > 0.0) for value in targets.detach().cpu().reshape(-1).tolist())

            close_values.extend(float(value) for value in batch["close"].detach().cpu().reshape(-1).tolist())
            next_close_values.extend(float(value) for value in batch["next_close"].detach().cpu().reshape(-1).tolist())

            mean_return = extract_mean_return(outputs)
            if mean_return is not None:
                mean_returns.extend(float(value) for value in mean_return.detach().cpu().reshape(-1).tolist())

            log_scale = extract_log_scale(outputs)
            if log_scale is not None:
                log_scales.extend(float(value) for value in log_scale.detach().cpu().reshape(-1).tolist())

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    probabilities = logits_to_probabilities(logits_list)

    if labels and len(labels) == len(probabilities):
        metrics = compute_classification_metrics(
            labels,
            probabilities,
            y_return=forward_returns if forward_returns else None,
            mean_return=mean_returns if mean_returns else None,
            log_scale=log_scales if log_scales else None,
            distribution=str(getattr(model, "distribution", "gaussian")),
        )
    else:
        metrics = _empty_metrics()

    return ValidationResult(
        loss=avg_loss,
        metrics=metrics,
        probabilities=probabilities,
        labels=labels,
        close=close_values,
        next_close=next_close_values,
        forward_returns=forward_returns,
        mean_returns=mean_returns,
        log_scales=log_scales,
    )

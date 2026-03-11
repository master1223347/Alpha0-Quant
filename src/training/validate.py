"""Validation loop helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics, logits_to_probabilities


@dataclass(slots=True)
class ValidationResult:
    loss: float
    metrics: ClassificationMetrics
    probabilities: list[float]
    labels: list[int]
    close: list[float]
    next_close: list[float]

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

    with torch.no_grad():
        for batch in dataloader:
            features = batch["X"].to(device)
            targets = batch["y"].to(device)

            logits = model(features)
            loss = loss_fn(logits, targets)
            total_loss += float(loss.item())
            batch_count += 1

            logits_list.extend(logits.detach().cpu().tolist())
            labels.extend(int(value) for value in targets.detach().cpu().tolist())
            close_values.extend(float(value) for value in batch["close"].detach().cpu().tolist())
            next_close_values.extend(float(value) for value in batch["next_close"].detach().cpu().tolist())

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    probabilities = logits_to_probabilities(logits_list)

    if labels:
        metrics = compute_classification_metrics(labels, probabilities)
    else:
        metrics = _empty_metrics()

    return ValidationResult(
        loss=avg_loss,
        metrics=metrics,
        probabilities=probabilities,
        labels=labels,
        close=close_values,
        next_close=next_close_values,
    )

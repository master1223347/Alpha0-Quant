"""Uncertainty estimation helpers for trained neural models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from src.models.losses_event import extract_event_direction_logit, extract_event_logit
from src.models.losses_prob import extract_direction_logit, extract_threshold_logits


@dataclass(slots=True)
class MCDropoutUncertaintyReport:
    sample_count: int
    mc_samples: int
    mean_probability_std: float
    mean_predictive_entropy: float
    high_uncertainty_share: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entropy(probability: float) -> float:
    p = min(max(float(probability), 1e-8), 1.0 - 1e-8)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def _activate_dropout_only(model: Any) -> None:
    model.eval()
    for module in model.modules():
        name = module.__class__.__name__.lower()
        if "dropout" in name:
            module.train()


def _extract_trade_probability(outputs: Any) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for MC dropout uncertainty") from exc

    event_logit = extract_event_logit(outputs)
    event_direction_logit = extract_event_direction_logit(outputs)
    if event_logit is not None and event_direction_logit is not None:
        event_prob = torch.sigmoid(event_logit.reshape(-1))
        direction_prob = torch.sigmoid(event_direction_logit.reshape(-1))
        up_event = event_prob * direction_prob
        down_event = event_prob * (1.0 - direction_prob)
        denominator = torch.clamp(up_event + down_event, min=1e-8)
        return up_event / denominator

    threshold_logits = extract_threshold_logits(outputs)
    if threshold_logits is not None:
        probs = torch.softmax(threshold_logits, dim=-1)
        denominator = torch.clamp(probs[:, 0] + probs[:, 2], min=1e-8)
        return probs[:, 2] / denominator

    direction_logit = extract_direction_logit(outputs)
    return torch.sigmoid(direction_logit.reshape(-1))


def estimate_mc_dropout_uncertainty(
    model: Any,
    dataloader: Any,
    *,
    device: Any,
    samples: int = 20,
    high_uncertainty_std: float = 0.10,
) -> MCDropoutUncertaintyReport:
    """Run stochastic dropout inference and summarize probability dispersion."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for MC dropout uncertainty") from exc

    sample_count = max(1, int(samples))
    all_passes: list[list[float]] = []
    was_training = bool(getattr(model, "training", False))
    with torch.no_grad():
        for _ in range(sample_count):
            _activate_dropout_only(model)
            probabilities: list[float] = []
            for batch in dataloader:
                features = batch["X"].to(device)
                outputs = model(features)
                probs = _extract_trade_probability(outputs)
                probabilities.extend(float(value) for value in probs.detach().cpu().reshape(-1).tolist())
            all_passes.append(probabilities)
    model.train(was_training)

    if not all_passes or not all_passes[0]:
        return MCDropoutUncertaintyReport(
            sample_count=0,
            mc_samples=sample_count,
            mean_probability_std=float("nan"),
            mean_predictive_entropy=float("nan"),
            high_uncertainty_share=float("nan"),
        )

    n = min(len(values) for values in all_passes)
    std_values: list[float] = []
    entropies: list[float] = []
    for index in range(n):
        values = [float(run[index]) for run in all_passes]
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        std_value = math.sqrt(variance)
        std_values.append(std_value)
        entropies.append(_entropy(mean_value))

    high_uncertainty_count = sum(1 for value in std_values if value >= float(high_uncertainty_std))
    return MCDropoutUncertaintyReport(
        sample_count=n,
        mc_samples=sample_count,
        mean_probability_std=sum(std_values) / len(std_values),
        mean_predictive_entropy=sum(entropies) / len(entropies),
        high_uncertainty_share=high_uncertainty_count / len(std_values),
    )

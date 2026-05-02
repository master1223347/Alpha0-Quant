"""Loss helpers for event-meta targets.

The event target is decomposed as:
  P(up_event) = P(event) * P(up | event)
  P(down_event) = P(event) * (1 - P(up | event))
"""

from __future__ import annotations

from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


def _extract_from_dict(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def extract_event_logit(outputs: Any) -> Any | None:
    if not isinstance(outputs, dict):
        return None
    return _extract_from_dict(outputs, ("event_logit", "event_logits"))


def extract_event_direction_logit(outputs: Any) -> Any | None:
    if not isinstance(outputs, dict):
        return None
    return _extract_from_dict(
        outputs,
        ("event_direction_logit", "direction_logit_cond_event", "conditional_direction_logit"),
    )


def _resolve_tensor(value: Any, *, device: Any, dtype: Any | None = None) -> Any | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.to(device=device)
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    tensor = torch.as_tensor(value, device=device)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def extract_event_target(batch: dict[str, Any] | None, *, device: Any) -> Any | None:
    if batch is None:
        return None
    target = _extract_from_dict(batch, ("event_label", "event_target", "y_event"))
    if target is None:
        return None
    return _resolve_tensor(target, device=device, dtype=torch.float32).reshape(-1)


def extract_event_direction_target(batch: dict[str, Any] | None, *, device: Any) -> Any | None:
    if batch is None:
        return None
    target = _extract_from_dict(
        batch,
        ("event_direction_label", "conditional_direction_label", "y_event_direction", "direction_label"),
    )
    if target is None:
        return None
    return _resolve_tensor(target, device=device, dtype=torch.float32).reshape(-1)


def extract_event_sample_weight(
    batch: dict[str, Any] | None,
    *,
    device: Any,
    cap: float = 2.0,
) -> Any | None:
    if batch is None:
        return None
    value = _extract_from_dict(batch, ("event_magnitude", "event_sample_weight", "event_strength"))
    if value is None:
        return None
    weights = _resolve_tensor(value, device=device, dtype=torch.float32).reshape(-1)
    return torch.clamp(weights, min=0.0, max=float(cap))


def event_focal_loss(
    logits: Any,
    targets: Any,
    *,
    gamma: float = 2.0,
    sample_weight: Any | None = None,
    pos_weight: Any | None = None,
) -> Any:
    """Binary focal BCE with optional clipped per-sample weights."""
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).to(dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    prob = torch.sigmoid(logits)
    p_t = torch.where(targets >= 0.5, prob, 1.0 - prob)
    focal = torch.pow(torch.clamp(1.0 - p_t, min=0.0), float(gamma))
    loss = bce * focal
    if sample_weight is not None:
        loss = loss * sample_weight.reshape(-1).to(dtype=loss.dtype, device=loss.device)
    return loss.mean()


def masked_direction_loss(
    logits: Any,
    targets: Any,
    event_mask: Any,
    *,
    sample_weight: Any | None = None,
    pos_weight: Any | None = None,
) -> Any:
    """BCE on conditional direction, evaluated only where event_mask is true."""
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).to(dtype=torch.float32)
    mask = event_mask.reshape(-1).to(dtype=torch.bool)
    if not torch.any(mask):
        return logits.new_tensor(0.0)
    selected_logits = logits[mask]
    selected_targets = targets[mask]
    loss = F.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    if sample_weight is not None:
        loss = loss * sample_weight.reshape(-1).to(dtype=loss.dtype, device=loss.device)[mask]
    return loss.mean()


if nn is not None:

    class EventDirectionLoss(nn.Module):
        """Standalone two-head event/direction loss."""

        def __init__(
            self,
            *,
            event_weight: float = 1.0,
            direction_weight: float = 1.0,
            focal_gamma: float = 2.0,
            sample_weight_cap: float = 2.0,
        ) -> None:
            super().__init__()
            self.event_weight = float(event_weight)
            self.direction_weight = float(direction_weight)
            self.focal_gamma = float(focal_gamma)
            self.sample_weight_cap = float(sample_weight_cap)
            self.last_components: dict[str, float] = {}

        def forward(self, outputs: Any, batch: dict[str, Any]) -> Any:
            event_logit = extract_event_logit(outputs)
            direction_logit = extract_event_direction_logit(outputs)
            if event_logit is None:
                raise ValueError("event loss requires outputs['event_logit']")

            device = event_logit.device if torch.is_tensor(event_logit) else torch.device("cpu")
            event_target = extract_event_target(batch, device=device)
            direction_target = extract_event_direction_target(batch, device=device)
            sample_weight = extract_event_sample_weight(batch, device=device, cap=self.sample_weight_cap)
            if event_target is None:
                raise ValueError("event loss requires batch['event_label']")

            total = event_logit.new_tensor(0.0)
            components: dict[str, float] = {}
            event_loss_value = event_focal_loss(
                event_logit,
                event_target,
                gamma=self.focal_gamma,
                sample_weight=sample_weight,
            )
            total = total + self.event_weight * event_loss_value
            components["event_loss"] = float(event_loss_value.detach().cpu().item())

            if direction_logit is not None and direction_target is not None:
                direction_loss_value = masked_direction_loss(
                    direction_logit,
                    direction_target,
                    event_target >= 0.5,
                    sample_weight=sample_weight,
                )
                total = total + self.direction_weight * direction_loss_value
                components["event_direction_loss"] = float(direction_loss_value.detach().cpu().item())

            self.last_components = components
            return total

else:

    class EventDirectionLoss:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise ModuleNotFoundError("torch is required to use EventDirectionLoss")

"""Learning-rate scheduler helpers."""

from __future__ import annotations

from typing import Any


def create_scheduler(
    optimizer: Any,
    *,
    scheduler_name: str,
    total_epochs: int,
    step_size: int = 4,
    gamma: float = 0.5,
) -> Any | None:
    """Create optional scheduler for optimizer."""
    try:
        from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for scheduler creation") from exc

    name = scheduler_name.lower().strip()
    if name in {"none", "", "off"}:
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(total_epochs, 1))
    if name == "step":
        return StepLR(optimizer, step_size=max(step_size, 1), gamma=gamma)
    raise ValueError(f"Unsupported scheduler_name: {scheduler_name}")

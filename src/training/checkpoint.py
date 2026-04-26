"""Model checkpoint save/load helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_checkpoint(
    *,
    path: str | Path,
    model: Any,
    optimizer: Any,
    epoch: int,
    metric_score: float,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Save model training state to disk."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for checkpoint saving") from exc

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "metric_score": metric_score,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history or [],
    }
    torch.save(payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    *,
    path: str | Path,
    model: Any,
    optimizer: Any | None = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load checkpoint into model and optional optimizer."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for checkpoint loading") from exc

    checkpoint_path = Path(path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint payload at {checkpoint_path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint

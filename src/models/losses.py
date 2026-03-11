"""Loss helpers for binary classification."""

from __future__ import annotations

from typing import Any


def compute_pos_weight(labels: list[int] | Any) -> float:
    """Compute positive-class weight as neg_count / pos_count."""
    positives = 0
    negatives = 0
    for label in labels:
        if int(label) == 1:
            positives += 1
        else:
            negatives += 1
    if positives == 0:
        return 1.0
    return negatives / positives


def build_bce_with_logits_loss(*, pos_weight: float | None = None) -> Any:
    """Build BCEWithLogitsLoss with optional positive class weighting."""
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for build_bce_with_logits_loss") from exc

    if pos_weight is None:
        return nn.BCEWithLogitsLoss()

    weight_tensor = torch.tensor(float(pos_weight), dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

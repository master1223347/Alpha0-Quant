"""Model training loop."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.models.losses import compute_pos_weight
from src.models.losses_prob import build_model_loss
from src.training.checkpoint import save_checkpoint
from src.training.scheduler import create_scheduler
from src.training.validate import ValidationResult, validate_epoch
from src.utils.seed import set_global_seed


@dataclass(slots=True)
class TrainingArtifacts:
    best_epoch: int
    best_score: float
    best_checkpoint_path: str
    history: list[dict[str, Any]]
    final_validation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_device(device_name: str) -> Any:
    import torch

    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _score_from_validation(result: ValidationResult) -> float:
    auc = result.metrics.auc
    if isinstance(auc, float) and not math.isnan(auc):
        return auc
    return result.metrics.accuracy


def _is_binary_labels(values: list[float], *, tol: float = 1e-6) -> bool:
    if not values:
        return False
    for value in values:
        if abs(value - 0.0) <= tol or abs(value - 1.0) <= tol:
            continue
        return False
    return True


def train_model(config: Any, dataloaders: dict[str, Any], model: Any) -> TrainingArtifacts:
    """Train model using train/val dataloaders and save best checkpoint."""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for training") from exc

    set_global_seed(int(config.training.seed))
    device = _resolve_device(str(config.training.device))
    model.to(device)

    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val", train_loader)

    raw_labels = [float(value) for value in train_loader.dataset.y.tolist()]
    pos_weight = None
    if _is_binary_labels(raw_labels):
        labels = [int(round(value)) for value in raw_labels]
        pos_weight = compute_pos_weight(labels)
    loss_fn = build_model_loss(
        model,
        pos_weight=pos_weight,
        distribution=str(getattr(config.model, "distribution", "gaussian")),
        direction_weight=float(getattr(config.training, "direction_loss_weight", 1.0)),
        threshold_weight=float(getattr(config.training, "threshold_loss_weight", 0.25)),
        regression_weight=float(getattr(config.training, "regression_loss_weight", 1.0)),
        rank_weight=float(getattr(config.training, "rank_loss_weight", 0.10)),
        regime_weight=float(getattr(config.training, "regime_loss_weight", 0.10)),
        student_t_df=float(getattr(config.training, "student_t_df", 3.0)),
        regression_loss=str(getattr(config.training, "regression_loss", "nll")),
        regression_huber_delta=float(getattr(config.training, "regression_huber_delta", 1.0)),
        volatility_consistency_weight=float(getattr(config.training, "volatility_consistency_weight", 0.0)),
        volatility_consistency_limit=float(getattr(config.training, "volatility_consistency_limit", 2.5)),
        temporal_smoothness_weight=float(getattr(config.training, "temporal_smoothness_weight", 0.0)),
        temporal_smoothness_max_gap_seconds=int(getattr(config.training, "temporal_smoothness_max_gap_seconds", 3600)),
        cross_sectional_reg_weight=float(getattr(config.training, "cross_sectional_reg_weight", 0.0)),
        cross_sectional_reg_limit=float(getattr(config.training, "cross_sectional_reg_limit", 2.5)),
        calibration_aux_weight=float(getattr(config.training, "calibration_aux_weight", 0.0)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=str(config.training.scheduler_name),
        total_epochs=int(config.training.epochs),
        step_size=int(config.training.scheduler_step_size),
        gamma=float(config.training.scheduler_gamma),
    )

    checkpoint_dir = Path(config.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_model.pt"

    history: list[dict[str, Any]] = []
    best_epoch = -1
    best_score = float("-inf")
    best_validation: ValidationResult | None = None

    for epoch in range(1, int(config.training.epochs) + 1):
        model.train()
        running_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            features = batch["X"].to(device)
            targets = batch["y"].to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = loss_fn(outputs, targets, batch=batch)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batch_count += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / batch_count if batch_count > 0 else 0.0
        validation = validate_epoch(model, val_loader, loss_fn, device=device)
        score = _score_from_validation(validation)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation.loss,
            "val_metrics": validation.metrics.to_dict(),
            "score": score,
        }
        history.append(epoch_record)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_validation = validation
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metric_score=score,
                history=history,
            )

    log_path = Path(config.training.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if best_validation is None:
        best_validation = validate_epoch(model, val_loader, loss_fn, device=device)
        best_score = _score_from_validation(best_validation)
        best_epoch = int(config.training.epochs)

    return TrainingArtifacts(
        best_epoch=best_epoch,
        best_score=best_score,
        best_checkpoint_path=str(best_checkpoint_path),
        history=history,
        final_validation=best_validation.to_dict(),
    )

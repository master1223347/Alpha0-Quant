"""Evaluation pipeline entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.default_config import ExperimentConfig, get_default_config
from src.dataset.dataloader import create_dataloaders
from src.evaluation.analysis import EvaluationReport, build_evaluation_report, save_evaluation_report
from src.evaluation.backtest import run_backtest
from src.models.losses import compute_pos_weight
from src.models.losses_prob import build_model_loss
from src.training.checkpoint import load_checkpoint
from src.training.validate import validate_epoch
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EvaluatePipelineArtifacts:
    report: EvaluationReport
    report_path: str


def _is_binary_labels(values: list[float], *, tol: float = 1e-6) -> bool:
    if not values:
        return False
    for value in values:
        if abs(value - 0.0) <= tol or abs(value - 1.0) <= tol:
            continue
        return False
    return True


def run_evaluation_pipeline(
    *,
    config: ExperimentConfig | None = None,
    model: Any,
    dataset_artifacts: Any,
    checkpoint_path: str | Path | None = None,
) -> EvaluatePipelineArtifacts:
    """Evaluate trained model on the test split and run backtest."""
    config = config or get_default_config()
    dataloaders = create_dataloaders(
        dataset_artifacts.datasets,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle_train=False,
    )

    test_loader = dataloaders.get("test")
    if test_loader is None or len(test_loader.dataset) == 0:
        test_loader = dataloaders.get("val") or dataloaders.get("train")
    if test_loader is None:
        raise ValueError("No DataLoader available for evaluation")

    if checkpoint_path is not None:
        load_checkpoint(path=checkpoint_path, model=model)

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for evaluation") from exc

    device_name = str(config.training.device)
    if device_name == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device_name
    device = torch.device(resolved_device)
    model.to(device)

    pos_weight = None
    if hasattr(test_loader.dataset, "y"):
        raw_labels = [float(value) for value in test_loader.dataset.y.tolist()]
        if _is_binary_labels(raw_labels):
            labels = [int(round(value)) for value in raw_labels]
            pos_weight = compute_pos_weight(labels) if labels else None
    loss_fn = build_model_loss(model, pos_weight=pos_weight).to(device)

    include_costs = bool(getattr(config.backtest, "include_costs", True))
    cost_bps_per_trade = float(getattr(config.backtest, "cost_bps_per_trade", 0.0)) if include_costs else 0.0
    slippage_bps = float(getattr(config.backtest, "slippage_bps", 0.0)) if include_costs else 0.0

    validation = validate_epoch(model, test_loader, loss_fn, device=device)
    backtest = run_backtest(
        validation.probabilities,
        validation.close,
        validation.next_close,
        long_threshold=config.backtest.long_threshold,
        short_threshold=config.backtest.short_threshold,
        periods_per_year=config.backtest.periods_per_year,
        flip_positions=bool(getattr(config.backtest, "flip_positions", False)),
        cost_bps_per_trade=cost_bps_per_trade,
        slippage_bps=slippage_bps,
    )

    report = build_evaluation_report(
        model_name=config.model.model_name,
        split="test",
        metrics=validation.metrics,
        backtest=backtest,
    )

    report_path = Path(config.training.metrics_path).with_name("evaluation_report.json")
    save_evaluation_report(report, report_path)

    backtest_path = Path(config.training.metrics_path).with_name("backtest.json")
    with backtest_path.open("w", encoding="utf-8") as handle:
        json.dump(backtest.to_dict(), handle, indent=2)

    LOGGER.info("Evaluation report written to %s", report_path)
    return EvaluatePipelineArtifacts(report=report, report_path=str(report_path))

"""Training pipeline entrypoint."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.default_config import ExperimentConfig, get_default_config
from src.dataset.dataloader import create_dataloaders
from src.models.baseline import BaselineMLP
from src.models.gnn import gnn_panel
from src.models.panel_transformer import panel_transformer
from src.models.tcn import tcn_encoder
from src.pipeline.build_dataset import BuildDatasetArtifacts, build_dataset
from src.training.train import TrainingArtifacts, train_model
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class TrainPipelineArtifacts:
    training: TrainingArtifacts
    dataset: BuildDatasetArtifacts
    model: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": self.training.to_dict(),
            "dataset": self.dataset.to_dict(),
        }


def _requires_multitask_objective(config: ExperimentConfig) -> bool:
    weighted_terms = (
        float(getattr(config.training, "threshold_loss_weight", 0.0)),
        float(getattr(config.training, "regression_loss_weight", 0.0)),
        float(getattr(config.training, "rank_loss_weight", 0.0)),
        float(getattr(config.training, "return_rank_weight", 0.0)),
        float(getattr(config.training, "regime_loss_weight", 0.0)),
        float(getattr(config.training, "score_alignment_weight", 0.0)),
        float(getattr(config.training, "volatility_consistency_weight", 0.0)),
        float(getattr(config.training, "temporal_smoothness_weight", 0.0)),
        float(getattr(config.training, "cross_sectional_reg_weight", 0.0)),
        float(getattr(config.training, "calibration_aux_weight", 0.0)),
    )
    return any(weight > 0.0 for weight in weighted_terms)


def _requires_probabilistic_output(config: ExperimentConfig) -> bool:
    weighted_terms = (
        float(getattr(config.training, "regression_loss_weight", 0.0)),
        float(getattr(config.training, "return_rank_weight", 0.0)),
        float(getattr(config.training, "score_alignment_weight", 0.0)),
        float(getattr(config.training, "volatility_consistency_weight", 0.0)),
        float(getattr(config.training, "temporal_smoothness_weight", 0.0)),
        float(getattr(config.training, "cross_sectional_reg_weight", 0.0)),
        float(getattr(config.training, "calibration_aux_weight", 0.0)),
    )
    return any(weight > 0.0 for weight in weighted_terms)


def _resolve_model_window_size(train_dataset: Any, *, fallback: int) -> int:
    values = getattr(train_dataset, "X", None)
    shape = getattr(values, "shape", None)
    if shape is None or len(shape) < 2:
        return int(fallback)
    return int(shape[1])


def _build_example_input(train_dataset: Any) -> Any | None:
    values = getattr(train_dataset, "X", None)
    if values is None:
        return None
    try:
        import torch
    except ModuleNotFoundError:
        return None
    try:
        if hasattr(values, "__getitem__") and len(values) > 0:
            sample = values[0]
        else:
            return None
        tensor = torch.as_tensor(sample, dtype=torch.float32)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.dim() == 1:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor
    except Exception:
        return None


def _export_trained_model(config: ExperimentConfig, *, model: Any, train_dataset: Any) -> str | None:
    export_format = str(getattr(config.deployment, "export_format", "none")).strip().lower()
    if export_format in {"", "none", "off", "disabled"}:
        return None

    try:
        import torch
    except ModuleNotFoundError:
        LOGGER.warning("Skipping model export because torch is unavailable")
        return None

    example = _build_example_input(train_dataset)
    if example is None:
        LOGGER.warning("Skipping model export because example input could not be derived from train dataset")
        return None

    export_path = Path(str(getattr(config.deployment, "export_path", "models/exports/model_export.pt2")))
    export_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    if export_format == "torchscript":
        if not bool(getattr(config.deployment, "allow_deprecated_torchscript", False)):
            LOGGER.warning(
                "Skipping TorchScript export because allow_deprecated_torchscript is False "
                "(TorchScript is deprecated in favor of torch.export)."
            )
            return None
        scripted = torch.jit.trace(model, example)
        scripted.save(str(export_path))
        LOGGER.info("Exported TorchScript artifact to %s", export_path)
        return str(export_path)

    if export_format in {"torch_export", "export", "pt2"}:
        if not hasattr(torch, "export") or not hasattr(torch.export, "export"):
            LOGGER.warning("Skipping torch.export artifact because current torch build lacks torch.export APIs")
            return None
        exported_program = torch.export.export(model, (example,))
        if hasattr(torch.export, "save"):
            torch.export.save(exported_program, str(export_path))
        else:
            raise RuntimeError("torch.export.save is unavailable in current torch build")
        LOGGER.info("Exported torch.export artifact to %s", export_path)
        return str(export_path)

    LOGGER.warning("Unknown deployment.export_format=%s; skipping export", export_format)
    return None


def _build_model(config: ExperimentConfig, *, num_features: int, window_size: int) -> Any:
    model_name = config.model.model_name.lower()
    minn_enabled = bool(getattr(config.model, "minn_enabled", False))
    multitask_output = bool(getattr(config.model, "multitask_output", minn_enabled))
    probabilistic_output = bool(getattr(config.model, "probabilistic_output", minn_enabled))
    requires_multitask = _requires_multitask_objective(config)
    requires_probabilistic = _requires_probabilistic_output(config)
    if requires_multitask and not multitask_output:
        LOGGER.warning(
            "Enabling multitask_output for %s because configured training weights require multi-head outputs",
            config.name,
        )
        multitask_output = True
    if requires_probabilistic and not probabilistic_output:
        LOGGER.warning(
            "Enabling probabilistic_output for %s because configured training weights require return distribution heads",
            config.name,
        )
        probabilistic_output = True
    if probabilistic_output and not multitask_output:
        LOGGER.warning(
            "Enabling multitask_output for %s because probabilistic_output requires dict model outputs",
            config.name,
        )
        multitask_output = True
    include_rank_head = bool(getattr(config.model, "include_rank_head", False))
    include_regime_head = bool(getattr(config.model, "include_regime_head", False))
    if float(getattr(config.training, "rank_loss_weight", 0.0)) > 0.0:
        include_rank_head = True
    if float(getattr(config.training, "regime_loss_weight", 0.0)) > 0.0:
        include_regime_head = True
    regime_classes = int(getattr(config.model, "regime_classes", 3))
    distribution = str(getattr(config.model, "distribution", "gaussian"))
    if model_name in {"baseline", "baseline_mlp", "mlp", "encoder", "encoder_mlp"}:
        return BaselineMLP(
            window_size=window_size,
            num_features=num_features,
            hidden_dims=tuple(config.model.hidden_dims),
            dropout=float(config.model.dropout),
            multitask_output=multitask_output,
            probabilistic_output=probabilistic_output,
            include_rank_score=include_rank_head,
            include_regime_logits=include_regime_head,
            regime_classes=regime_classes,
            distribution=distribution,
        )
    if model_name in {"tcn", "tcn_encoder"}:
        return tcn_encoder(
            window_size=window_size,
            num_features=num_features,
            dropout=float(config.model.dropout),
            multitask_output=multitask_output,
            include_rank_score=include_rank_head,
            include_regime_logits=include_regime_head,
            regime_classes=regime_classes,
            distribution=distribution,
            probabilistic_output=probabilistic_output,
        )
    if model_name in {"panel_transformer", "transformer", "minn_transformer"}:
        return panel_transformer(
            window_size=window_size,
            num_features=num_features,
            dropout=float(config.model.dropout),
            multitask_output=multitask_output,
            include_rank_score=include_rank_head,
            include_regime_logits=include_regime_head,
            regime_classes=regime_classes,
            distribution=distribution,
            probabilistic_output=probabilistic_output,
        )
    if model_name in {"gnn_panel", "gnn"}:
        return gnn_panel(
            window_size=window_size,
            num_features=num_features,
            dropout=float(config.model.dropout),
            multitask_output=multitask_output,
            include_rank_score=include_rank_head,
            include_regime_logits=include_regime_head,
            regime_classes=regime_classes,
            distribution=distribution,
            probabilistic_output=probabilistic_output,
        )
    raise ValueError(f"Unsupported model_name: {config.model.model_name}")


def run_training_pipeline(
    config: ExperimentConfig | None = None,
    *,
    exchange: str | None = None,
    asset_type: str | None = None,
    dataset_artifacts: BuildDatasetArtifacts | None = None,
) -> TrainPipelineArtifacts:
    """Build dataset, train model, and persist training outputs."""
    config = config or get_default_config()
    if dataset_artifacts is None:
        dataset_artifacts = build_dataset(config, exchange=exchange, asset_type=asset_type)

    train_dataset = dataset_artifacts.datasets.get("train")
    if train_dataset is None or len(train_dataset) == 0:
        raise ValueError("Training dataset is empty after preprocessing")

    dataloaders = create_dataloaders(
        dataset_artifacts.datasets,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle_train=True,
    )
    model_window_size = _resolve_model_window_size(train_dataset, fallback=int(config.dataset.window_size))
    model = _build_model(
        config,
        num_features=len(dataset_artifacts.feature_columns),
        window_size=model_window_size,
    )
    training_artifacts = train_model(config, dataloaders, model)
    _export_trained_model(config, model=model, train_dataset=train_dataset)

    metrics_path = Path(config.training.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(training_artifacts.to_dict(), handle, indent=2)
    LOGGER.info("Training metrics written to %s", metrics_path)

    return TrainPipelineArtifacts(training=training_artifacts, dataset=dataset_artifacts, model=model)

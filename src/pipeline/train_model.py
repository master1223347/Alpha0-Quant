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


def _build_model(config: ExperimentConfig, *, num_features: int) -> Any:
    model_name = config.model.model_name.lower()
    minn_enabled = bool(getattr(config.model, "minn_enabled", False))
    multitask_output = bool(getattr(config.model, "multitask_output", minn_enabled))
    probabilistic_output = bool(getattr(config.model, "probabilistic_output", minn_enabled))
    include_rank_head = bool(getattr(config.model, "include_rank_head", False))
    include_regime_head = bool(getattr(config.model, "include_regime_head", False))
    regime_classes = int(getattr(config.model, "regime_classes", 3))
    distribution = str(getattr(config.model, "distribution", "gaussian"))
    if model_name in {"baseline", "baseline_mlp", "mlp", "encoder", "encoder_mlp"}:
        return BaselineMLP(
            window_size=config.dataset.window_size,
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
            window_size=config.dataset.window_size,
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
            window_size=config.dataset.window_size,
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
            window_size=config.dataset.window_size,
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
    model = _build_model(config, num_features=len(dataset_artifacts.feature_columns))
    training_artifacts = train_model(config, dataloaders, model)

    metrics_path = Path(config.training.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(training_artifacts.to_dict(), handle, indent=2)
    LOGGER.info("Training metrics written to %s", metrics_path)

    return TrainPipelineArtifacts(training=training_artifacts, dataset=dataset_artifacts, model=model)

"""Default runtime configuration for Alpha0 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DataConfig:
    raw_root: str = "data/raw"
    processed_dir: str = "data/processed"
    metadata_dir: str = "data/metadata"
    features_path: str = "data/processed/features.parquet"
    dataset_path: str = "data/processed/dataset.parquet"
    labels_path: str = "data/processed/labels.parquet"
    max_tickers: int | None = None
    min_sequence_length: int = 40


@dataclass(slots=True)
class UniverseConfig:
    exchange: str | None = None
    asset_type: str | None = None
    tickers: tuple[str, ...] | None = None
    max_tickers: int | None = None
    min_sequence_length: int | None = None


@dataclass(slots=True)
class TargetConfig:
    horizon: int | None = None
    threshold: float = 0.001
    volatility_window: int = 20
    zscore_window: int = 20
    primary_target: str = "label"


@dataclass(slots=True)
class FeatureConfig:
    momentum_lookback: int = 3
    volatility_window: int = 5
    relative_volume_window: int = 5
    volume_window: int = 10
    normalize: bool = True


@dataclass(slots=True)
class DatasetConfig:
    window_size: int = 32
    stride: int = 1
    label_horizon: int = 1
    split_mode: str = "per_ticker"
    dataset_type: str = "auto"
    panel_context_size: int | None = None
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 256
    num_workers: int = 0


@dataclass(slots=True)
class ModelConfig:
    model_name: str = "baseline_mlp"
    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.10


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_name: str = "cosine"
    scheduler_step_size: int = 4
    scheduler_gamma: float = 0.5
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str = "models/checkpoints"
    log_path: str = "models/logs/training_log.json"
    metrics_path: str = "models/logs/metrics.json"


@dataclass(slots=True)
class BacktestConfig:
    long_threshold: float = 0.55
    short_threshold: float = 0.45
    periods_per_year: int = 252 * 78
    split_mode: str = "per_ticker"
    include_costs: bool = True
    cost_bps_per_trade: float = 0.0
    slippage_bps: float = 0.0


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)


def get_default_config(name: str = "baseline") -> ExperimentConfig:
    """Return a mutable default experiment configuration."""
    return ExperimentConfig(name=name)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize config dataclasses into plain dictionaries."""
    return asdict(config)


def _update_dataclass(instance: Any, overrides: dict[str, Any]) -> None:
    for key, value in overrides.items():
        if not hasattr(instance, key):
            continue
        current_value = getattr(instance, key)
        if is_dataclass(current_value) and isinstance(value, dict):
            _update_dataclass(current_value, value)
        else:
            setattr(instance, key, value)


def apply_overrides(config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Apply nested dictionary overrides to an ExperimentConfig."""
    _update_dataclass(config, overrides)
    return config


def guess_raw_root(config: ExperimentConfig) -> Path:
    """Resolve raw root across the two supported layouts."""
    preferred = Path(config.data.raw_root)
    if preferred.exists():
        return preferred

    fallback = preferred / "us"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Raw data directory not found at {preferred} or {fallback}")

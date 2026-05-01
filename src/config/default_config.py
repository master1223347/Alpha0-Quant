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
    source_timezone: str = "UTC"
    market_timezone: str = "America/New_York"
    max_tickers: int | None = None
    min_sequence_length: int = 40
    min_required_rows: int = 0
    min_required_train_rows: int = 0
    min_required_tickers: int = 0
    min_required_history_days: int = 0
    corporate_actions_path: str | None = None


@dataclass(slots=True)
class UniverseConfig:
    exchange: str | None = None
    asset_type: str | None = None
    tickers: tuple[str, ...] | None = None
    max_tickers: int | None = None
    min_sequence_length: int | None = None
    membership_path: str | None = None
    include_delisted: bool = True


@dataclass(slots=True)
class TargetConfig:
    horizon: int | None = None
    threshold: float = 0.001
    volatility_window: int = 20
    zscore_window: int = 20
    volatility_label_k: float = 0.25
    regression_clip: float = 3.0
    primary_target: str = "vol_direction_label"
    return_target: str = "vol_target_clipped"
    threshold_target: str = "vol_direction_label"
    direction_target: str = "label"


@dataclass(slots=True)
class FeatureConfig:
    momentum_lookback: int = 3
    volatility_window: int = 5
    relative_volume_window: int = 5
    volume_window: int = 10
    normalize: bool = True
    use_cross_sectional: bool = True
    use_realized_volatility: bool = True
    realized_vol_window: int = 20
    use_factor_features: bool = True
    factor_window: int = 78
    use_cointegration_features: bool = True
    cointegration_window: int = 78
    cointegration_min_samples: int = 40
    cointegration_half_life_clip: float = 200.0


@dataclass(slots=True)
class DatasetConfig:
    window_size: int = 32
    stride: int = 1
    label_horizon: int = 1
    split_mode: str = "global_time"
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
    minn_enabled: bool = False
    multitask_output: bool = True
    probabilistic_output: bool = True
    include_rank_head: bool = False
    include_regime_head: bool = False
    regime_classes: int = 3
    distribution: str = "gaussian"


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
    direction_loss_weight: float = 0.0
    threshold_loss_weight: float = 0.8
    regression_loss_weight: float = 0.2
    rank_loss_weight: float = 0.0
    regime_loss_weight: float = 0.0
    regression_loss: str = "huber"
    regression_huber_delta: float = 1.0
    student_t_df: float = 3.0
    return_rank_weight: float = 0.0
    score_alignment_weight: float = 0.0
    score_alignment_floor: float = 0.10
    volatility_consistency_weight: float = 0.0
    volatility_consistency_limit: float = 2.5
    temporal_smoothness_weight: float = 0.0
    temporal_smoothness_max_gap_seconds: int = 3600
    cross_sectional_reg_weight: float = 0.0
    cross_sectional_reg_limit: float = 2.5
    calibration_aux_weight: float = 0.0
    min_auc_sample_count: int = 200


@dataclass(slots=True)
class BacktestConfig:
    long_threshold: float = 0.60
    short_threshold: float = 0.40
    periods_per_year: int = 252 * 78
    execution_lag_bars: int = 1
    confidence_threshold: float = 0.60
    top_percentile: float | None = None
    confidence_threshold_sweep: tuple[float, ...] = (0.55, 0.60, 0.65, 0.70)
    confidence_top_percent_sweep: tuple[float, ...] = (0.05, 0.10, 0.20)
    selection_mode: str = "global_abs"
    score_source: str = "expected_utility"
    long_short_percentile: float | None = None
    split_mode: str = "global_time"
    flip_positions: bool = False
    include_costs: bool = True
    cost_bps_per_trade: float = 0.0
    slippage_bps: float = 0.0
    signal_source: str = "classification_prob"
    signal_mu_sigma_floor: float = 0.05
    require_directional_agreement: bool = False
    confidence_mu_agreement_weight: float = 0.50
    enable_regime_adaptation: bool = False
    regime_states: int = 3
    regime_feature_window: int = 32
    regime_random_state: int = 7
    trending_policy: str = "follow"
    mean_reverting_policy: str = "flip"
    volatile_policy: str = "flat"
    volatile_confidence_threshold: float = 0.70


@dataclass(slots=True)
class EvaluationConfig:
    walk_forward_enabled: bool = False
    walk_forward_train_days: int = 252
    walk_forward_val_days: int = 63
    walk_forward_test_days: int = 21
    walk_forward_step_days: int = 21
    walk_forward_embargo_bars: int = 0
    use_temperature_scaling: bool = False
    calibration_bins: int = 10
    run_selection_bias_tests: bool = False
    reality_check_bootstrap: int = 500
    spa_bootstrap: int = 500
    fdr_alpha: float = 0.10
    candidate_metric_keys: tuple[str, ...] = ("metrics.auc", "backtest.sharpe", "backtest.pnl")


@dataclass(slots=True)
class DeploymentConfig:
    export_format: str = "none"
    export_path: str = "models/exports/model_export.pt2"
    allow_deprecated_torchscript: bool = False


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
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)


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

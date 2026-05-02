"""True walk-forward retraining pipeline.

Distinct from the post-hoc sliced evaluation in :mod:`evaluate_model`. This
runner retrains a fresh model on every fold, with frozen hyperparameters,
refit normalization, and per-fold evaluation. Features are built and labels
attached once globally (both are causal and PIT-safe), then sliced by trading
date into rolling train/val/test windows.

Phase 3 of the design report. Deferred to later phases:
  * per-fold calibration recalibration window
  * adaptive event-threshold ``k`` per fold
  * per-fold full backtest, stacked ensembles, deep ensembles
  * regime-wise aggregation
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from src.config.default_config import ExperimentConfig, get_default_config
from src.dataset.dataloader import create_dataloaders
from src.dataset.sampler import split_ticker_sequences
from src.pipeline.build_dataset import (
    assemble_split_datasets,
    normalize_split_sequences,
)
from src.pipeline.build_features import build_feature_store
from src.pipeline.train_model import build_model
from src.targets.labeling import assign_cross_sectional_rank, label_ticker_sequences
from src.training.train import train_model
from src.training.validate import validate_epoch
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class WalkForwardFoldReport:
    fold_index: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str
    train_rows: int
    val_rows: int
    test_rows: int
    best_epoch: int
    best_val_score: float
    test_loss: float
    test_metrics: dict[str, Any]
    test_directional_sample_count: int
    test_total_sample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start,
            "val_end": self.val_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_rows": self.train_rows,
            "val_rows": self.val_rows,
            "test_rows": self.test_rows,
            "best_epoch": self.best_epoch,
            "best_val_score": self.best_val_score,
            "test_loss": self.test_loss,
            "test_metrics": self.test_metrics,
            "test_directional_sample_count": self.test_directional_sample_count,
            "test_total_sample_count": self.test_total_sample_count,
        }


@dataclass(slots=True)
class WalkForwardArtifacts:
    folds: list[WalkForwardFoldReport] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "folds": [fold.to_dict() for fold in self.folds],
            "summary": self.summary,
            "output_path": self.output_path,
        }


# ----- fold construction -----------------------------------------------------


def _unique_session_dates(
    labeled_sequences: dict[str, list[list[dict[str, Any]]]],
) -> list[date]:
    dates: set[date] = set()
    for sequences in labeled_sequences.values():
        for sequence in sequences:
            for row in sequence:
                ts = row.get("timestamp")
                if isinstance(ts, datetime):
                    dates.add(ts.date())
    return sorted(dates)


def _build_retrain_folds(
    unique_dates: list[date],
    *,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
    max_folds: int | None,
) -> list[tuple[date, date, date, date, date, date]]:
    """Return list of (train_start, train_end, val_start, val_end, test_start, test_end) date tuples."""
    if not unique_dates:
        return []
    train_days = max(1, int(train_days))
    val_days = max(0, int(val_days))
    test_days = max(1, int(test_days))
    step_days = max(1, int(step_days))

    folds: list[tuple[date, date, date, date, date, date]] = []
    n = len(unique_dates)
    cursor = 0
    while True:
        train_start_idx = cursor
        train_end_idx = train_start_idx + train_days - 1
        val_start_idx = train_end_idx + 1
        val_end_idx = val_start_idx + val_days - 1 if val_days > 0 else train_end_idx
        test_start_idx = val_end_idx + 1
        test_end_idx = test_start_idx + test_days - 1
        if test_end_idx >= n:
            break
        folds.append(
            (
                unique_dates[train_start_idx],
                unique_dates[train_end_idx],
                unique_dates[val_start_idx] if val_days > 0 else unique_dates[train_end_idx],
                unique_dates[val_end_idx] if val_days > 0 else unique_dates[train_end_idx],
                unique_dates[test_start_idx],
                unique_dates[test_end_idx],
            )
        )
        cursor += step_days
        if max_folds is not None and len(folds) >= int(max_folds):
            break
    return folds


# ----- slicing ---------------------------------------------------------------


def _slice_labeled_sequences_by_date(
    labeled_sequences: dict[str, list[list[dict[str, Any]]]],
    *,
    start_date: date,
    end_date: date,
    drop_last_bars: int = 0,
) -> dict[str, list[list[dict[str, Any]]]]:
    """Slice ticker sequences to rows in [start_date, end_date]; deep-copies rows.

    Optionally drops the last ``drop_last_bars`` rows from each ticker's
    in-window slice (used to apply an embargo at the trailing edge of the
    train/val window before the test window starts).
    """
    out: dict[str, list[list[dict[str, Any]]]] = {}
    for ticker, sequences in labeled_sequences.items():
        kept_sequences: list[list[dict[str, Any]]] = []
        all_rows_in_window: list[dict[str, Any]] = []
        for sequence in sequences:
            slice_rows: list[dict[str, Any]] = []
            for row in sequence:
                ts = row.get("timestamp")
                if not isinstance(ts, datetime):
                    continue
                day = ts.date()
                if start_date <= day <= end_date:
                    slice_rows.append(copy.deepcopy(row))
            if slice_rows:
                kept_sequences.append(slice_rows)
                all_rows_in_window.extend(slice_rows)
        if drop_last_bars > 0 and all_rows_in_window:
            all_rows_in_window.sort(key=lambda r: r["timestamp"])
            cutoff_index = max(0, len(all_rows_in_window) - int(drop_last_bars))
            keep_set = {id(r) for r in all_rows_in_window[:cutoff_index]}
            kept_sequences = [
                [r for r in seq if id(r) in keep_set] for seq in kept_sequences
            ]
            kept_sequences = [seq for seq in kept_sequences if seq]
        if kept_sequences:
            out[ticker] = kept_sequences
    return out


# ----- fold runner -----------------------------------------------------------


def _row_count(split_sequences: dict[str, list[list[dict[str, Any]]]]) -> int:
    return sum(
        len(seq) for sequences in split_sequences.values() for seq in sequences
    )


def _run_single_fold(
    *,
    fold_index: int,
    config: ExperimentConfig,
    labeled_sequences: dict[str, list[list[dict[str, Any]]]],
    feature_columns: list[str],
    train_start: date,
    train_end: date,
    val_start: date,
    val_end: date,
    test_start: date,
    test_end: date,
    embargo_bars: int,
    fold_output_dir: Path,
) -> WalkForwardFoldReport | None:
    LOGGER.info(
        "Fold %d: train %s..%s | val %s..%s | test %s..%s | embargo=%d bars",
        fold_index,
        train_start,
        train_end,
        val_start,
        val_end,
        test_start,
        test_end,
        embargo_bars,
    )
    train_seqs = _slice_labeled_sequences_by_date(
        labeled_sequences,
        start_date=train_start,
        end_date=train_end,
        drop_last_bars=embargo_bars,
    )
    val_seqs = _slice_labeled_sequences_by_date(
        labeled_sequences,
        start_date=val_start,
        end_date=val_end,
        drop_last_bars=embargo_bars,
    )
    test_seqs = _slice_labeled_sequences_by_date(
        labeled_sequences,
        start_date=test_start,
        end_date=test_end,
        drop_last_bars=0,
    )
    if not train_seqs or not test_seqs:
        LOGGER.warning("Fold %d skipped: empty train (%d) or test (%d)", fold_index, len(train_seqs), len(test_seqs))
        return None

    split_sequences = {"train": train_seqs, "val": val_seqs, "test": test_seqs}
    for split_name, ticker_map in split_sequences.items():
        if ticker_map:
            assign_cross_sectional_rank(ticker_map, target_column="next_log_return")

    if config.features.normalize:
        split_sequences, _ = normalize_split_sequences(split_sequences, feature_columns)

    datasets, split_row_counts, resolved_window_size = assemble_split_datasets(
        config, split_sequences, feature_columns
    )

    train_rows = split_row_counts.get("train", 0)
    val_rows = split_row_counts.get("val", 0)
    test_rows = split_row_counts.get("test", 0)
    if train_rows == 0 or test_rows == 0:
        LOGGER.warning(
            "Fold %d skipped after dataset assembly: train_rows=%d test_rows=%d",
            fold_index, train_rows, test_rows,
        )
        return None

    dataloaders = create_dataloaders(
        datasets,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        shuffle_train=True,
    )

    fold_config = copy.deepcopy(config)
    fold_config.training.checkpoint_dir = str(fold_output_dir / f"fold_{fold_index:03d}_checkpoints")
    fold_config.training.log_path = str(fold_output_dir / f"fold_{fold_index:03d}_log.json")
    fold_config.training.metrics_path = str(fold_output_dir / f"fold_{fold_index:03d}_metrics.json")

    model = build_model(
        fold_config,
        num_features=len(feature_columns),
        window_size=resolved_window_size,
    )
    training_artifacts = train_model(fold_config, dataloaders, model)

    test_loader = dataloaders.get("test")
    test_loss = float("nan")
    test_metrics_dict: dict[str, Any] = {}
    test_dir_count = 0
    test_total = 0
    if test_loader is not None and len(test_loader.dataset) > 0:
        from src.models.losses import compute_pos_weight
        from src.models.losses_prob import build_model_loss
        from src.training.train import _extract_binary_labels_for_pos_weight, _resolve_device

        device = _resolve_device(str(fold_config.training.device))
        labels_for_weight = _extract_binary_labels_for_pos_weight(dataloaders["train"].dataset)
        pos_weight = compute_pos_weight(labels_for_weight) if labels_for_weight else None
        loss_fn = build_model_loss(
            model,
            pos_weight=pos_weight,
            distribution=str(getattr(fold_config.model, "distribution", "gaussian")),
            direction_weight=float(getattr(fold_config.training, "direction_loss_weight", 1.0)),
            threshold_weight=float(getattr(fold_config.training, "threshold_loss_weight", 0.25)),
            regression_weight=float(getattr(fold_config.training, "regression_loss_weight", 1.0)),
            rank_weight=float(getattr(fold_config.training, "rank_loss_weight", 0.10)),
            return_rank_weight=float(getattr(fold_config.training, "return_rank_weight", 0.0)),
            regime_weight=float(getattr(fold_config.training, "regime_loss_weight", 0.10)),
            student_t_df=float(getattr(fold_config.training, "student_t_df", 3.0)),
            regression_loss=str(getattr(fold_config.training, "regression_loss", "huber")),
            regression_huber_delta=float(getattr(fold_config.training, "regression_huber_delta", 1.0)),
            score_alignment_weight=float(getattr(fold_config.training, "score_alignment_weight", 0.0)),
            score_alignment_floor=float(getattr(fold_config.training, "score_alignment_floor", 0.10)),
            volatility_consistency_weight=float(getattr(fold_config.training, "volatility_consistency_weight", 0.0)),
            volatility_consistency_limit=float(getattr(fold_config.training, "volatility_consistency_limit", 2.5)),
            temporal_smoothness_weight=float(getattr(fold_config.training, "temporal_smoothness_weight", 0.0)),
            temporal_smoothness_max_gap_seconds=int(getattr(fold_config.training, "temporal_smoothness_max_gap_seconds", 3600)),
            cross_sectional_reg_weight=float(getattr(fold_config.training, "cross_sectional_reg_weight", 0.0)),
            cross_sectional_reg_limit=float(getattr(fold_config.training, "cross_sectional_reg_limit", 2.5)),
            calibration_aux_weight=float(getattr(fold_config.training, "calibration_aux_weight", 0.0)),
            event_weight=float(getattr(fold_config.training, "event_loss_weight", 0.0)),
            event_direction_weight=float(getattr(fold_config.training, "event_direction_loss_weight", 0.0)),
            event_focal_gamma=float(getattr(fold_config.training, "event_focal_gamma", 2.0)),
            event_sample_weight_cap=float(getattr(fold_config.training, "event_sample_weight_cap", 2.0)),
        ).to(device)

        validation_result = validate_epoch(model, test_loader, loss_fn, device=device)
        test_loss = float(validation_result.loss)
        test_metrics_dict = validation_result.metrics.to_dict()
        if validation_result.event_metrics is not None:
            test_metrics_dict["event_head"] = validation_result.event_metrics.to_dict()
        if validation_result.event_direction_metrics is not None:
            test_metrics_dict["conditional_direction_head"] = validation_result.event_direction_metrics.to_dict()
        test_dir_count = int(getattr(validation_result, "directional_sample_count", 0))
        test_total = int(getattr(validation_result, "total_sample_count", 0))

    return WalkForwardFoldReport(
        fold_index=fold_index,
        train_start=train_start.isoformat(),
        train_end=train_end.isoformat(),
        val_start=val_start.isoformat(),
        val_end=val_end.isoformat(),
        test_start=test_start.isoformat(),
        test_end=test_end.isoformat(),
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        best_epoch=int(training_artifacts.best_epoch),
        best_val_score=float(training_artifacts.best_score),
        test_loss=test_loss,
        test_metrics=test_metrics_dict,
        test_directional_sample_count=test_dir_count,
        test_total_sample_count=test_total,
    )


# ----- public entry point ----------------------------------------------------


def run_walk_forward_retrain(
    config: ExperimentConfig | None = None,
    *,
    exchange: str | None = None,
    asset_type: str | None = None,
    output_path: str | Path | None = None,
) -> WalkForwardArtifacts:
    """Run end-to-end walk-forward retraining and persist a summary."""
    config = config or get_default_config()
    retrain_cfg = getattr(config, "walk_forward_retrain", None)
    if retrain_cfg is None or not bool(getattr(retrain_cfg, "enabled", False)):
        raise ValueError(
            "walk_forward_retrain.enabled must be True to run the retrain pipeline"
        )

    feature_artifacts = build_feature_store(config, exchange=exchange, asset_type=asset_type)
    if not feature_artifacts.feature_columns:
        raise ValueError(
            "No usable feature rows were built. Increase data coverage or relax sequence constraints."
        )

    event_target_cfg = getattr(config, "event_target", None)
    if event_target_cfg is not None and bool(getattr(event_target_cfg, "enabled", False)):
        event_horizon = getattr(event_target_cfg, "event_horizon_bars", None)
    else:
        event_horizon = None
    target_horizon = int(
        event_horizon
        if event_horizon is not None
        else (config.targets.horizon if config.targets.horizon is not None else config.dataset.label_horizon)
    )
    event_k = (
        float(getattr(event_target_cfg, "event_k", getattr(event_target_cfg, "k", 1.0)))
        if event_target_cfg is not None
        else 1.0
    )
    event_vol_window = (
        int(getattr(event_target_cfg, "event_vol_lookback_bars", getattr(event_target_cfg, "vol_window", 78)))
        if event_target_cfg is not None
        else 78
    )
    labeled_sequences = label_ticker_sequences(
        feature_artifacts.ticker_sequences,
        horizon=target_horizon,
        threshold=float(config.targets.threshold),
        volatility_window=int(config.targets.volatility_window),
        zscore_window=int(config.targets.zscore_window),
        volatility_label_k=float(getattr(config.targets, "volatility_label_k", 0.25)),
        regression_clip=float(getattr(config.targets, "regression_clip", 3.0)),
        event_k=event_k,
        event_vol_window=event_vol_window,
    )

    unique_dates = _unique_session_dates(labeled_sequences)
    if not unique_dates:
        raise ValueError("No session dates found in labeled sequences")

    folds_spec = _build_retrain_folds(
        unique_dates,
        train_days=int(retrain_cfg.train_days),
        val_days=int(retrain_cfg.val_days),
        test_days=int(retrain_cfg.test_days),
        step_days=int(retrain_cfg.step_days),
        max_folds=retrain_cfg.max_folds,
    )
    if not folds_spec:
        raise ValueError(
            f"Not enough history for walk-forward retrain: have {len(unique_dates)} session dates, "
            f"need at least train_days+val_days+test_days = {retrain_cfg.train_days + retrain_cfg.val_days + retrain_cfg.test_days}"
        )
    LOGGER.info("Walk-forward retrain: %d folds over %d session dates", len(folds_spec), len(unique_dates))

    fold_output_dir = Path(config.training.checkpoint_dir).parent / "walk_forward_retrain"
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    fold_reports: list[WalkForwardFoldReport] = []
    for i, (train_start, train_end, val_start, val_end, test_start, test_end) in enumerate(folds_spec):
        report = _run_single_fold(
            fold_index=i,
            config=config,
            labeled_sequences=labeled_sequences,
            feature_columns=feature_artifacts.feature_columns,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            embargo_bars=int(retrain_cfg.embargo_bars),
            fold_output_dir=fold_output_dir,
        )
        if report is not None:
            fold_reports.append(report)

    summary = _aggregate_fold_reports(fold_reports)
    artifacts = WalkForwardArtifacts(
        folds=fold_reports,
        summary=summary,
        output_path="",
    )

    if output_path is None:
        output_path = fold_output_dir / "walk_forward_retrain_summary.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(artifacts.to_dict(), handle, indent=2, default=str)
    artifacts.output_path = str(output_path)
    LOGGER.info("Walk-forward retrain summary written to %s", output_path)
    return artifacts


def _aggregate_fold_reports(folds: list[WalkForwardFoldReport]) -> dict[str, Any]:
    if not folds:
        return {"fold_count": 0}
    aucs = [
        f.test_metrics.get("auc")
        for f in folds
        if isinstance(f.test_metrics.get("auc"), (int, float))
        and not _is_nan(f.test_metrics.get("auc"))
    ]
    accs = [f.test_metrics.get("accuracy") for f in folds if isinstance(f.test_metrics.get("accuracy"), (int, float))]
    test_rows = [f.test_rows for f in folds]

    return {
        "fold_count": len(folds),
        "test_total_rows": sum(test_rows),
        "test_mean_rows": _mean(test_rows),
        "test_auc_mean": _mean(aucs) if aucs else None,
        "test_auc_std": _std(aucs) if aucs else None,
        "test_accuracy_mean": _mean(accs) if accs else None,
        "test_accuracy_std": _std(accs) if accs else None,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    m = _mean(values)
    return (sum((v - m) ** 2 for v in values) / len(values)) ** 0.5


def _is_nan(value: Any) -> bool:
    try:
        return value != value  # NaN is not equal to itself
    except Exception:
        return False


# ----- CLI -------------------------------------------------------------------


def _parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run Alpha0 walk-forward retraining")
    parser.add_argument("--config", required=True, help="Path to experiment YAML file")
    parser.add_argument("--output", default=None, help="Optional override path for the summary JSON")
    return parser.parse_args()


def main() -> None:
    from src.pipeline.run_experiment import load_experiment_config

    args = _parse_args()
    config = load_experiment_config(args.config)
    if not config.walk_forward_retrain.enabled:
        LOGGER.warning("walk_forward_retrain.enabled is False in config; forcing True for CLI run")
        config.walk_forward_retrain.enabled = True
    run_walk_forward_retrain(config, output_path=args.output)


if __name__ == "__main__":
    main()

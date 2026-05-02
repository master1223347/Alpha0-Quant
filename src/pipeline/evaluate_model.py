"""Evaluation pipeline entrypoint."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.default_config import ExperimentConfig, get_default_config
from src.dataset.dataloader import create_dataloaders
from src.evaluation.analysis import EvaluationReport, build_evaluation_report, save_evaluation_report
from src.evaluation.backtest import run_backtest
from src.evaluation.calibration import fit_temperature_scaling
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.uncertainty import estimate_mc_dropout_uncertainty
from src.models.losses import compute_pos_weight
from src.models.losses_prob import build_model_loss
from src.evaluation.stat_tests import benjamini_hochberg, deflated_sharpe_ratio, hansen_spa_test, white_reality_check
from src.training.checkpoint import load_checkpoint
from src.training.validate import validate_epoch
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class EvaluatePipelineArtifacts:
    report: EvaluationReport
    report_path: str


def _float_sequence(values: Any, *, fallback: tuple[float, ...]) -> list[float]:
    if values is None:
        return [float(value) for value in fallback]
    if isinstance(values, (list, tuple)):
        return [float(value) for value in values]
    return [float(values)]


def _build_regime_backtest_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "enable_regime_adaptation": bool(getattr(config.backtest, "enable_regime_adaptation", False)),
        "regime_states": int(getattr(config.backtest, "regime_states", 3)),
        "regime_feature_window": int(getattr(config.backtest, "regime_feature_window", 32)),
        "regime_random_state": int(getattr(config.backtest, "regime_random_state", 7)),
        "trending_policy": str(getattr(config.backtest, "trending_policy", "follow")),
        "mean_reverting_policy": str(getattr(config.backtest, "mean_reverting_policy", "flip")),
        "volatile_policy": str(getattr(config.backtest, "volatile_policy", "flat")),
        "volatile_confidence_threshold": float(getattr(config.backtest, "volatile_confidence_threshold", 0.70)),
    }


def _build_execution_model_kwargs(config: ExperimentConfig) -> dict[str, Any]:
    execution_cfg = getattr(config, "execution_model", None)
    if execution_cfg is None:
        return {}
    return {
        "execution_model_enabled": bool(getattr(execution_cfg, "enabled", False)),
        "use_open_auction": bool(getattr(execution_cfg, "use_open_auction", True)),
        "use_close_auction": bool(getattr(execution_cfg, "use_close_auction", True)),
        "regular_max_pov": float(getattr(execution_cfg, "regular_max_pov", 0.05)),
        "open_max_pov": float(getattr(execution_cfg, "open_max_pov", 0.03)),
        "close_max_pov": float(getattr(execution_cfg, "close_max_pov", 0.05)),
        "open_penalty_bars": int(getattr(execution_cfg, "open_penalty_bars", 3)),
        "close_penalty_bars": int(getattr(execution_cfg, "close_penalty_bars", 3)),
        "base_spread_bps": float(getattr(execution_cfg, "base_spread_bps", 6.0)),
        "base_impact_bps": float(getattr(execution_cfg, "base_impact_bps", 25.0)),
        "order_notional_fraction": float(getattr(execution_cfg, "order_notional_fraction", 0.01)),
        "reject_excess_pov": bool(getattr(execution_cfg, "reject_excess_pov", False)),
    }


def _safe_probability(up_probability: float, down_probability: float) -> float:
    denominator = float(up_probability + down_probability)
    if denominator <= 1e-8:
        return 0.5
    return float(up_probability / denominator)


def _precision_at_k(*, labels: list[int], scores: list[float], top_fraction: float) -> float:
    if not labels or not scores or len(labels) != len(scores):
        return float("nan")
    k = max(1, int(len(scores) * float(top_fraction)))
    ranked = sorted(range(len(scores)), key=lambda index: abs(scores[index]), reverse=True)
    selected = ranked[:k]
    positives = sum(1 for index in selected if labels[index] == 1)
    return positives / len(selected) if selected else float("nan")


def _build_confidence_bucket_summary(
    *,
    validation: Any,
    top_percentile: float,
    periods_per_year: int,
    execution_lag_bars: int,
    flip_positions: bool,
    cost_bps_per_trade: float,
    slippage_bps: float,
    regime_backtest_kwargs: dict[str, Any] | None = None,
    regime_fit_kwargs: dict[str, Any] | None = None,
    execution_model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src.evaluation.metrics import compute_classification_metrics

    if not validation.up_probabilities or not validation.down_probabilities:
        return {"top_percentile": float(top_percentile), "error": "missing directional probabilities"}

    sample_count = len(validation.up_probabilities)
    if sample_count == 0:
        return {"top_percentile": float(top_percentile), "error": "empty validation samples"}

    keep_count = max(1, int(sample_count * float(top_percentile)))
    if getattr(validation, "action_scores", None):
        score_values = [float(value) for value in validation.action_scores]
        ranked_indices = sorted(range(sample_count), key=lambda index: abs(score_values[index]), reverse=True)
    else:
        confidence_scores = [max(float(up), float(down)) for up, down in zip(validation.up_probabilities, validation.down_probabilities)]
        ranked_indices = sorted(range(sample_count), key=lambda index: confidence_scores[index], reverse=True)
    selected_indices = ranked_indices[:keep_count]

    y_true: list[int] = []
    y_prob: list[float] = []
    signed_realized_returns: list[float] = []
    non_neutral_count = 0
    for index in selected_indices:
        if index >= len(validation.three_class_labels):
            continue
        label = int(validation.three_class_labels[index])
        if label == 1:
            continue
        non_neutral_count += 1
        up_probability = float(validation.up_probabilities[index])
        down_probability = float(validation.down_probabilities[index])
        y_true.append(1 if label == 2 else 0)
        y_prob.append(_safe_probability(up_probability, down_probability))

        if getattr(validation, "action_scores", None):
            direction = 1.0 if float(validation.action_scores[index]) >= 0.0 else -1.0
        else:
            direction = 1.0 if up_probability >= down_probability else -1.0
        close_value = float(validation.close[index])
        next_close_value = float(validation.next_close[index])
        raw_return = (next_close_value - close_value) / close_value if close_value > 0 else 0.0
        signed_realized_returns.append(direction * raw_return)

    if y_true and y_prob and len(y_true) == len(y_prob):
        directional_metrics = compute_classification_metrics(y_true, y_prob)
        directional_accuracy = float(directional_metrics.accuracy)
        directional_precision = float(directional_metrics.precision)
    else:
        directional_accuracy = float("nan")
        directional_precision = float("nan")

    backtest = run_backtest(
        probabilities=validation.probabilities,
        up_probabilities=validation.up_probabilities,
        down_probabilities=validation.down_probabilities,
        close=validation.close,
        next_close=validation.next_close,
        timestamps=getattr(validation, "timestamps", None),
        tickers=getattr(validation, "tickers", None),
        confidence_threshold=0.0,
        top_percentile=float(top_percentile),
        selection_mode="global_abs",
        periods_per_year=periods_per_year,
        execution_lag_bars=int(execution_lag_bars),
        flip_positions=bool(flip_positions),
        cost_bps_per_trade=cost_bps_per_trade,
        slippage_bps=slippage_bps,
        signal_scores=getattr(validation, "action_scores", None),
        sigma_values=getattr(validation, "sigma_values", None),
        **(regime_fit_kwargs or {}),
        **(regime_backtest_kwargs or {}),
        **(execution_model_kwargs or {}),
    )

    average_realized_return = (
        sum(signed_realized_returns) / len(signed_realized_returns) if signed_realized_returns else float("nan")
    )
    return {
        "top_percentile": float(top_percentile),
        "selected_rows": int(len(selected_indices)),
        "non_neutral_rows": int(non_neutral_count),
        "directional_accuracy": directional_accuracy,
        "directional_precision": directional_precision,
        "mean_action_score": float(sum(float(validation.action_scores[index]) for index in selected_indices) / len(selected_indices))
        if getattr(validation, "action_scores", None)
        else float("nan"),
        "average_realized_return": float(average_realized_return),
        "hit_rate": float(backtest.hit_rate),
        "sharpe_after_costs": float(backtest.sharpe),
        "trade_count": int(backtest.trade_count),
        "pnl_after_costs": float(backtest.pnl),
    }


def _build_uncertainty_bucket_summary(
    *,
    validation: Any,
    quantiles: tuple[float, float] = (0.33, 0.66),
) -> dict[str, Any]:
    if not validation.sigma_values or not validation.forward_returns or not validation.mean_returns:
        return {"error": "missing sigma/return predictions for uncertainty bucket analysis"}
    if not (len(validation.sigma_values) == len(validation.forward_returns) == len(validation.mean_returns)):
        return {"error": "sigma/return arrays have incompatible lengths"}

    import statistics

    sigma = [float(value) for value in validation.sigma_values]
    sorted_sigma = sorted(sigma)
    if len(sorted_sigma) < 3:
        return {"error": "not enough samples for uncertainty buckets"}

    def _at_quantile(q: float) -> float:
        idx = min(len(sorted_sigma) - 1, max(0, int(q * (len(sorted_sigma) - 1))))
        return sorted_sigma[idx]

    low_cut = _at_quantile(float(quantiles[0]))
    high_cut = _at_quantile(float(quantiles[1]))
    buckets: dict[str, list[int]] = {"low_sigma": [], "mid_sigma": [], "high_sigma": []}
    for index, value in enumerate(sigma):
        if value <= low_cut:
            buckets["low_sigma"].append(index)
        elif value >= high_cut:
            buckets["high_sigma"].append(index)
        else:
            buckets["mid_sigma"].append(index)

    result: dict[str, Any] = {"low_cut": low_cut, "high_cut": high_cut}
    for name, indices in buckets.items():
        if not indices:
            result[name] = {"count": 0}
            continue
        errors = [
            abs(float(validation.forward_returns[index]) - float(validation.mean_returns[index]))
            for index in indices
        ]
        result[name] = {
            "count": len(indices),
            "mae": sum(errors) / len(errors),
            "median_sigma": statistics.median(sigma[index] for index in indices),
        }
    return result


def _is_binary_labels(values: list[float], *, tol: float = 1e-6) -> bool:
    if not values:
        return False
    for value in values:
        if abs(value - 0.0) <= tol or abs(value - 1.0) <= tol:
            continue
        return False
    return True


def _extract_binary_labels_for_pos_weight(dataset: Any) -> list[int] | None:
    for attribute in ("direction_label", "y", "labels", "target"):
        values = getattr(dataset, attribute, None)
        if values is None:
            continue
        if hasattr(values, "tolist"):
            raw_values = values.tolist()
        elif isinstance(values, (list, tuple)):
            raw_values = list(values)
        else:
            continue
        flattened = [float(value) for value in raw_values]
        if not _is_binary_labels(flattened):
            continue
        return [int(round(value)) for value in flattened]
    return None


def _to_flat_list(values: Any) -> list[Any]:
    if values is None:
        return []
    if hasattr(values, "detach"):
        return values.detach().cpu().reshape(-1).tolist()
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        try:
            return values.reshape(-1).tolist()
        except Exception:
            pass
    if hasattr(values, "tolist"):
        converted = values.tolist()
        if isinstance(converted, list):
            return converted
        return [converted]
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _equity_curve_to_returns(curve: list[float] | None) -> list[float]:
    if not curve or len(curve) < 2:
        return []
    output: list[float] = []
    for previous, current in zip(curve[:-1], curve[1:]):
        prev_value = float(previous)
        curr_value = float(current)
        if prev_value <= 0 or not math.isfinite(prev_value) or not math.isfinite(curr_value):
            output.append(0.0)
        else:
            output.append((curr_value / prev_value) - 1.0)
    return output


def _normal_positive_mean_pvalue(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / max(1, len(values) - 1)
    std_error = math.sqrt(max(variance, 1e-12)) / math.sqrt(len(values))
    if std_error <= 1e-12:
        return 1.0 if mean_value <= 0 else 0.0
    z = mean_value / std_error
    from statistics import NormalDist

    return 1.0 - NormalDist().cdf(z)


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            return None
    if hasattr(value, "timestamp"):
        try:
            return datetime.fromtimestamp(float(value.timestamp()))
        except Exception:
            return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except Exception:
            return None
    if isinstance(value, str):
        for parser in (datetime.fromisoformat,):
            try:
                return parser(value)
            except Exception:
                continue
        return None
    return None


def _build_walk_forward_windows(
    unique_days: list[date],
    *,
    train_days: int,
    val_days: int,
    test_days: int,
    step_days: int,
) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    if not unique_days:
        return windows
    train_days = max(1, int(train_days))
    val_days = max(0, int(val_days))
    test_days = max(1, int(test_days))
    step_days = max(1, int(step_days))

    index = 0
    while True:
        test_start_index = index + train_days + val_days
        test_end_index = test_start_index + test_days - 1
        if test_end_index >= len(unique_days):
            break
        windows.append((unique_days[test_start_index], unique_days[test_end_index]))
        index += step_days
    return windows


def _walk_forward_summary(
    *,
    config: ExperimentConfig,
    validation: Any,
    cost_bps_per_trade: float,
    slippage_bps: float,
    regime_fit_kwargs: dict[str, Any],
    regime_backtest_kwargs: dict[str, Any],
    execution_model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    timestamps = getattr(validation, "timestamps", None) or []
    parsed_timestamps = [_coerce_datetime(value) for value in timestamps]
    rows: list[dict[str, Any]] = []
    for index, timestamp in enumerate(parsed_timestamps):
        if timestamp is None:
            continue
        if index >= len(validation.probabilities):
            continue
        rows.append(
            {
                "index": index,
                "timestamp": timestamp,
                "date": timestamp.date(),
            }
        )
    if not rows:
        return {"enabled": True, "error": "missing or invalid timestamps for walk-forward"}

    rows.sort(key=lambda item: (item["timestamp"], item["index"]))
    unique_days = sorted({entry["date"] for entry in rows})
    windows = _build_walk_forward_windows(
        unique_days,
        train_days=int(getattr(config.evaluation, "walk_forward_train_days", 252)),
        val_days=int(getattr(config.evaluation, "walk_forward_val_days", 63)),
        test_days=int(getattr(config.evaluation, "walk_forward_test_days", 21)),
        step_days=int(getattr(config.evaluation, "walk_forward_step_days", 21)),
    )
    if not windows:
        return {"enabled": True, "error": "not enough history to form walk-forward windows"}

    embargo_bars = int(max(0, getattr(config.evaluation, "walk_forward_embargo_bars", 0)))
    window_reports: list[dict[str, Any]] = []
    for start_day, end_day in windows:
        selected = [entry for entry in rows if start_day <= entry["date"] <= end_day]
        if embargo_bars > 0 and len(selected) > embargo_bars:
            selected = selected[embargo_bars:]
        indices = [entry["index"] for entry in selected]
        if not indices:
            continue

        probabilities = [float(validation.probabilities[index]) for index in indices]
        labels = [int(validation.labels[index]) for index in indices] if len(validation.labels) > max(indices) else []
        close = [float(validation.close[index]) for index in indices]
        next_close = [float(validation.next_close[index]) for index in indices]
        up_probs = [float(validation.up_probabilities[index]) for index in indices] if len(validation.up_probabilities) > max(indices) else None
        down_probs = [float(validation.down_probabilities[index]) for index in indices] if len(validation.down_probabilities) > max(indices) else None
        window_timestamps = [validation.timestamps[index] for index in indices] if len(validation.timestamps) > max(indices) else None
        window_tickers = [validation.tickers[index] for index in indices] if len(validation.tickers) > max(indices) else None
        mu_values = [float(validation.mean_returns[index]) for index in indices] if len(validation.mean_returns) > max(indices) else None
        sigma_values = [float(validation.sigma_values[index]) for index in indices] if len(validation.sigma_values) > max(indices) else None
        action_scores = [float(validation.action_scores[index]) for index in indices] if len(validation.action_scores) > max(indices) else None

        backtest = run_backtest(
            probabilities=probabilities,
            up_probabilities=up_probs,
            down_probabilities=down_probs,
            close=close,
            next_close=next_close,
            timestamps=window_timestamps,
            tickers=window_tickers,
            long_threshold=config.backtest.long_threshold,
            short_threshold=config.backtest.short_threshold,
            confidence_threshold=getattr(config.backtest, "confidence_threshold", None),
            top_percentile=getattr(config.backtest, "top_percentile", None),
            selection_mode=str(getattr(config.backtest, "selection_mode", "global_abs")).lower().strip(),
            long_short_percentile=float(getattr(config.backtest, "long_short_percentile", None))
            if getattr(config.backtest, "long_short_percentile", None) is not None
            else None,
            periods_per_year=int(config.backtest.periods_per_year),
            execution_lag_bars=int(getattr(config.backtest, "execution_lag_bars", 1)),
            flip_positions=bool(getattr(config.backtest, "flip_positions", False)),
            cost_bps_per_trade=float(cost_bps_per_trade),
            slippage_bps=float(slippage_bps),
            signal_source=str(getattr(config.backtest, "signal_source", "classification_prob")),
            signal_scores=action_scores,
            mu_values=mu_values,
            sigma_values=sigma_values,
            require_directional_agreement=bool(getattr(config.backtest, "require_directional_agreement", False)),
            confidence_mu_agreement_weight=float(getattr(config.backtest, "confidence_mu_agreement_weight", 0.50)),
            signal_mu_sigma_floor=float(getattr(config.backtest, "signal_mu_sigma_floor", 0.05)),
            **regime_fit_kwargs,
            **regime_backtest_kwargs,
            **execution_model_kwargs,
        )

        if labels and len(labels) == len(probabilities):
            metrics = compute_classification_metrics(
                labels,
                probabilities,
                calibration_bins=int(getattr(config.evaluation, "calibration_bins", 10)),
            )
            metrics_dict = metrics.to_dict()
        else:
            metrics_dict = {"accuracy": float("nan"), "auc": float("nan"), "ece": float("nan"), "brier_score": float("nan")}

        window_reports.append(
            {
                "test_start_day": start_day.isoformat(),
                "test_end_day": end_day.isoformat(),
                "sample_count": len(indices),
                "metrics": metrics_dict,
                "backtest": backtest.to_dict(),
            }
        )

    if not window_reports:
        return {"enabled": True, "error": "no walk-forward windows with usable samples"}

    avg_sharpe = _mean([float(item["backtest"]["sharpe"]) for item in window_reports])
    avg_auc_values = [float(item["metrics"].get("auc", float("nan"))) for item in window_reports]
    finite_auc = [value for value in avg_auc_values if math.isfinite(value)]
    return {
        "enabled": True,
        "window_count": len(window_reports),
        "avg_sharpe": float(avg_sharpe),
        "avg_auc": float(_mean(finite_auc)) if finite_auc else float("nan"),
        "windows": window_reports,
    }


def _extract_regime_fit_kwargs(train_loader: Any | None) -> dict[str, Any]:
    if train_loader is None:
        return {}
    dataset = getattr(train_loader, "dataset", None)
    if dataset is None:
        return {}

    close_values = [float(value) for value in _to_flat_list(getattr(dataset, "close", None))]
    next_close_values = [float(value) for value in _to_flat_list(getattr(dataset, "next_close", None))]
    timestamp_values = _to_flat_list(getattr(dataset, "timestamps", None))
    ticker_values = [str(value) for value in _to_flat_list(getattr(dataset, "tickers", None))]

    output: dict[str, Any] = {}
    if close_values and len(close_values) == len(next_close_values):
        output["regime_fit_close"] = close_values
        output["regime_fit_next_close"] = next_close_values
        if timestamp_values and len(timestamp_values) == len(close_values):
            output["regime_fit_timestamps"] = timestamp_values
        if ticker_values and len(ticker_values) == len(close_values):
            output["regime_fit_tickers"] = ticker_values
    return output


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
    train_loader = dataloaders.get("train")
    fit_loader = train_loader or dataloaders.get("val")

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

    labels_for_weight = _extract_binary_labels_for_pos_weight(test_loader.dataset)
    pos_weight = compute_pos_weight(labels_for_weight) if labels_for_weight else None
    loss_fn = build_model_loss(
        model,
        pos_weight=pos_weight,
        distribution=str(getattr(config.model, "distribution", "gaussian")),
        direction_weight=float(getattr(config.training, "direction_loss_weight", 1.0)),
        threshold_weight=float(getattr(config.training, "threshold_loss_weight", 0.25)),
        regression_weight=float(getattr(config.training, "regression_loss_weight", 1.0)),
        rank_weight=float(getattr(config.training, "rank_loss_weight", 0.10)),
        return_rank_weight=float(getattr(config.training, "return_rank_weight", 0.0)),
        regime_weight=float(getattr(config.training, "regime_loss_weight", 0.10)),
        student_t_df=float(getattr(config.training, "student_t_df", 3.0)),
        regression_loss=str(getattr(config.training, "regression_loss", "nll")),
        regression_huber_delta=float(getattr(config.training, "regression_huber_delta", 1.0)),
        score_alignment_weight=float(getattr(config.training, "score_alignment_weight", 0.0)),
        score_alignment_floor=float(getattr(config.training, "score_alignment_floor", 0.10)),
        volatility_consistency_weight=float(getattr(config.training, "volatility_consistency_weight", 0.0)),
        volatility_consistency_limit=float(getattr(config.training, "volatility_consistency_limit", 2.5)),
        temporal_smoothness_weight=float(getattr(config.training, "temporal_smoothness_weight", 0.0)),
        temporal_smoothness_max_gap_seconds=int(getattr(config.training, "temporal_smoothness_max_gap_seconds", 3600)),
        cross_sectional_reg_weight=float(getattr(config.training, "cross_sectional_reg_weight", 0.0)),
        cross_sectional_reg_limit=float(getattr(config.training, "cross_sectional_reg_limit", 2.5)),
        calibration_aux_weight=float(getattr(config.training, "calibration_aux_weight", 0.0)),
        event_weight=float(getattr(config.training, "event_loss_weight", 0.0)),
        event_direction_weight=float(getattr(config.training, "event_direction_loss_weight", 0.0)),
        event_focal_gamma=float(getattr(config.training, "event_focal_gamma", 2.0)),
        event_sample_weight_cap=float(getattr(config.training, "event_sample_weight_cap", 2.0)),
    ).to(device)

    include_costs = bool(getattr(config.backtest, "include_costs", True))
    cost_bps_per_trade = float(getattr(config.backtest, "cost_bps_per_trade", 0.0)) if include_costs else 0.0
    slippage_bps = float(getattr(config.backtest, "slippage_bps", 0.0)) if include_costs else 0.0

    validation = validate_epoch(model, test_loader, loss_fn, device=device)
    regime_backtest_kwargs = _build_regime_backtest_kwargs(config)
    regime_fit_kwargs = _extract_regime_fit_kwargs(fit_loader)
    execution_model_kwargs = _build_execution_model_kwargs(config)
    selection_mode = str(getattr(config.backtest, "selection_mode", "global_abs")).lower().strip()
    score_source = str(getattr(config.backtest, "score_source", "expected_utility")).lower().strip()
    long_short_percentile = getattr(config.backtest, "long_short_percentile", None)
    if score_source == "expected_utility":
        signal_scores = getattr(validation, "action_scores", None)
    else:
        signal_scores = None
    backtest = run_backtest(
        probabilities=validation.probabilities,
        up_probabilities=validation.up_probabilities,
        down_probabilities=validation.down_probabilities,
        close=validation.close,
        next_close=validation.next_close,
        timestamps=validation.timestamps,
        tickers=validation.tickers,
        long_threshold=config.backtest.long_threshold,
        short_threshold=config.backtest.short_threshold,
        confidence_threshold=getattr(config.backtest, "confidence_threshold", None),
        top_percentile=getattr(config.backtest, "top_percentile", None),
        selection_mode=selection_mode,
        long_short_percentile=float(long_short_percentile) if long_short_percentile is not None else None,
        periods_per_year=config.backtest.periods_per_year,
        execution_lag_bars=int(getattr(config.backtest, "execution_lag_bars", 1)),
        flip_positions=bool(getattr(config.backtest, "flip_positions", False)),
        cost_bps_per_trade=cost_bps_per_trade,
        slippage_bps=slippage_bps,
        signal_source=str(getattr(config.backtest, "signal_source", "classification_prob")),
        signal_scores=signal_scores,
        mu_values=validation.mean_returns,
        sigma_values=validation.sigma_values,
        require_directional_agreement=bool(getattr(config.backtest, "require_directional_agreement", False)),
        confidence_mu_agreement_weight=float(getattr(config.backtest, "confidence_mu_agreement_weight", 0.50)),
        signal_mu_sigma_floor=float(getattr(config.backtest, "signal_mu_sigma_floor", 0.05)),
        **regime_fit_kwargs,
        **regime_backtest_kwargs,
        **execution_model_kwargs,
    )

    report = build_evaluation_report(
        model_name=config.model.model_name,
        split="test",
        metrics=validation.metrics,
        backtest=backtest,
    )
    if validation.event_metrics is not None:
        report.metrics["event_head"] = validation.event_metrics.to_dict()
    if validation.event_direction_metrics is not None:
        report.metrics["conditional_direction_head"] = validation.event_direction_metrics.to_dict()
    if validation.event_probabilities and validation.event_labels:
        prevalence = sum(int(value) for value in validation.event_labels) / max(1, len(validation.event_labels))
        report.metrics["event_target_summary"] = {
            "sample_count": len(validation.event_labels),
            "event_prevalence": float(prevalence),
            "mean_p_event": float(sum(float(value) for value in validation.event_probabilities) / len(validation.event_probabilities)),
        }
    uncertainty_cfg = getattr(config, "uncertainty", None)
    uncertainty_method = str(getattr(uncertainty_cfg, "method", "none")).lower().strip() if uncertainty_cfg is not None else "none"
    if uncertainty_method in {"mc_dropout", "dropout"}:
        uncertainty_report = estimate_mc_dropout_uncertainty(
            model,
            test_loader,
            device=device,
            samples=int(getattr(uncertainty_cfg, "mc_dropout_samples", 20)),
        )
        report.metrics["mc_dropout_uncertainty"] = uncertainty_report.to_dict()
    elif uncertainty_method in {"deep_ensemble", "ensemble"}:
        report.metrics["ensemble_uncertainty"] = {
            "configured_members": int(getattr(uncertainty_cfg, "ensemble_members", 1)),
            "status": "configure multiple seeded runs and stack out-of-fold predictions",
        }

    if getattr(validation, "action_scores", None):
        precision_labels = validation.labels
        precision_scores = validation.action_scores
        if validation.three_class_labels and len(validation.three_class_labels) == len(validation.action_scores):
            directional_indices = [index for index, label in enumerate(validation.three_class_labels) if label != 1]
            precision_labels = [1 if validation.three_class_labels[index] == 2 else 0 for index in directional_indices]
            precision_scores = [validation.action_scores[index] for index in directional_indices]
        report.metrics["precision_at_k"] = {
            "top_1%": _precision_at_k(labels=precision_labels, scores=precision_scores, top_fraction=0.01),
            "top_5%": _precision_at_k(labels=precision_labels, scores=precision_scores, top_fraction=0.05),
            "top_10%": _precision_at_k(labels=precision_labels, scores=precision_scores, top_fraction=0.10),
        }

    confidence_bucket_sweep = _float_sequence(
        getattr(config.backtest, "confidence_top_percent_sweep", None),
        fallback=(0.05, 0.10, 0.20),
    )
    report.metrics["confidence_buckets"] = [
        _build_confidence_bucket_summary(
            validation=validation,
            top_percentile=float(top_percentile),
            periods_per_year=int(config.backtest.periods_per_year),
            execution_lag_bars=int(getattr(config.backtest, "execution_lag_bars", 1)),
            flip_positions=bool(getattr(config.backtest, "flip_positions", False)),
            cost_bps_per_trade=cost_bps_per_trade,
            slippage_bps=slippage_bps,
            regime_backtest_kwargs=regime_backtest_kwargs,
            regime_fit_kwargs=regime_fit_kwargs,
            execution_model_kwargs=execution_model_kwargs,
        )
        for top_percentile in confidence_bucket_sweep
    ]
    report.metrics["uncertainty_buckets"] = _build_uncertainty_bucket_summary(validation=validation)
    if validation.confidence_scores and validation.forward_returns:
        high_conf_indices = [index for index, value in enumerate(validation.confidence_scores) if value >= 0.70]
        if high_conf_indices:
            high_conf_hit = sum(
                1
                for index in high_conf_indices
                if (validation.probabilities[index] >= 0.5) == (validation.forward_returns[index] >= 0.0)
            ) / len(high_conf_indices)
        else:
            high_conf_hit = float("nan")
        report.metrics["high_confidence_reliability"] = {
            "threshold": 0.70,
            "sample_count": len(high_conf_indices),
            "directional_hit_rate": float(high_conf_hit),
        }

    if bool(getattr(config.evaluation, "use_temperature_scaling", False)):
        calibration_labels = validation.labels
        calibration_probabilities = validation.probabilities
        if validation.three_class_labels and len(validation.three_class_labels) == len(validation.probabilities):
            directional_indices = [index for index, label in enumerate(validation.three_class_labels) if label != 1]
            calibration_labels = [1 if validation.three_class_labels[index] == 2 else 0 for index in directional_indices]
            calibration_probabilities = [validation.probabilities[index] for index in directional_indices]
        if calibration_labels and len(calibration_labels) == len(calibration_probabilities):
            calibration_result = fit_temperature_scaling(
                labels=[int(value) for value in calibration_labels],
                probabilities=[float(value) for value in calibration_probabilities],
            )
            report.metrics["temperature_scaling"] = {
                "temperature": float(calibration_result.temperature),
                "nll_before": float(calibration_result.nll_before),
                "nll_after": float(calibration_result.nll_after),
                "brier_before": float(calibration_result.brier_before),
                "brier_after": float(calibration_result.brier_after),
            }

    confidence_threshold_sweep = _float_sequence(
        getattr(config.backtest, "confidence_threshold_sweep", None),
        fallback=(0.55, 0.60, 0.65, 0.70),
    )
    top_percentile = getattr(config.backtest, "top_percentile", None)
    long_short_percentile = getattr(config.backtest, "long_short_percentile", None)
    selection_mode = str(getattr(config.backtest, "selection_mode", "global_abs")).lower().strip()
    score_source = str(getattr(config.backtest, "score_source", "expected_utility")).lower().strip()
    if score_source == "expected_utility":
        sweep_signal_scores = getattr(validation, "action_scores", None)
    else:
        sweep_signal_scores = None
    report.backtest["confidence_threshold_sweep"] = {
        f"{float(threshold):.2f}": run_backtest(
            probabilities=validation.probabilities,
            up_probabilities=validation.up_probabilities,
            down_probabilities=validation.down_probabilities,
            close=validation.close,
            next_close=validation.next_close,
            timestamps=validation.timestamps,
            tickers=validation.tickers,
            confidence_threshold=float(threshold),
            top_percentile=float(top_percentile) if top_percentile is not None else None,
            selection_mode=selection_mode,
            long_short_percentile=float(long_short_percentile) if long_short_percentile is not None else None,
            periods_per_year=int(config.backtest.periods_per_year),
            execution_lag_bars=int(getattr(config.backtest, "execution_lag_bars", 1)),
            flip_positions=bool(getattr(config.backtest, "flip_positions", False)),
            cost_bps_per_trade=cost_bps_per_trade,
            slippage_bps=slippage_bps,
            signal_source=str(getattr(config.backtest, "signal_source", "classification_prob")),
            signal_scores=sweep_signal_scores,
            mu_values=validation.mean_returns,
            sigma_values=validation.sigma_values,
            require_directional_agreement=bool(getattr(config.backtest, "require_directional_agreement", False)),
            confidence_mu_agreement_weight=float(getattr(config.backtest, "confidence_mu_agreement_weight", 0.50)),
            signal_mu_sigma_floor=float(getattr(config.backtest, "signal_mu_sigma_floor", 0.05)),
            **regime_fit_kwargs,
            **regime_backtest_kwargs,
            **execution_model_kwargs,
        ).to_dict()
        for threshold in confidence_threshold_sweep
    }

    if bool(getattr(config.evaluation, "run_selection_bias_tests", False)):
        strategy_curves: list[list[float]] = []
        trial_sharpes: list[float] = [float(backtest.sharpe)]
        primary_returns = _equity_curve_to_returns(backtest.equity_curve)
        if primary_returns:
            strategy_curves.append(primary_returns)
        for payload in report.backtest["confidence_threshold_sweep"].values():
            sharpe_value = payload.get("sharpe")
            if isinstance(sharpe_value, (float, int)):
                trial_sharpes.append(float(sharpe_value))
            curve = payload.get("equity_curve")
            if isinstance(curve, list):
                candidate_returns = _equity_curve_to_returns([float(value) for value in curve])
                if candidate_returns:
                    strategy_curves.append(candidate_returns)

        if primary_returns and trial_sharpes and strategy_curves:
            dsr = deflated_sharpe_ratio(strategy_returns=primary_returns, trial_sharpes=trial_sharpes)
            white = white_reality_check(
                strategy_returns=strategy_curves,
                bootstrap=int(getattr(config.evaluation, "reality_check_bootstrap", 500)),
            )
            spa = hansen_spa_test(
                strategy_returns=strategy_curves,
                bootstrap=int(getattr(config.evaluation, "spa_bootstrap", 500)),
            )
            p_values = [_normal_positive_mean_pvalue(values) for values in strategy_curves]
            p_values = [value if math.isfinite(value) else 1.0 for value in p_values]
            fdr = benjamini_hochberg(
                p_values,
                alpha=float(getattr(config.evaluation, "fdr_alpha", 0.10)),
            )
            report.metrics["selection_bias_controls"] = {
                "deflated_sharpe": {
                    "observed_sharpe": float(dsr.observed_sharpe),
                    "benchmark_max_sharpe": float(dsr.benchmark_max_sharpe),
                    "deflated_sharpe_z": float(dsr.deflated_sharpe_z),
                    "deflated_sharpe_pvalue": float(dsr.deflated_sharpe_pvalue),
                    "trial_count": int(dsr.trial_count),
                    "sample_length": int(dsr.sample_length),
                },
                "white_reality_check": white,
                "hansen_spa": spa,
                "candidate_p_values": p_values,
                "fdr": fdr,
            }

    if bool(getattr(config.evaluation, "walk_forward_enabled", False)):
        report.backtest["walk_forward"] = _walk_forward_summary(
            config=config,
            validation=validation,
            cost_bps_per_trade=cost_bps_per_trade,
            slippage_bps=slippage_bps,
            regime_fit_kwargs=regime_fit_kwargs,
            regime_backtest_kwargs=regime_backtest_kwargs,
            execution_model_kwargs=execution_model_kwargs,
        )

    report_path = Path(config.training.metrics_path).with_name("evaluation_report.json")
    save_evaluation_report(report, report_path)

    backtest_path = Path(config.training.metrics_path).with_name("backtest.json")
    with backtest_path.open("w", encoding="utf-8") as handle:
        json.dump(backtest.to_dict(), handle, indent=2)

    LOGGER.info("Evaluation report written to %s", report_path)
    return EvaluatePipelineArtifacts(report=report, report_path=str(report_path))

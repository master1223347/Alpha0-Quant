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
        confidence_threshold=0.0,
        top_percentile=float(top_percentile),
        selection_mode="global_abs",
        periods_per_year=periods_per_year,
        execution_lag_bars=int(execution_lag_bars),
        flip_positions=bool(flip_positions),
        cost_bps_per_trade=cost_bps_per_trade,
        slippage_bps=slippage_bps,
        signal_scores=getattr(validation, "action_scores", None),
        **(regime_backtest_kwargs or {}),
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
    ).to(device)

    include_costs = bool(getattr(config.backtest, "include_costs", True))
    cost_bps_per_trade = float(getattr(config.backtest, "cost_bps_per_trade", 0.0)) if include_costs else 0.0
    slippage_bps = float(getattr(config.backtest, "slippage_bps", 0.0)) if include_costs else 0.0

    validation = validate_epoch(model, test_loader, loss_fn, device=device)
    regime_backtest_kwargs = _build_regime_backtest_kwargs(config)
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
        **regime_backtest_kwargs,
    )

    report = build_evaluation_report(
        model_name=config.model.model_name,
        split="test",
        metrics=validation.metrics,
        backtest=backtest,
    )

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
            **regime_backtest_kwargs,
        ).to_dict()
        for threshold in confidence_threshold_sweep
    }

    report_path = Path(config.training.metrics_path).with_name("evaluation_report.json")
    save_evaluation_report(report, report_path)

    backtest_path = Path(config.training.metrics_path).with_name("backtest.json")
    with backtest_path.open("w", encoding="utf-8") as handle:
        json.dump(backtest.to_dict(), handle, indent=2)

    LOGGER.info("Evaluation report written to %s", report_path)
    return EvaluatePipelineArtifacts(report=report, report_path=str(report_path))

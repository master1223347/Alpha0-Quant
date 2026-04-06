"""Probabilistic multitask loss helpers for MINN-style models."""

from __future__ import annotations

import math
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:
    torch = None
    nn = None
    F = None


def _resolve_tensor(value: Any, *, device: Any, dtype: Any | None = None) -> Any | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        tensor = value.to(device=device)
        return tensor.to(dtype=dtype) if dtype is not None else tensor
    tensor = torch.as_tensor(value, device=device)
    return tensor.to(dtype=dtype) if dtype is not None else tensor


def _extract_from_dict(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _is_binary_tensor(values: Any, *, tol: float = 1e-6) -> bool:
    if values is None:
        return False
    tensor = values.reshape(-1)
    if tensor.numel() == 0:
        return False
    zeros = torch.abs(tensor - 0.0) <= tol
    ones = torch.abs(tensor - 1.0) <= tol
    return bool(torch.all(zeros | ones))


def extract_direction_logit(outputs: Any) -> Any:
    if isinstance(outputs, dict):
        value = _extract_from_dict(outputs, ("direction_logit", "direction_logits", "logit", "logits"))
        if value is None:
            raise KeyError("Model outputs do not contain a direction logit")
        return value
    return outputs


def extract_mean_return(outputs: Any) -> Any | None:
    if not isinstance(outputs, dict):
        return None
    return _extract_from_dict(outputs, ("mean_return", "return_mean", "mu"))


def extract_log_scale(outputs: Any) -> Any | None:
    if not isinstance(outputs, dict):
        return None
    return _extract_from_dict(outputs, ("log_scale", "scale_log", "log_sigma"))


def extract_threshold_logits(outputs: Any) -> Any | None:
    if not isinstance(outputs, dict):
        return None
    return _extract_from_dict(outputs, ("threshold_logits", "threshold_logit", "bucket_logits"))


def extract_rank_score(outputs: Any) -> Any | None:
    if not isinstance(outputs, dict):
        return None
    return _extract_from_dict(outputs, ("rank_score", "rank_logit", "ranking_score"))


def extract_regime_logits(outputs: Any) -> Any | None:
    if not isinstance(outputs, dict):
        return None
    return _extract_from_dict(outputs, ("regime_logits", "regime_logit", "regime"))


def extract_direction_target(targets: Any, batch: dict[str, Any] | None, *, device: Any) -> Any:
    if batch is None:
        target = _resolve_tensor(targets, device=device, dtype=torch.float32) if targets is not None else None
        return target if (target is not None and _is_binary_tensor(target)) else None

    explicit_target = _extract_from_dict(batch, ("direction_label", "binary_label", "label", "labels"))
    if explicit_target is not None:
        resolved = _resolve_tensor(explicit_target, device=device, dtype=torch.float32)
        if _is_binary_tensor(resolved):
            return resolved

    if targets is not None:
        resolved = _resolve_tensor(targets, device=device, dtype=torch.float32)
        if _is_binary_tensor(resolved):
            return resolved

    fallback_target = _extract_from_dict(batch, ("y", "target", "targets"))
    if fallback_target is None:
        return None
    resolved = _resolve_tensor(fallback_target, device=device, dtype=torch.float32)
    if _is_binary_tensor(resolved):
        return resolved
    return None


def extract_forward_return(batch: dict[str, Any] | None, *, device: Any) -> Any | None:
    if batch is None:
        return None
    for key in (
        "target_return",
        "forward_return",
        "next_log_return",
        "vol_target",
        "z_return",
        "return",
        "returns",
        "future_return",
        "y_return",
    ):
        if key in batch and batch[key] is not None:
            return _resolve_tensor(batch[key], device=device, dtype=torch.float32).reshape(-1)

    close = _extract_from_dict(batch, ("close", "price", "prices", "close_price"))
    next_close = _extract_from_dict(batch, ("next_close", "future_close", "next_price", "next_price_close"))
    if close is None or next_close is None:
        return None

    close_tensor = _resolve_tensor(close, device=device, dtype=torch.float32).reshape(-1)
    next_close_tensor = _resolve_tensor(next_close, device=device, dtype=torch.float32).reshape(-1)
    close_tensor = torch.clamp(close_tensor, min=1e-8)
    return (next_close_tensor - close_tensor) / close_tensor


def extract_regime_target(batch: dict[str, Any] | None, *, device: Any) -> Any | None:
    if batch is None:
        return None
    target = _extract_from_dict(batch, ("regime", "regime_label", "market_regime", "state", "class"))
    if target is None:
        return None
    return _resolve_tensor(target, device=device, dtype=torch.long).reshape(-1)


def extract_threshold_target(batch: dict[str, Any] | None, *, device: Any) -> Any | None:
    if batch is None:
        return None
    target = _extract_from_dict(batch, ("threshold_label", "threshold_target", "bucket_target"))
    if target is None:
        return None
    return _resolve_tensor(target, device=device, dtype=torch.long).reshape(-1)


def extract_rank_target(batch: dict[str, Any] | None, *, device: Any) -> Any | None:
    if batch is None:
        return None
    target = _extract_from_dict(batch, ("rank_target", "cross_sectional_rank", "rank_label", "ranking_target"))
    if target is None:
        return None
    return _resolve_tensor(target, device=device, dtype=torch.float32).reshape(-1)


def extract_timestamp_values(batch: dict[str, Any] | None) -> list[Any]:
    if batch is None:
        return []
    values = batch.get("timestamp")
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    return [values]


def extract_ticker_values(batch: dict[str, Any] | None) -> list[str]:
    if batch is None:
        return []
    values = batch.get("ticker")
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [str(value) for value in values]
    return [str(values)]


def _threshold_labels(forward_return: Any, *, neutral_band: float) -> Any:
    negative = forward_return < -neutral_band
    positive = forward_return > neutral_band
    neutral = ~(negative | positive)
    labels = torch.zeros_like(forward_return, dtype=torch.long)
    labels[neutral] = 1
    labels[positive] = 2
    return labels


def _pairwise_ranking_loss(prediction: Any, target: Any) -> Any:
    if prediction.numel() < 2:
        return prediction.new_tensor(0.0)
    diff_target = target.unsqueeze(1) - target.unsqueeze(0)
    diff_prediction = prediction.unsqueeze(1) - prediction.unsqueeze(0)
    positive_pairs = diff_target > 0
    if not torch.any(positive_pairs):
        return prediction.new_tensor(0.0)
    pairwise_scores = diff_prediction[positive_pairs]
    return F.softplus(-pairwise_scores).mean()


def gaussian_nll(target: Any, mean: Any, log_scale: Any, *, eps: float = 1e-6) -> Any:
    scale = F.softplus(log_scale) + eps
    standardized = (target - mean) / scale
    return 0.5 * standardized.pow(2) + torch.log(scale) + 0.5 * math.log(2.0 * math.pi)


def student_t_nll(target: Any, mean: Any, log_scale: Any, *, df: float = 3.0, eps: float = 1e-6) -> Any:
    scale = F.softplus(log_scale) + eps
    z = (target - mean) / scale
    df_tensor = torch.as_tensor(float(df), dtype=target.dtype, device=target.device)
    half_df = 0.5 * df_tensor
    constant = (
        torch.lgamma(half_df + 0.5)
        - torch.lgamma(half_df)
        + 0.5 * (torch.log(df_tensor) + math.log(math.pi))
        + torch.log(scale)
    )
    return constant + 0.5 * (df_tensor + 1.0) * torch.log1p(z.pow(2) / df_tensor)


def _safe_scale(log_scale: Any, *, eps: float = 1e-6) -> Any:
    return F.softplus(log_scale) + eps


def _volatility_consistency_penalty(mean_return: Any, log_scale: Any, *, limit: float) -> Any:
    if log_scale is None:
        scale = torch.ones_like(mean_return)
    else:
        scale = _safe_scale(log_scale)
    normalized_magnitude = torch.abs(mean_return) / scale
    return F.relu(normalized_magnitude - float(limit)).pow(2).mean()


def _temporal_smoothness_penalty(
    mean_return: Any,
    timestamps: list[Any],
    tickers: list[str],
    *,
    max_gap_seconds: int,
) -> Any:
    if mean_return.numel() < 2 or not timestamps or not tickers:
        return mean_return.new_tensor(0.0)
    if len(timestamps) != mean_return.numel() or len(tickers) != mean_return.numel():
        return mean_return.new_tensor(0.0)

    from datetime import datetime

    def _to_epoch_seconds(value: Any) -> float | None:
        if hasattr(value, "timestamp"):
            try:
                return float(value.timestamp())
            except Exception:
                return None
        if isinstance(value, datetime):
            return float(value.timestamp())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    by_ticker: dict[str, list[tuple[float, int]]] = {}
    for index, (timestamp, ticker) in enumerate(zip(timestamps, tickers)):
        epoch_seconds = _to_epoch_seconds(timestamp)
        if epoch_seconds is None:
            continue
        by_ticker.setdefault(ticker, []).append((epoch_seconds, index))

    penalties: list[Any] = []
    max_gap = float(max(0, int(max_gap_seconds)))
    for entries in by_ticker.values():
        if len(entries) < 2:
            continue
        entries.sort(key=lambda item: item[0])
        for (prev_ts, prev_index), (curr_ts, curr_index) in zip(entries[:-1], entries[1:]):
            if max_gap > 0 and (curr_ts - prev_ts) > max_gap:
                continue
            penalties.append((mean_return[curr_index] - mean_return[prev_index]).pow(2))
    if not penalties:
        return mean_return.new_tensor(0.0)
    return torch.stack(penalties).mean()


def _cross_sectional_extreme_penalty(mean_return: Any, *, limit: float) -> Any:
    centered = mean_return - mean_return.mean()
    return F.relu(torch.abs(centered) - float(limit)).pow(2).mean()


def _calibration_alignment_penalty(direction_logit: Any, mean_return: Any) -> Any:
    confidence = torch.sigmoid(torch.abs(direction_logit))
    agreement = torch.sigmoid(direction_logit * torch.sign(mean_return))
    return F.mse_loss(confidence, agreement)


if nn is not None:

    class DirectionOnlyLoss(nn.Module):
        """Wrapper around BCEWithLogitsLoss that tolerates dict outputs."""

        def __init__(self, *, pos_weight: float | None = None) -> None:
            super().__init__()
            if pos_weight is None:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                weight_tensor = torch.tensor(float(pos_weight), dtype=torch.float32)
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
            self.last_components: dict[str, float] = {}

        def forward(self, outputs: Any, targets: Any, batch: dict[str, Any] | None = None) -> Any:
            logits = extract_direction_logit(outputs)
            if not torch.is_tensor(logits):
                logits = torch.as_tensor(logits, dtype=torch.float32)
            logits = logits.reshape(-1)
            target_tensor = _resolve_tensor(targets, device=logits.device, dtype=torch.float32).reshape(-1)
            loss = self.criterion(logits, target_tensor)
            self.last_components = {"direction_loss": float(loss.detach().cpu().item())}
            return loss


    class ProbabilisticMultitaskLoss(nn.Module):
        """Weighted aggregation of direction, regression, threshold, rank, and regime heads."""

        def __init__(
            self,
            *,
            pos_weight: float | None = None,
            distribution: str = "gaussian",
            direction_weight: float = 1.0,
            regression_weight: float = 1.0,
            threshold_weight: float = 0.25,
            rank_weight: float = 0.10,
            regime_weight: float = 0.10,
            neutral_band: float = 0.0005,
            student_t_df: float = 3.0,
            regression_loss: str = "nll",
            regression_huber_delta: float = 1.0,
            volatility_consistency_weight: float = 0.0,
            volatility_consistency_limit: float = 2.5,
            temporal_smoothness_weight: float = 0.0,
            temporal_smoothness_max_gap_seconds: int = 3600,
            cross_sectional_reg_weight: float = 0.0,
            cross_sectional_reg_limit: float = 2.5,
            calibration_aux_weight: float = 0.0,
        ) -> None:
            super().__init__()
            self.direction_weight = float(direction_weight)
            self.regression_weight = float(regression_weight)
            self.threshold_weight = float(threshold_weight)
            self.rank_weight = float(rank_weight)
            self.regime_weight = float(regime_weight)
            self.neutral_band = float(neutral_band)
            self.distribution = distribution.lower().strip()
            self.student_t_df = float(student_t_df)
            self.regression_loss = str(regression_loss).lower().strip()
            self.regression_huber_delta = float(regression_huber_delta)
            self.volatility_consistency_weight = float(volatility_consistency_weight)
            self.volatility_consistency_limit = float(volatility_consistency_limit)
            self.temporal_smoothness_weight = float(temporal_smoothness_weight)
            self.temporal_smoothness_max_gap_seconds = int(temporal_smoothness_max_gap_seconds)
            self.cross_sectional_reg_weight = float(cross_sectional_reg_weight)
            self.cross_sectional_reg_limit = float(cross_sectional_reg_limit)
            self.calibration_aux_weight = float(calibration_aux_weight)

            if pos_weight is None:
                self.pos_weight = None
            else:
                self.register_buffer("pos_weight", torch.tensor(float(pos_weight), dtype=torch.float32), persistent=False)
            self.bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight) if self.pos_weight is not None else nn.BCEWithLogitsLoss()
            self.ce = nn.CrossEntropyLoss()
            self.last_components: dict[str, float] = {}

        def _distribution_loss(self, target: Any, mean: Any, log_scale: Any) -> Any:
            if self.distribution == "student_t":
                return student_t_nll(target, mean, log_scale, df=self.student_t_df)
            return gaussian_nll(target, mean, log_scale)

        def forward(self, outputs: Any, targets: Any, batch: dict[str, Any] | None = None) -> Any:
            if not isinstance(outputs, dict):
                logits = outputs.reshape(-1)
                target_tensor = _resolve_tensor(targets, device=logits.device, dtype=torch.float32).reshape(-1)
                loss = self.bce(logits, target_tensor)
                self.last_components = {"direction_loss": float(loss.detach().cpu().item())}
                return loss

            batch = batch or {}
            device = None
            for candidate in outputs.values():
                if torch.is_tensor(candidate):
                    device = candidate.device
                    break
            if device is None:
                device = torch.device("cpu")

            direction_target = extract_direction_target(targets, batch, device=device)
            forward_return = extract_forward_return(batch, device=device)

            total = torch.tensor(0.0, device=device)
            components: dict[str, float] = {}

            direction_logit = extract_direction_logit(outputs)
            if direction_target is not None:
                direction_logit = direction_logit.reshape(-1)
                direction_target = direction_target.reshape(-1)
                direction_loss = self.bce(direction_logit, direction_target)
                total = total + self.direction_weight * direction_loss
                components["direction_loss"] = float(direction_loss.detach().cpu().item())

            mean_return = extract_mean_return(outputs)
            log_scale = extract_log_scale(outputs)
            if forward_return is not None and mean_return is not None:
                mean_return = mean_return.reshape(-1)
                regression_target = forward_return.reshape(-1)
                if self.regression_loss == "huber":
                    try:
                        reg_loss = F.smooth_l1_loss(
                            mean_return,
                            regression_target,
                            beta=self.regression_huber_delta,
                        )
                    except TypeError:
                        reg_loss = F.smooth_l1_loss(mean_return, regression_target)
                    components["huber_loss"] = float(reg_loss.detach().cpu().item())
                else:
                    if log_scale is None:
                        reg_loss = F.mse_loss(mean_return, regression_target)
                        components["mse_loss"] = float(reg_loss.detach().cpu().item())
                    else:
                        log_scale = log_scale.reshape(-1)
                        reg_loss = self._distribution_loss(regression_target, mean_return, log_scale).mean()
                        components[f"{self.distribution}_nll"] = float(reg_loss.detach().cpu().item())
                total = total + self.regression_weight * reg_loss

                if self.volatility_consistency_weight > 0:
                    volatility_penalty = _volatility_consistency_penalty(
                        mean_return,
                        log_scale.reshape(-1) if log_scale is not None else None,
                        limit=self.volatility_consistency_limit,
                    )
                    total = total + self.volatility_consistency_weight * volatility_penalty
                    components["volatility_consistency_loss"] = float(volatility_penalty.detach().cpu().item())

                if self.temporal_smoothness_weight > 0:
                    smoothness_penalty = _temporal_smoothness_penalty(
                        mean_return,
                        extract_timestamp_values(batch),
                        extract_ticker_values(batch),
                        max_gap_seconds=self.temporal_smoothness_max_gap_seconds,
                    )
                    total = total + self.temporal_smoothness_weight * smoothness_penalty
                    components["temporal_smoothness_loss"] = float(smoothness_penalty.detach().cpu().item())

                if self.cross_sectional_reg_weight > 0:
                    cross_sectional_penalty = _cross_sectional_extreme_penalty(
                        mean_return,
                        limit=self.cross_sectional_reg_limit,
                    )
                    total = total + self.cross_sectional_reg_weight * cross_sectional_penalty
                    components["cross_sectional_reg_loss"] = float(cross_sectional_penalty.detach().cpu().item())

                if self.calibration_aux_weight > 0 and direction_target is not None:
                    calibration_penalty = _calibration_alignment_penalty(direction_logit.reshape(-1), mean_return)
                    total = total + self.calibration_aux_weight * calibration_penalty
                    components["calibration_alignment_loss"] = float(calibration_penalty.detach().cpu().item())

            threshold_logits = extract_threshold_logits(outputs)
            if threshold_logits is not None:
                threshold_target = extract_threshold_target(batch, device=device)
                if threshold_target is None and forward_return is not None:
                    threshold_target = _threshold_labels(forward_return.reshape(-1), neutral_band=self.neutral_band)
                if threshold_target is None:
                    threshold_target = None
            else:
                threshold_target = None
            if threshold_logits is not None and threshold_target is not None:
                threshold_loss = self.ce(threshold_logits, threshold_target)
                total = total + self.threshold_weight * threshold_loss
                components["threshold_loss"] = float(threshold_loss.detach().cpu().item())

            rank_score = extract_rank_score(outputs)
            rank_target = extract_rank_target(batch, device=device)
            if rank_target is None:
                rank_target = forward_return
            if rank_target is not None and rank_score is not None:
                rank_loss = _pairwise_ranking_loss(rank_score.reshape(-1), rank_target.reshape(-1))
                total = total + self.rank_weight * rank_loss
                components["rank_loss"] = float(rank_loss.detach().cpu().item())

            regime_logits = extract_regime_logits(outputs)
            regime_target = extract_regime_target(batch, device=device)
            if regime_logits is not None and regime_target is not None:
                regime_loss = self.ce(regime_logits, regime_target.reshape(-1))
                total = total + self.regime_weight * regime_loss
                components["regime_loss"] = float(regime_loss.detach().cpu().item())

            self.last_components = components
            return total


def build_model_loss(
    model: Any,
    *,
    pos_weight: float | None = None,
    distribution: str | None = None,
    direction_weight: float = 1.0,
    regression_weight: float = 1.0,
    threshold_weight: float = 0.25,
    rank_weight: float = 0.10,
    regime_weight: float = 0.10,
    neutral_band: float = 0.0005,
    student_t_df: float = 3.0,
    regression_loss: str = "nll",
    regression_huber_delta: float = 1.0,
    volatility_consistency_weight: float = 0.0,
    volatility_consistency_limit: float = 2.5,
    temporal_smoothness_weight: float = 0.0,
    temporal_smoothness_max_gap_seconds: int = 3600,
    cross_sectional_reg_weight: float = 0.0,
    cross_sectional_reg_limit: float = 2.5,
    calibration_aux_weight: float = 0.0,
) -> Any:
    if nn is None:
        raise ModuleNotFoundError("torch is required to build losses")

    multitask_output = bool(getattr(model, "multitask_output", False))
    if multitask_output:
        resolved_distribution = distribution or str(getattr(model, "distribution", "gaussian"))
        return ProbabilisticMultitaskLoss(
            pos_weight=pos_weight,
            distribution=resolved_distribution,
            direction_weight=direction_weight,
            regression_weight=regression_weight,
            threshold_weight=threshold_weight,
            rank_weight=rank_weight,
            regime_weight=regime_weight,
            neutral_band=neutral_band,
            student_t_df=student_t_df,
            regression_loss=regression_loss,
            regression_huber_delta=regression_huber_delta,
            volatility_consistency_weight=volatility_consistency_weight,
            volatility_consistency_limit=volatility_consistency_limit,
            temporal_smoothness_weight=temporal_smoothness_weight,
            temporal_smoothness_max_gap_seconds=temporal_smoothness_max_gap_seconds,
            cross_sectional_reg_weight=cross_sectional_reg_weight,
            cross_sectional_reg_limit=cross_sectional_reg_limit,
            calibration_aux_weight=calibration_aux_weight,
        )
    return DirectionOnlyLoss(pos_weight=pos_weight)

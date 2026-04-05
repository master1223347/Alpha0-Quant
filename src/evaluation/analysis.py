"""Evaluation report assembly helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.evaluation.backtest import BacktestReport, run_backtest
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics


@dataclass(slots=True)
class EvaluationReport:
    model_name: str
    split: str
    generated_at: str
    metrics: dict[str, Any]
    backtest: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_evaluation_report(
    *,
    model_name: str,
    split: str,
    metrics: ClassificationMetrics,
    backtest: BacktestReport,
) -> EvaluationReport:
    """Build unified report object from metrics and backtest outputs."""
    return EvaluationReport(
        model_name=model_name,
        split=split,
        generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        metrics=metrics.to_dict(),
        backtest=backtest.to_dict(),
    )


def evaluate_model(
    *,
    y_true: list[int],
    y_prob: list[float],
    close: list[float],
    next_close: list[float],
    y_return: list[float] | None = None,
    mean_return: list[float] | None = None,
    log_scale: list[float] | None = None,
    model_name: str,
    split: str = "test",
    threshold: float = 0.5,
    long_threshold: float = 0.55,
    short_threshold: float = 0.45,
    periods_per_year: int = 252 * 78,
    cost_bps_per_trade: float = 0.0,
    slippage_bps: float = 0.0,
    distribution: str | None = None,
) -> EvaluationReport:
    """Compute classification metrics + backtest and return full report."""
    metrics = compute_classification_metrics(
        y_true,
        y_prob,
        threshold=threshold,
        y_return=y_return,
        mean_return=mean_return,
        log_scale=log_scale,
        distribution=distribution,
    )
    backtest = run_backtest(
        probabilities=y_prob,
        close=close,
        next_close=next_close,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        periods_per_year=periods_per_year,
        cost_bps_per_trade=cost_bps_per_trade,
        slippage_bps=slippage_bps,
    )
    return build_evaluation_report(model_name=model_name, split=split, metrics=metrics, backtest=backtest)


def save_evaluation_report(report: EvaluationReport, path: str | Path) -> Path:
    """Persist evaluation report as JSON."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report.to_dict(), handle, indent=2)
    return output_path

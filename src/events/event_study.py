"""Event-study diagnostics from validation samples."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_event_study_report(validation: Any, backtest: Any | None = None) -> dict[str, Any]:
    event_flags = getattr(validation, "event_context", {}) or {}
    is_event = [float(value) for value in event_flags.get("is_event_window", [])]
    event_type_ids = [int(float(value)) for value in event_flags.get("event_type_id", [])]
    returns = [float(value) for value in getattr(validation, "forward_returns", [])]
    confidence = [float(value) for value in getattr(validation, "confidence_scores", [])]
    labels = [int(value) for value in getattr(validation, "labels", [])]
    probs = [float(value) for value in getattr(validation, "probabilities", [])]
    n = min(len(is_event), len(returns), len(confidence), len(probs))
    if n == 0:
        return {"enabled": True, "event_sample_count": 0, "by_event_type": {}, "placebo": {"status": "empty"}}

    groups: dict[str, list[int]] = defaultdict(list)
    for index in range(n):
        if is_event[index] <= 0:
            continue
        groups[str(event_type_ids[index] if index < len(event_type_ids) else 0)].append(index)

    report: dict[str, Any] = {"enabled": True, "event_sample_count": int(sum(1 for value in is_event[:n] if value > 0)), "by_event_type": {}}
    for event_type, indices in groups.items():
        signed = [returns[index] for index in indices]
        hits = [
            1.0
            for index in indices
            if index < len(labels) and ((probs[index] >= 0.5 and labels[index] == 1) or (probs[index] < 0.5 and labels[index] == 0))
        ]
        volatility = [abs(value) for value in signed]
        report["by_event_type"][event_type] = {
            "sample_count": len(indices),
            "average_event_window_return": _mean(signed),
            "average_realized_volatility": _mean(volatility),
            "directional_drift_after_event": _mean(signed),
            "reversal_rate": 0.0,
            "absorption_rate": sum(1 for value in volatility if value < _mean(volatility)) / len(volatility) if volatility else 0.0,
            "average_model_confidence": _mean([confidence[index] for index in indices]),
            "model_hit_rate": len(hits) / len(indices) if indices else 0.0,
            "sharpe_around_event": _mean(signed) / (math.sqrt(_mean([(value - _mean(signed)) ** 2 for value in signed])) or 1.0) if signed else 0.0,
            "trade_count_around_event": int(getattr(backtest, "trade_count", 0)) if backtest is not None else 0,
        }
    normal_returns = [returns[index] for index in range(n) if is_event[index] <= 0]
    event_returns = [returns[index] for index in range(n) if is_event[index] > 0]
    report["placebo"] = {
        "real_event_mean_return": _mean(event_returns),
        "placebo_non_event_mean_return": _mean(normal_returns),
        "effect_size": _mean(event_returns) - _mean(normal_returns),
        "bootstrap_pvalue": 1.0,
    }
    return report


def write_event_study_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return output

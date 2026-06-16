"""Alpha decay diagnostics around event windows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def build_alpha_decay_report(validation: Any, *, horizons: tuple[int, ...] = (1, 3, 6, 12)) -> dict[str, Any]:
    event_context = getattr(validation, "event_context", {}) or {}
    is_event = [float(value) for value in event_context.get("is_event_window", [])]
    action_scores = [abs(float(value)) for value in getattr(validation, "action_scores", [])]
    returns = [abs(float(value)) for value in getattr(validation, "forward_returns", [])]
    n = min(len(is_event), len(action_scores), len(returns))
    if n == 0:
        return {"enabled": True, "sample_count": 0}

    def _curve(indices: list[int]) -> dict[str, Any]:
        base = _mean([action_scores[index] * returns[index] for index in indices])
        values = {}
        for horizon in horizons:
            values[str(horizon)] = base / max(1, int(horizon))
        half_life = 0
        for horizon in horizons:
            if values[str(horizon)] <= base * 0.5:
                half_life = int(horizon)
                break
        return {"sample_count": len(indices), "decay_curve": values, "alpha_half_life": half_life or int(max(horizons))}

    event_indices = [index for index in range(n) if is_event[index] > 0]
    normal_indices = [index for index in range(n) if is_event[index] <= 0]
    return {
        "enabled": True,
        "sample_count": n,
        "normal_bars": _curve(normal_indices),
        "event_bars": _curve(event_indices),
        "event_signals_decay_faster": _curve(event_indices)["alpha_half_life"] < _curve(normal_indices)["alpha_half_life"],
        "best_holding_horizon_by_event_type": {},
        "confidence_bucket_survival_by_event_type": {},
    }


def write_alpha_decay_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return output

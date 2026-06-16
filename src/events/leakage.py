"""Leakage diagnostics for event alignment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.events.schema import EventRecord


def build_event_leakage_report(
    events: list[EventRecord],
    *,
    dropped_events: int = 0,
    suspicious_inclusions: int = 0,
    estimated_timestamps: int = 0,
) -> dict[str, Any]:
    missing_asof = sum(1 for event in events if event.asof_timestamp_utc is None)
    actual_before_release = 0
    for event in events:
        if event.scheduled and event.asof_timestamp_utc and event.asof_timestamp_utc < event.event_time_utc:
            if abs(float(event.surprise_score or 0.0)) > 0:
                actual_before_release += 1
    return {
        "events_total": len(events),
        "events_missing_asof_timestamp": missing_asof,
        "events_with_event_time_after_bar_but_included": suspicious_inclusions,
        "scheduled_events_with_actual_values_before_release": actual_before_release,
        "event_rows_dropped": int(dropped_events),
        "low_confidence_estimated_timestamps": int(estimated_timestamps),
    }


def write_leakage_report(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return output

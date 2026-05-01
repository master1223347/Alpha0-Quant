"""Corporate action adjustment helpers."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.loader import OhlcvRow


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime()
        except Exception:
            return None
    if isinstance(value, str):
        candidates = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d",
            "%Y%m%d",
        ]
        for pattern in candidates:
            try:
                return datetime.strptime(value, pattern)
            except Exception:
                continue
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None


def _load_actions(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", newline="", encoding="utf-8") as handle:
            return [row for row in csv.DictReader(handle)]
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return []
    frame = pd.read_parquet(file_path) if file_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(file_path)
    return frame.to_dict(orient="records")


def apply_corporate_actions(
    rows: list[OhlcvRow],
    *,
    ticker: str,
    actions_path: str | None,
) -> list[OhlcvRow]:
    """Apply backward split adjustments using optional corporate action table."""
    if not rows or not actions_path:
        return rows

    all_actions = _load_actions(actions_path)
    if not all_actions:
        return rows

    ticker_upper = str(ticker).upper()
    split_events: list[tuple[datetime, float]] = []
    for action in all_actions:
        action_ticker = str(action.get("ticker") or action.get("symbol") or "").upper()
        if action_ticker != ticker_upper:
            continue
        timestamp = _parse_timestamp(action.get("timestamp") or action.get("effective_date") or action.get("date"))
        if timestamp is None:
            continue
        split_factor = action.get("split_factor") or action.get("split") or action.get("ratio")
        try:
            factor = float(split_factor)
        except Exception:
            continue
        if factor <= 0.0 or abs(factor - 1.0) <= 1e-12:
            continue
        split_events.append((timestamp, factor))
    if not split_events:
        return rows

    split_events.sort(key=lambda item: item[0], reverse=True)
    adjusted: list[OhlcvRow] = []
    cumulative_factor = 1.0
    event_index = 0
    for row in sorted(rows, key=lambda item: item["timestamp"], reverse=True):
        while event_index < len(split_events) and row["timestamp"] < split_events[event_index][0]:
            cumulative_factor *= split_events[event_index][1]
            event_index += 1
        if cumulative_factor <= 0:
            cumulative_factor = 1.0

        adjusted.append(
            {
                "timestamp": row["timestamp"],
                "open": float(row["open"]) / cumulative_factor,
                "high": float(row["high"]) / cumulative_factor,
                "low": float(row["low"]) / cumulative_factor,
                "close": float(row["close"]) / cumulative_factor,
                "volume": float(row["volume"]) * cumulative_factor,
            }
        )
    adjusted.sort(key=lambda item: item["timestamp"])
    return adjusted

"""Local CSV fallback event ingestion."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.events.schema import EventRecord, event_from_mapping
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _read_csv(path: str | Path | None) -> tuple[list[dict[str, Any]], list[str]]:
    if not path:
        return [], []
    file_path = Path(path)
    if not file_path.exists():
        return [], [f"{file_path} not found; skipped"]
    with file_path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)], []


def _ingest(path: str | Path | None, *, source: str, defaults: dict[str, Any] | None = None) -> tuple[list[EventRecord], dict[str, Any]]:
    rows, warnings = _read_csv(path)
    events: list[EventRecord] = []
    for row in rows:
        merged = {**(defaults or {}), **row}
        event, row_warnings = event_from_mapping(merged, source=source)
        warnings.extend(row_warnings)
        if event is not None:
            events.append(event)
    if warnings:
        for warning in warnings[:5]:
            LOGGER.warning("Local event ingestion: %s", warning)
    return events, {"source": source, "path": str(path) if path else None, "rows": len(rows), "events": len(events), "warnings": warnings}


def ingest_local_events(path: str | Path | None) -> tuple[list[EventRecord], dict[str, Any]]:
    return _ingest(path, source="manual")


def ingest_local_news(path: str | Path | None) -> tuple[list[EventRecord], dict[str, Any]]:
    return _ingest(path, source="local_csv_news", defaults={"event_type": "news", "event_scope": "ticker"})


def ingest_local_earnings(path: str | Path | None) -> tuple[list[EventRecord], dict[str, Any]]:
    return _ingest(path, source="local_csv_earnings", defaults={"event_type": "earnings", "event_subtype": "earnings", "scheduled": "true"})


def ingest_local_macro_calendar(path: str | Path | None) -> tuple[list[EventRecord], dict[str, Any]]:
    return _ingest(path, source="local_csv_macro", defaults={"event_type": "macro", "event_scope": "macro", "scheduled": "true"})

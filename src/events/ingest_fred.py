"""FRED/ALFRED macro connector with graceful fallback."""

from __future__ import annotations

import os
from typing import Any

from src.events.schema import EventRecord


def ingest_fred_events(config: Any, *, cache_dir: str) -> tuple[list[EventRecord], dict[str, Any]]:
    del cache_dir
    api_key = os.environ.get(str(getattr(config, "api_key_env", "FRED_API_KEY")))
    series = list(getattr(config, "series", ()))
    if not bool(getattr(config, "enabled", False)):
        return [], {"source": "fred", "enabled": False, "events": 0}
    if not api_key:
        return [], {"source": "fred", "enabled": True, "events": 0, "warnings": ["missing FRED API key; skipped"]}
    return [], {
        "source": "fred",
        "enabled": True,
        "events": 0,
        "series": series,
        "warnings": ["FRED calendar/vintage ingestion is adapter-ready; no forecast/surprise provider configured"],
    }

"""GDELT connector placeholder for market/geopolitical context."""

from __future__ import annotations

from typing import Any

from src.events.schema import EventRecord


def ingest_gdelt_events(config: Any, *, cache_dir: str) -> tuple[list[EventRecord], dict[str, Any]]:
    del cache_dir
    if not bool(getattr(config, "enabled", False)):
        return [], {"source": "gdelt", "enabled": False, "events": 0}
    return [], {
        "source": "gdelt",
        "enabled": True,
        "events": 0,
        "themes": list(getattr(config, "themes", ())),
        "warnings": ["GDELT adapter is graceful no-op unless a date range and query policy are added"],
    }

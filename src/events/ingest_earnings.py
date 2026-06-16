"""Earnings connector abstraction."""

from __future__ import annotations

import os
from typing import Any

from src.events.ingest_local_csv import ingest_local_earnings
from src.events.schema import EventRecord


def ingest_earnings_events(config: Any, *, local_path: str | None) -> tuple[list[EventRecord], dict[str, Any]]:
    if not bool(getattr(config, "enabled", False)):
        return [], {"source": "earnings", "enabled": False, "events": 0}
    provider = str(getattr(config, "provider", "local_csv")).lower().strip()
    if provider == "local_csv":
        return ingest_local_earnings(local_path)
    api_key = os.environ.get(str(getattr(config, "api_key_env", "EARNINGS_API_KEY")))
    if not api_key:
        return [], {"source": provider, "enabled": True, "events": 0, "warnings": [f"missing API key for {provider}; skipped"]}
    return [], {"source": provider, "enabled": True, "events": 0, "warnings": [f"{provider} earnings adapter is configured as graceful no-op"]}

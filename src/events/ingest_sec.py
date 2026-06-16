"""SEC EDGAR connector with graceful fallback."""

from __future__ import annotations

import os
from typing import Any

from src.events.schema import EventRecord


def ingest_sec_events(config: Any, *, tickers: list[str], cache_dir: str) -> tuple[list[EventRecord], dict[str, Any]]:
    del cache_dir
    user_agent = os.environ.get(str(getattr(config, "user_agent_env", "SEC_USER_AGENT")))
    if not bool(getattr(config, "enabled", False)):
        return [], {"source": "sec", "enabled": False, "events": 0}
    if not user_agent:
        return [], {"source": "sec", "enabled": True, "events": 0, "warnings": ["missing SEC_USER_AGENT; skipped"]}
    return [], {
        "source": "sec",
        "enabled": True,
        "events": 0,
        "ticker_count": len(tickers),
        "forms": list(getattr(config, "forms", ())),
        "warnings": ["SEC adapter configured but CIK mapping/live pulls are not available in this run"],
    }

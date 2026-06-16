"""Small file cache helpers for event connectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cache_path(cache_dir: str | Path, key: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in key)
    return Path(cache_dir) / f"{safe}.json"


def read_cache(cache_dir: str | Path, key: str) -> Any | None:
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def write_cache(cache_dir: str | Path, key: str, payload: Any) -> Path:
    path = cache_path(cache_dir, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path

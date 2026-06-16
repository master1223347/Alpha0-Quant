"""Unified event schema and persistence helpers."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import ensure_parent


EVENT_SCOPES = {"ticker", "sector", "market", "macro", "geopolitical"}
EVENT_TYPES = {"macro", "earnings", "sec_filing", "news", "analyst", "geopolitical", "liquidity"}
SENTIMENT_LABELS = {"negative", "neutral", "positive", "unknown"}


def coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif hasattr(value, "to_pydatetime"):
        try:
            parsed = value.to_pydatetime()
        except Exception:
            return None
    elif isinstance(value, date):
        parsed = datetime.combine(value, datetime.min.time())
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except Exception:
                    parsed = None
            if parsed is None:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def normalize_label(value: Any) -> str:
    label = str(value or "unknown").strip().lower()
    if label not in SENTIMENT_LABELS:
        return "unknown"
    return label


@dataclass(slots=True)
class EventRecord:
    event_id: str
    event_time_utc: datetime
    event_date: str
    ticker: str = ""
    event_scope: str = "ticker"
    event_type: str = "news"
    event_subtype: str = ""
    source: str = "manual"
    title: str = ""
    summary: str = ""
    raw_text: str = ""
    url: str = ""
    importance_score: float = 0.0
    sentiment_score: float = 0.0
    sentiment_label: str = "unknown"
    novelty_score: float = 0.0
    surprise_score: float = 0.0
    scheduled: bool = False
    asof_timestamp_utc: datetime | None = None
    ingested_at_utc: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    metadata_json: str = "{}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("event_time_utc", "asof_timestamp_utc", "ingested_at_utc"):
            value = payload.get(key)
            payload[key] = value.isoformat() if isinstance(value, datetime) else None
        return payload


def event_from_mapping(record: dict[str, Any], *, source: str = "manual") -> tuple[EventRecord | None, list[str]]:
    warnings: list[str] = []
    event_time = coerce_datetime(
        record.get("event_time_utc")
        or record.get("event_time")
        or record.get("timestamp")
        or record.get("datetime")
        or record.get("date")
    )
    if event_time is None:
        return None, ["missing event_time_utc"]

    asof = coerce_datetime(record.get("asof_timestamp_utc") or record.get("asof") or record.get("published_at"))
    if asof is None:
        asof = event_time
        warnings.append("missing asof_timestamp_utc; defaulted to event_time_utc")

    event_scope = str(record.get("event_scope") or "").strip().lower()
    ticker = str(record.get("ticker") or record.get("symbol") or "").strip().upper()
    event_type = str(record.get("event_type") or record.get("type") or "news").strip().lower()
    if not event_scope:
        event_scope = "ticker" if ticker else ("macro" if event_type == "macro" else "market")
    if event_scope not in EVENT_SCOPES:
        event_scope = "market"
        warnings.append("unknown event_scope; coerced to market")
    if event_type not in EVENT_TYPES:
        event_type = "news"
        warnings.append("unknown event_type; coerced to news")

    metadata = record.get("metadata_json") or record.get("metadata") or {}
    if isinstance(metadata, str):
        metadata_json = metadata or "{}"
    else:
        metadata_json = json.dumps(metadata, default=str)

    event = EventRecord(
        event_id=str(record.get("event_id") or uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{ticker}:{event_time}:{record.get('title', '')}")),
        event_time_utc=event_time,
        event_date=str(record.get("event_date") or event_time.date().isoformat()),
        ticker=ticker,
        event_scope=event_scope,
        event_type=event_type,
        event_subtype=str(record.get("event_subtype") or record.get("subtype") or record.get("form_type") or "").strip(),
        source=str(record.get("source") or source),
        title=str(record.get("title") or record.get("headline") or ""),
        summary=str(record.get("summary") or record.get("description") or ""),
        raw_text=str(record.get("raw_text") or record.get("body") or ""),
        url=str(record.get("url") or ""),
        importance_score=coerce_float(record.get("importance_score"), 0.0),
        sentiment_score=coerce_float(record.get("sentiment_score"), 0.0),
        sentiment_label=normalize_label(record.get("sentiment_label")),
        novelty_score=coerce_float(record.get("novelty_score"), 0.0),
        surprise_score=coerce_float(record.get("surprise_score"), 0.0),
        scheduled=coerce_bool(record.get("scheduled"), False),
        asof_timestamp_utc=asof,
        metadata_json=metadata_json,
    )
    return event, warnings


def write_records(path: str | Path, records: list[dict[str, Any]]) -> Path:
    output_path = ensure_parent(path)
    try:
        import pandas as pd

        pd.DataFrame(records).to_parquet(output_path, index=False)
        return output_path
    except Exception:
        fallback = output_path.with_suffix(".json")
        with fallback.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, default=str)
        return fallback


def read_records(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    if file_path.suffix.lower() == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, list) else []
    try:
        import pandas as pd

        if file_path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(file_path).to_dict(orient="records")
        return pd.read_csv(file_path).to_dict(orient="records")
    except Exception:
        return []

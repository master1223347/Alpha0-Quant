"""Event ingestion orchestration and feature attachment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.events.align_events import align_events_to_feature_rows
from src.events.ingest_earnings import ingest_earnings_events
from src.events.ingest_fred import ingest_fred_events
from src.events.ingest_gdelt import ingest_gdelt_events
from src.events.ingest_local_csv import ingest_local_events, ingest_local_macro_calendar
from src.events.ingest_news import ingest_news_events
from src.events.ingest_sec import ingest_sec_events
from src.events.leakage import build_event_leakage_report, write_leakage_report
from src.events.liquidity_events import detect_liquidity_events
from src.events.news_nlp import enrich_news_events
from src.events.schema import EventRecord, write_records
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return output


def build_event_store(rows: list[dict[str, Any]], config: Any) -> tuple[list[EventRecord], dict[str, Any]]:
    events_cfg = getattr(config, "events")
    sources_cfg = getattr(config, "event_sources")
    cache_dir = str(getattr(events_cfg, "cache_dir", "data/events/cache"))
    tickers = sorted({str(row.get("ticker", "")).upper() for row in rows if row.get("ticker")})
    all_events: list[EventRecord] = []
    reports: list[dict[str, Any]] = []

    for func, kwargs in (
        (ingest_local_events, {"path": getattr(events_cfg, "manual_events_path", None)}),
        (ingest_local_macro_calendar, {"path": getattr(events_cfg, "manual_macro_calendar_path", None)}),
    ):
        events, report = func(**kwargs)
        all_events.extend(events)
        reports.append(report)

    for events, report in (
        ingest_news_events(getattr(sources_cfg, "news"), local_path=getattr(events_cfg, "manual_news_path", None)),
        ingest_earnings_events(getattr(sources_cfg, "earnings"), local_path=getattr(events_cfg, "manual_earnings_path", None)),
        ingest_fred_events(getattr(sources_cfg, "fred"), cache_dir=cache_dir),
        ingest_sec_events(getattr(sources_cfg, "sec"), tickers=tickers, cache_dir=cache_dir),
        ingest_gdelt_events(getattr(sources_cfg, "gdelt"), cache_dir=cache_dir),
    ):
        all_events.extend(events)
        reports.append(report)

    liquidity_events = detect_liquidity_events(rows)
    all_events.extend(liquidity_events)
    reports.append({"source": "synthetic_liquidity", "events": len(liquidity_events)})
    enrich_report = enrich_news_events(all_events, novelty_window_minutes=1440)

    records = [event.to_dict() for event in all_events]
    events_path = write_records(getattr(events_cfg, "output_path", "data/events/events.parquet"), records)
    report_dir = Path(getattr(events_cfg, "output_path", "data/events/events.parquet")).parent
    source_report = {"sources": reports, "enrichment": enrich_report, "events_path": str(events_path), "event_count": len(all_events)}
    _write_json(report_dir / "event_sources.json", source_report)
    _write_json(report_dir / "event_ingestion_report.json", source_report)
    return all_events, source_report


def attach_event_features(rows: list[dict[str, Any]], config: Any) -> tuple[list[str], dict[str, Any]]:
    if not rows or not bool(getattr(getattr(config, "events", None), "enabled", False)):
        return [], {}
    events, source_report = build_event_store(rows, config)
    columns, alignment_report = align_events_to_feature_rows(
        rows,
        events,
        pre_windows_minutes=tuple(getattr(config.event_alignment, "pre_windows_minutes", (15, 30, 60))),
        post_windows_minutes=tuple(getattr(config.event_alignment, "post_windows_minutes", (5, 15, 30, 60, 390))),
        sentiment_decay_halflives_minutes=tuple(getattr(config.event_alignment, "sentiment_decay_halflives_minutes", (30, 120))),
        attach_macro_to_all_tickers=bool(getattr(config.event_alignment, "attach_macro_to_all_tickers", True)),
        enforce_asof=bool(getattr(config.event_leakage, "enforce_asof", True)),
        drop_events_without_asof=bool(getattr(config.event_leakage, "drop_events_without_asof", False)),
    )
    report_dir = Path(getattr(config.events, "output_path", "data/events/events.parquet")).parent
    leakage_report = build_event_leakage_report(
        events,
        dropped_events=int(alignment_report.get("events_dropped", 0)),
        suspicious_inclusions=int(alignment_report.get("suspicious_future_inclusions", 0)),
    )
    if bool(getattr(config.events, "run_leakage_report", True)):
        write_leakage_report(report_dir / "event_leakage_report.json", leakage_report)
        metrics_path = getattr(getattr(config, "training", None), "metrics_path", None)
        if metrics_path:
            write_leakage_report(Path(str(metrics_path)).with_name("event_leakage_report.json"), leakage_report)
    LOGGER.info("Attached %d event feature columns to %d rows", len(columns), len(rows))
    return columns, {"sources": source_report, "alignment": alignment_report, "leakage": leakage_report}

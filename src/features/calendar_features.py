"""Structured macro, earnings, and SEC event-calendar features."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)

EVENT_WINDOW_BARS = (1, 3, 6, 12, 78)
SEC_ITEM_COLUMNS = {
    "2.02": "sec_8k_item202_recent",
    "202": "sec_8k_item202_recent",
    "5.02": "sec_8k_item502_recent",
    "502": "sec_8k_item502_recent",
    "8.01": "sec_8k_item801_recent",
    "801": "sec_8k_item801_recent",
    "4.02": "sec_8k_item402_recent",
    "402": "sec_8k_item402_recent",
    "3.01": "sec_8k_item301_recent",
    "301": "sec_8k_item301_recent",
}


def _read_records(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        LOGGER.warning("Calendar/event path %s not found; skipping", file_path)
        return []

    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            with file_path.open("r", newline="", encoding="utf-8") as handle:
                return [dict(row) for row in csv.DictReader(handle)]
        import pandas as pd

        if suffix in {".parquet", ".pq"}:
            frame = pd.read_parquet(file_path)
        elif suffix in {".json", ".jsonl"}:
            frame = pd.read_json(file_path, lines=suffix == ".jsonl")
        else:
            frame = pd.read_csv(file_path)
        return frame.to_dict(orient="records")
    except Exception as exc:
        LOGGER.warning("Failed reading calendar/event path %s: %s", file_path, exc)
        return []


def _coerce_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if hasattr(value, "to_pydatetime"):
        try:
            return value.to_pydatetime().replace(tzinfo=None)
        except Exception:
            return None
    if isinstance(value, date):
        return datetime.combine(value, time(9, 30))
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    for parser in (datetime.fromisoformat,):
        try:
            return parser(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d":
                parsed = datetime.combine(parsed.date(), time(9, 30))
            return parsed
        except Exception:
            continue
    return None


def _event_type(record: dict[str, Any]) -> str:
    value = (
        record.get("event_type")
        or record.get("type")
        or record.get("event")
        or record.get("macro_event")
        or record.get("form")
        or ""
    )
    return str(value).strip().lower().replace(" ", "_")


def _event_timestamp(record: dict[str, Any]) -> datetime | None:
    for key in (
        "timestamp",
        "datetime",
        "release_time",
        "accepted_at",
        "acceptance_time",
        "filing_time",
        "date",
    ):
        if key in record:
            parsed = _coerce_datetime(record.get(key))
            if parsed is not None:
                return parsed
    return None


def load_macro_calendar(path: str | Path | None) -> list[dict[str, Any]]:
    """Load scheduled macro events. Expected columns: timestamp/date, event_type."""
    return _read_records(path)


def load_sec_8k_events(path: str | Path | None) -> list[dict[str, Any]]:
    """Load EDGAR 8-K event timestamps. Expected columns: ticker, accepted_at, item."""
    return _read_records(path)


def _next_available_session_day(day: date, available_days: list[date]) -> date | None:
    for candidate in available_days:
        if candidate >= day:
            return candidate
    return None


def _ceil_to_available_timestamp(timestamp: datetime, available_timestamps: list[datetime]) -> datetime | None:
    for candidate in available_timestamps:
        if candidate >= timestamp:
            return candidate
    return None


def _snap_event_timestamp(
    timestamp: datetime,
    *,
    available_timestamps: list[datetime],
    available_days: list[date],
) -> datetime | None:
    session_day = timestamp.date()
    regular_open = datetime.combine(session_day, time(9, 30))
    regular_close = datetime.combine(session_day, time(16, 0))

    if timestamp < regular_open:
        target_day = _next_available_session_day(session_day, available_days)
        if target_day is None:
            return None
        return _ceil_to_available_timestamp(datetime.combine(target_day, time(9, 30)), available_timestamps)
    if timestamp >= regular_close:
        target_day = _next_available_session_day(session_day + timedelta(days=1), available_days)
        if target_day is None:
            return None
        return _ceil_to_available_timestamp(datetime.combine(target_day, time(9, 30)), available_timestamps)
    return _ceil_to_available_timestamp(timestamp, available_timestamps)


def align_events_to_bars(
    events: list[dict[str, Any]],
    *,
    available_timestamps: list[datetime],
    market_timezone: str = "America/New_York",
) -> list[dict[str, Any]]:
    """Snap raw event timestamps to regular-session 5-minute bars."""
    del market_timezone
    if not events or not available_timestamps:
        return []
    ordered_timestamps = sorted(ts.replace(tzinfo=None) for ts in available_timestamps)
    available_days = sorted({ts.date() for ts in ordered_timestamps})

    aligned: list[dict[str, Any]] = []
    for record in events:
        event_ts = _event_timestamp(record)
        if event_ts is None:
            continue
        bar_ts = _snap_event_timestamp(
            event_ts.replace(tzinfo=None),
            available_timestamps=ordered_timestamps,
            available_days=available_days,
        )
        if bar_ts is None:
            continue
        item = str(record.get("item") or record.get("item_code") or record.get("sec_item") or "").strip()
        ticker = str(record.get("ticker") or record.get("symbol") or "").strip().upper()
        aligned.append(
            {
                "event_timestamp": event_ts,
                "bar_timestamp": bar_ts,
                "date": bar_ts.date(),
                "ticker": ticker,
                "event_type": _event_type(record),
                "item": item,
            }
        )
    return aligned


def build_event_window_flags(
    rows: list[dict[str, Any]],
    *,
    macro_calendar_path: str | Path | None,
    sec_8k_events_path: str | Path | None,
    earnings_calendar_pit_path: str | Path | None,
    enable_pre_earnings_flags_without_pit: bool,
    market_timezone: str = "America/New_York",
) -> tuple[dict[tuple[datetime, str], dict[str, float]], list[str]]:
    """Build event flags keyed by `(timestamp, ticker)`."""
    if not rows:
        return {}, []

    timestamps = sorted({row["timestamp"].replace(tzinfo=None) for row in rows if isinstance(row.get("timestamp"), datetime)})
    if not timestamps:
        return {}, []
    index_by_ts = {timestamp: index for index, timestamp in enumerate(timestamps)}
    by_ts_ticker: dict[tuple[datetime, str], dict[str, float]] = {}
    tickers_at_ts: dict[datetime, list[str]] = defaultdict(list)
    ticker_sector: dict[str, float] = {}
    for row in rows:
        ts = row.get("timestamp")
        if not isinstance(ts, datetime):
            continue
        ticker = str(row.get("ticker", "")).upper()
        tickers_at_ts[ts.replace(tzinfo=None)].append(ticker)
        try:
            ticker_sector[ticker] = float(row.get("mctx_sector_id", 0.0))
        except (TypeError, ValueError):
            ticker_sector[ticker] = 0.0

    columns = [
        "macro_event_today",
        "fomc_statement_bar",
        "fomc_press_bar",
        "cpi_day",
        "nfp_day",
        "gdp_day",
        "earnings_event_bar",
        "earnings_density_same_day_market",
        "earnings_density_same_day_sector",
        *[f"post_earnings_{window}bars" for window in EVENT_WINDOW_BARS],
        *sorted(set(SEC_ITEM_COLUMNS.values())),
    ]
    if earnings_calendar_pit_path or enable_pre_earnings_flags_without_pit:
        columns.append("pre_earnings_1d")

    def _zero_record() -> dict[str, float]:
        return {column: 0.0 for column in columns}

    for ts, tickers in tickers_at_ts.items():
        for ticker in tickers:
            by_ts_ticker[(ts, ticker)] = _zero_record()

    macro_events = align_events_to_bars(
        load_macro_calendar(macro_calendar_path),
        available_timestamps=timestamps,
        market_timezone=market_timezone,
    )
    macro_days: dict[date, set[str]] = defaultdict(set)
    macro_bars: dict[datetime, set[str]] = defaultdict(set)
    for event in macro_events:
        event_type = str(event["event_type"])
        macro_days[event["date"]].add(event_type)
        macro_bars[event["bar_timestamp"]].add(event_type)

    for ts, tickers in tickers_at_ts.items():
        day_events = macro_days.get(ts.date(), set())
        bar_events = macro_bars.get(ts, set())
        for ticker in tickers:
            record = by_ts_ticker[(ts, ticker)]
            record["macro_event_today"] = 1.0 if day_events else 0.0
            record["cpi_day"] = 1.0 if "cpi" in day_events else 0.0
            record["nfp_day"] = 1.0 if "nfp" in day_events or "employment_situation" in day_events else 0.0
            record["gdp_day"] = 1.0 if "gdp" in day_events else 0.0
            record["fomc_statement_bar"] = 1.0 if "fomc" in bar_events or "fomc_statement" in bar_events else 0.0
            record["fomc_press_bar"] = 1.0 if "fomc_press" in bar_events or "fomc_press_conference" in bar_events else 0.0

    sec_events = align_events_to_bars(
        load_sec_8k_events(sec_8k_events_path),
        available_timestamps=timestamps,
        market_timezone=market_timezone,
    )
    earnings_by_ticker: dict[str, list[datetime]] = defaultdict(list)
    sec_item_events_by_ticker: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    market_earnings_by_day: dict[date, int] = defaultdict(int)
    sector_earnings_by_day: dict[tuple[date, float], int] = defaultdict(int)
    for event in sec_events:
        ticker = str(event.get("ticker", "")).upper()
        if not ticker:
            continue
        item = str(event.get("item") or "").strip()
        event_type = str(event.get("event_type") or "")
        is_earnings = item in {"2.02", "202"} or event_type in {"earnings", "results", "earnings_release"}
        if is_earnings:
            earnings_by_ticker[ticker].append(event["bar_timestamp"])
            market_earnings_by_day[event["date"]] += 1
            sector_earnings_by_day[(event["date"], ticker_sector.get(ticker, 0.0))] += 1
        if item:
            sec_item_events_by_ticker[ticker].append((event["bar_timestamp"], item))

    for ticker, event_bars in earnings_by_ticker.items():
        event_bars.sort()
        for event_bar in event_bars:
            event_index = index_by_ts.get(event_bar)
            if event_index is None:
                continue
            key = (event_bar, ticker)
            if key in by_ts_ticker:
                by_ts_ticker[key]["earnings_event_bar"] = 1.0
            for window in EVENT_WINDOW_BARS:
                for idx in range(event_index, min(len(timestamps), event_index + int(window) + 1)):
                    record = by_ts_ticker.get((timestamps[idx], ticker))
                    if record is not None:
                        record[f"post_earnings_{window}bars"] = 1.0

    for ts, tickers in tickers_at_ts.items():
        density = market_earnings_by_day.get(ts.date(), 0) / max(1, len(tickers))
        sector_counts: dict[float, int] = defaultdict(int)
        for ticker in tickers:
            sector_counts[ticker_sector.get(ticker, 0.0)] += 1
        for ticker in tickers:
            sector_id = ticker_sector.get(ticker, 0.0)
            by_ts_ticker[(ts, ticker)]["earnings_density_same_day_market"] = float(density)
            by_ts_ticker[(ts, ticker)]["earnings_density_same_day_sector"] = float(
                sector_earnings_by_day.get((ts.date(), sector_id), 0) / max(1, sector_counts.get(sector_id, 0))
            )

    for ticker, events in sec_item_events_by_ticker.items():
        events.sort(key=lambda item: item[0])
        for ts in timestamps:
            current_index = index_by_ts[ts]
            record = by_ts_ticker.get((ts, ticker))
            if record is None:
                continue
            for event_ts, item in events:
                event_index = index_by_ts.get(event_ts)
                if event_index is None or event_index > current_index:
                    continue
                delta = current_index - event_index
                if delta > 78:
                    continue
                col = SEC_ITEM_COLUMNS.get(item)
                if col is not None:
                    record[col] = max(record[col], math.exp(-float(delta) / 12.0))

    if earnings_calendar_pit_path or enable_pre_earnings_flags_without_pit:
        pit_records = _read_records(earnings_calendar_pit_path)
        if not pit_records and enable_pre_earnings_flags_without_pit:
            LOGGER.warning("pre_earnings_1d requested without a PIT earnings calendar; leaving zeros")
        pit_events = align_events_to_bars(pit_records, available_timestamps=timestamps, market_timezone=market_timezone)
        for event in pit_events:
            ticker = str(event.get("ticker", "")).upper()
            event_day = event["date"]
            for ts in timestamps:
                if ts.date() == event_day - timedelta(days=1):
                    record = by_ts_ticker.get((ts, ticker))
                    if record is not None:
                        record["pre_earnings_1d"] = 1.0

    return by_ts_ticker, columns


def attach_event_calendar_features(
    rows: list[dict[str, Any]],
    *,
    macro_calendar_path: str | Path | None,
    sec_8k_events_path: str | Path | None,
    earnings_calendar_pit_path: str | Path | None,
    enable_pre_earnings_flags_without_pit: bool,
    market_timezone: str = "America/New_York",
) -> list[str]:
    """Attach structured event flags in place. Returns new columns."""
    features, columns = build_event_window_flags(
        rows,
        macro_calendar_path=macro_calendar_path,
        sec_8k_events_path=sec_8k_events_path,
        earnings_calendar_pit_path=earnings_calendar_pit_path,
        enable_pre_earnings_flags_without_pit=enable_pre_earnings_flags_without_pit,
        market_timezone=market_timezone,
    )
    if not columns:
        return []
    for row in rows:
        ts = row.get("timestamp")
        if not isinstance(ts, datetime):
            for column in columns:
                row[column] = 0.0
            continue
        ticker = str(row.get("ticker", "")).upper()
        values = features.get((ts.replace(tzinfo=None), ticker))
        if values is None:
            values = {column: 0.0 for column in columns}
        for column in columns:
            row[column] = float(values.get(column, 0.0))
    return columns

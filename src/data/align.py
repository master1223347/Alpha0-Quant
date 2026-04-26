"""Session alignment utilities for OHLCV rows."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, TypedDict

from src.data.loader import OhlcvRow

if TYPE_CHECKING:
    import pandas as pd


REGULAR_SESSION_START = time(hour=9, minute=30)
REGULAR_SESSION_END = time(hour=16, minute=0)
CANDLE_INTERVAL = timedelta(minutes=5)


class SequenceBreak(TypedDict):
    previous_timestamp: datetime
    current_timestamp: datetime
    reason: str
    missing_candles: int


def is_regular_session_timestamp(timestamp: datetime) -> bool:
    """Return True when a timestamp falls inside regular market hours."""
    session_time = timestamp.time()
    return REGULAR_SESSION_START <= session_time < REGULAR_SESSION_END


def filter_regular_session(rows: list[OhlcvRow]) -> list[OhlcvRow]:
    """Drop rows outside the regular market session and sort by time."""
    return sorted(
        (row for row in rows if is_regular_session_timestamp(row["timestamp"])),
        key=lambda row: row["timestamp"],
    )


def _sequence_break(previous_timestamp: datetime, current_timestamp: datetime) -> SequenceBreak:
    if current_timestamp.date() != previous_timestamp.date():
        return {
            "previous_timestamp": previous_timestamp,
            "current_timestamp": current_timestamp,
            "reason": "new_trading_day",
            "missing_candles": 0,
        }

    gap = current_timestamp - previous_timestamp
    missing_candles = max(int(gap / CANDLE_INTERVAL) - 1, 0)
    return {
        "previous_timestamp": previous_timestamp,
        "current_timestamp": current_timestamp,
        "reason": "gap",
        "missing_candles": missing_candles,
    }


def split_contiguous_sequences(rows: list[OhlcvRow]) -> list[list[OhlcvRow]]:
    """Split OHLCV rows into contiguous 5-minute regular-session sequences."""
    regular_rows = filter_regular_session(rows)
    if not regular_rows:
        return []

    sequences: list[list[OhlcvRow]] = [[regular_rows[0]]]
    previous_timestamp = regular_rows[0]["timestamp"]

    for row in regular_rows[1:]:
        current_timestamp = row["timestamp"]
        same_day = current_timestamp.date() == previous_timestamp.date()
        is_contiguous = (current_timestamp - previous_timestamp) == CANDLE_INTERVAL
        if same_day and is_contiguous:
            sequences[-1].append(row)
        else:
            sequences.append([row])
        previous_timestamp = current_timestamp

    return sequences


def detect_sequence_breaks(rows: list[OhlcvRow]) -> list[SequenceBreak]:
    """Return explicit day-change and gap breaks after regular-session filtering."""
    regular_rows = filter_regular_session(rows)
    breaks: list[SequenceBreak] = []

    for previous_row, current_row in zip(regular_rows, regular_rows[1:]):
        previous_timestamp = previous_row["timestamp"]
        current_timestamp = current_row["timestamp"]
        same_day = current_timestamp.date() == previous_timestamp.date()
        is_contiguous = (current_timestamp - previous_timestamp) == CANDLE_INTERVAL
        if same_day and is_contiguous:
            continue
        breaks.append(_sequence_break(previous_timestamp, current_timestamp))

    return breaks


def _convert_timezone(rows: list[OhlcvRow], *, source_timezone: str, market_timezone: str) -> list[OhlcvRow]:
    if not rows:
        return rows
    source = source_timezone.strip()
    market = market_timezone.strip()
    if not source or not market or source == market:
        return rows

    try:
        from zoneinfo import ZoneInfo
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("zoneinfo is required for timezone conversion") from exc

    source_zone = ZoneInfo(source)
    market_zone = ZoneInfo(market)
    converted: list[OhlcvRow] = []
    for row in rows:
        timestamp = row["timestamp"]
        converted_timestamp = timestamp.replace(tzinfo=source_zone).astimezone(market_zone).replace(tzinfo=None)
        converted.append(
            {
                "timestamp": converted_timestamp,
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
    return converted


def align_ticker_rows(
    rows: list[OhlcvRow],
    *,
    source_timezone: str = "UTC",
    market_timezone: str = "America/New_York",
) -> list[list[OhlcvRow]]:
    """Filter to regular market session and split rows at day boundaries or gaps."""
    localized_rows = _convert_timezone(
        rows,
        source_timezone=source_timezone,
        market_timezone=market_timezone,
    )
    return split_contiguous_sequences(localized_rows)


def align_ticker_frames(rows: list[OhlcvRow]) -> list["pd.DataFrame"]:
    """Return contiguous aligned sequences as pandas DataFrames if pandas is installed."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for align_ticker_frames(); use align_ticker_rows() otherwise."
        ) from exc

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return [pd.DataFrame(sequence, columns=columns) for sequence in align_ticker_rows(rows)]

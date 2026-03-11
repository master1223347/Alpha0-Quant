"""OHLCV row validation and cleaning utilities."""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict

from src.data.loader import OhlcvRow

if TYPE_CHECKING:
    import pandas as pd


class InvalidRow(TypedDict):
    row: OhlcvRow
    reason: str


PRICE_FIELDS = ("open", "high", "low", "close")
REQUIRED_FIELDS = ("timestamp", "open", "high", "low", "close", "volume")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def validate_ohlcv_row(row: OhlcvRow) -> str | None:
    """Return None for a valid row, otherwise a short rejection reason."""
    for field in REQUIRED_FIELDS:
        if _is_missing(row.get(field)):
            return f"missing_{field}"

    timestamp = row["timestamp"]
    if not isinstance(timestamp, datetime):
        return "invalid_timestamp"

    for field in PRICE_FIELDS:
        price = row[field]
        if not isinstance(price, (int, float)) or math.isnan(price) or math.isinf(price):
            return f"invalid_{field}"
        if price <= 0:
            return f"non_positive_{field}"

    volume = row["volume"]
    if not isinstance(volume, (int, float)) or math.isnan(volume) or math.isinf(volume):
        return "invalid_volume"
    if volume < 0:
        return "negative_volume"

    if row["high"] < row["low"]:
        return "high_below_low"
    if row["high"] < row["open"]:
        return "high_below_open"
    if row["high"] < row["close"]:
        return "high_below_close"
    if row["low"] > row["open"]:
        return "low_above_open"
    if row["low"] > row["close"]:
        return "low_above_close"

    return None


def clean_ohlcv_rows(rows: list[OhlcvRow]) -> list[OhlcvRow]:
    """Drop rows that fail basic OHLCV validity checks."""
    cleaned_rows: list[OhlcvRow] = []
    for row in rows:
        if validate_ohlcv_row(row) is None:
            cleaned_rows.append(row)
    return cleaned_rows


def find_invalid_rows(rows: list[OhlcvRow]) -> list[InvalidRow]:
    """Return invalid rows together with their rejection reason."""
    invalid_rows: list[InvalidRow] = []
    for row in rows:
        reason = validate_ohlcv_row(row)
        if reason is not None:
            invalid_rows.append({"row": row, "reason": reason})
    return invalid_rows


def clean_ohlcv_frame(rows: list[OhlcvRow]) -> "pd.DataFrame":
    """Return cleaned rows as a pandas DataFrame if pandas is installed."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for clean_ohlcv_frame(); use clean_ohlcv_rows() otherwise."
        ) from exc

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return pd.DataFrame(clean_ohlcv_rows(rows), columns=columns)

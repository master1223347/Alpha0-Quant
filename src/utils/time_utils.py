"""Timestamp and session utility helpers."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any


def timestamps_are_sorted(rows: list[Mapping[str, Any]], *, strict: bool = True) -> bool:
    """Check whether rows are sorted by timestamp."""
    previous: datetime | None = None
    for row in rows:
        current = row["timestamp"]
        if previous is None:
            previous = current
            continue
        if strict and current <= previous:
            return False
        if not strict and current < previous:
            return False
        previous = current
    return True


def split_rows_by_day(rows: list[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    """Split rows into day-level chunks using timestamp date boundaries."""
    if not rows:
        return []

    days: list[list[Mapping[str, Any]]] = [[rows[0]]]
    previous_date = rows[0]["timestamp"].date()
    for row in rows[1:]:
        current_date = row["timestamp"].date()
        if current_date != previous_date:
            days.append([row])
        else:
            days[-1].append(row)
        previous_date = current_date
    return days


def iso_timestamp(value: datetime) -> str:
    """Render ISO-like second resolution timestamp string."""
    return value.strftime("%Y-%m-%d %H:%M:%S")

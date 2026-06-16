"""Synthetic liquidity-shock event detection from OHLCV-derived rows."""

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Any

from src.events.schema import EventRecord


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(var)


def detect_liquidity_events(rows: list[dict[str, Any]], *, window: int = 78, z_threshold: float = 4.0) -> list[EventRecord]:
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ticker.setdefault(str(row.get("ticker", "")).upper(), []).append(row)

    events: list[EventRecord] = []
    for ticker, ticker_rows in by_ticker.items():
        ordered = sorted(ticker_rows, key=lambda item: item.get("timestamp"))
        volumes: list[float] = []
        ranges: list[float] = []
        returns: list[float] = []
        prev_close: float | None = None
        for row in ordered:
            ts = row.get("timestamp")
            if not isinstance(ts, datetime):
                continue
            close = float(row.get("close", 0.0) or 0.0)
            high_low = abs(float(row.get("high_low_range", row.get("range", 0.0)) or 0.0))
            volume = float(row.get("relative_volume", row.get("volume", row.get("vol", 0.0))) or 0.0)
            ret = math.log(close / prev_close) if prev_close and close > 0 else 0.0
            checks = (("volume_spike", volume, volumes), ("volatility_spike", abs(ret), returns), ("range_spike", high_low, ranges))
            for subtype, value, history in checks:
                mean, std = _mean_std(history[-window:])
                z_score = (value - mean) / std if std > 1e-12 else 0.0
                if len(history) >= max(10, window // 4) and z_score >= float(z_threshold):
                    events.append(
                        EventRecord(
                            event_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"liquidity:{ticker}:{ts}:{subtype}")),
                            event_time_utc=ts,
                            event_date=ts.date().isoformat(),
                            ticker=ticker,
                            event_scope="ticker",
                            event_type="liquidity",
                            event_subtype=subtype,
                            source="synthetic_ohlcv",
                            title=f"{ticker} {subtype}",
                            importance_score=min(1.0, z_score / 10.0),
                            scheduled=False,
                            asof_timestamp_utc=ts,
                            metadata_json=f'{{"z_score": {z_score:.6f}}}',
                        )
                    )
            volumes.append(volume)
            ranges.append(high_low)
            returns.append(abs(ret))
            if close > 0:
                prev_close = close
    return events

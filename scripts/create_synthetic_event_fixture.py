"""Create a small synthetic OHLCV + event fixture for smoke testing."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


RAW_DIR = Path("data/events/synthetic/raw")
MANUAL_DIR = Path("data/events/manual")


def _session_times(start_day: datetime, days: int) -> list[datetime]:
    timestamps: list[datetime] = []
    for day_offset in range(days):
        day = start_day + timedelta(days=day_offset)
        current = day.replace(hour=9, minute=30, second=0, microsecond=0)
        for _ in range(78):
            timestamps.append(current)
            current += timedelta(minutes=5)
    return timestamps


def _write_ohlcv(ticker: str, base: float, shock_index: int) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{ticker}.US.csv"
    timestamps = _session_times(datetime(2024, 1, 2), 4)
    close = base
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<VOL>"])
        writer.writeheader()
        for index, ts in enumerate(timestamps):
            drift = 0.0003 * math.sin(index / 8.0)
            if index == shock_index:
                drift += 0.025
            if index == shock_index + 1:
                drift -= 0.012
            open_price = close
            close = max(1.0, close * (1.0 + drift))
            high = max(open_price, close) * 1.002
            low = min(open_price, close) * 0.998
            volume = 10000 + (60000 if index == shock_index else 1000 + (index % 20) * 50)
            writer.writerow(
                {
                    "<DATE>": ts.strftime("%Y%m%d"),
                    "<TIME>": ts.strftime("%H%M%S"),
                    "<OPEN>": f"{open_price:.4f}",
                    "<HIGH>": f"{high:.4f}",
                    "<LOW>": f"{low:.4f}",
                    "<CLOSE>": f"{close:.4f}",
                    "<VOL>": int(volume),
                }
            )


def _write_events() -> None:
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    event_rows = [
        {
            "event_time_utc": "2024-01-03T10:00:00",
            "ticker": "",
            "event_scope": "macro",
            "event_type": "macro",
            "event_subtype": "CPI",
            "title": "Synthetic CPI release",
            "importance_score": "0.9",
            "sentiment_score": "-0.2",
            "sentiment_label": "negative",
            "surprise_score": "0.4",
            "scheduled": "true",
            "asof_timestamp_utc": "2024-01-03T10:00:00",
        },
        {
            "event_time_utc": "2024-01-04T11:00:00",
            "ticker": "AAA",
            "event_scope": "ticker",
            "event_type": "sec_filing",
            "event_subtype": "8-K",
            "title": "AAA synthetic 8-K",
            "importance_score": "0.7",
            "sentiment_score": "-0.1",
            "sentiment_label": "negative",
            "scheduled": "false",
            "asof_timestamp_utc": "2024-01-04T11:00:00",
        },
    ]
    with (MANUAL_DIR / "events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows)

    earnings_rows = [
        {
            "event_time_utc": "2024-01-03T15:00:00",
            "ticker": "BBB",
            "event_type": "earnings",
            "event_subtype": "earnings",
            "title": "BBB synthetic earnings",
            "importance_score": "0.8",
            "sentiment_score": "0.3",
            "sentiment_label": "positive",
            "surprise_score": "0.2",
            "scheduled": "true",
            "asof_timestamp_utc": "2024-01-03T15:00:00",
            "metadata_json": '{"eps_actual": 1.2, "eps_estimate": 1.0}',
        }
    ]
    with (MANUAL_DIR / "earnings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(earnings_rows[0].keys()))
        writer.writeheader()
        writer.writerows(earnings_rows)

    news_rows = [
        {
            "event_time_utc": "2024-01-04T10:30:00",
            "ticker": "AAA",
            "event_type": "news",
            "event_subtype": "headline",
            "title": "AAA shares surge after strong update",
            "summary": "Synthetic positive headline",
            "importance_score": "0.6",
            "scheduled": "false",
            "asof_timestamp_utc": "2024-01-04T10:30:00",
        }
    ]
    with (MANUAL_DIR / "news.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(news_rows[0].keys()))
        writer.writeheader()
        writer.writerows(news_rows)

    macro_rows = [
        {
            "event_time_utc": "2024-01-02T14:00:00",
            "ticker": "",
            "event_type": "macro",
            "event_scope": "macro",
            "event_subtype": "FOMC",
            "title": "Synthetic FOMC minutes",
            "importance_score": "0.7",
            "scheduled": "true",
            "asof_timestamp_utc": "2024-01-02T14:00:00",
        }
    ]
    with (MANUAL_DIR / "macro_calendar.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(macro_rows[0].keys()))
        writer.writeheader()
        writer.writerows(macro_rows)


def main() -> None:
    _write_ohlcv("AAA", 100.0, 120)
    _write_ohlcv("BBB", 50.0, 150)
    _write_events()
    print("Synthetic event fixture written under data/events/")


if __name__ == "__main__":
    main()

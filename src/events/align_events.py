"""Leakage-aware event alignment to 5-minute bars."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any

from src.events.registry import EVENT_FEATURE_COLUMNS
from src.events.schema import EventRecord


EVENT_TYPES = ("macro", "earnings", "sec_filing", "news", "geopolitical", "liquidity")
EVENT_SCOPES = ("ticker", "sector", "market", "macro", "geopolitical")
EVENT_TYPE_ID = {name: index + 1 for index, name in enumerate(EVENT_TYPES)}
EVENT_SCOPE_ID = {name: index + 1 for index, name in enumerate(EVENT_SCOPES)}


def zero_event_features() -> dict[str, float]:
    values = {column: 0.0 for column in EVENT_FEATURE_COLUMNS}
    values["minutes_to_next_event"] = 1e6
    values["minutes_since_last_event"] = 1e6
    return values


def _row_ts(row: dict[str, Any]) -> datetime | None:
    value = row.get("timestamp")
    return value if isinstance(value, datetime) else None


def _applies(event: EventRecord, row: dict[str, Any], *, attach_macro_to_all_tickers: bool) -> bool:
    ticker = str(row.get("ticker", "")).upper()
    if event.event_scope == "ticker":
        return bool(event.ticker) and event.ticker.upper() == ticker
    if event.event_scope == "sector":
        event_sector = ""
        try:
            import json

            event_sector = str(json.loads(event.metadata_json or "{}").get("sector", ""))
        except Exception:
            event_sector = ""
        return bool(event_sector) and event_sector == str(row.get("sector") or row.get("mctx_sector_id") or "")
    if event.event_scope in {"market", "geopolitical"}:
        return True
    if event.event_scope == "macro":
        return bool(attach_macro_to_all_tickers)
    return False


def _bucket(minutes: float) -> float:
    if minutes < 0:
        return 0.0
    if minutes <= 15:
        return 1.0
    if minutes <= 30:
        return 2.0
    if minutes <= 60:
        return 3.0
    if minutes <= 390:
        return 4.0
    return 5.0


def _sentiment_parts(score: float) -> tuple[float, float, float]:
    if score > 0:
        return 0.0, 0.0, abs(score)
    if score < 0:
        return abs(score), 0.0, 0.0
    return 0.0, 1.0, 0.0


def align_events_to_feature_rows(
    rows: list[dict[str, Any]],
    events: list[EventRecord],
    *,
    pre_windows_minutes: tuple[int, ...] = (15, 30, 60),
    post_windows_minutes: tuple[int, ...] = (5, 15, 30, 60, 390),
    sentiment_decay_halflives_minutes: tuple[int, ...] = (30, 120),
    attach_macro_to_all_tickers: bool = True,
    enforce_asof: bool = True,
    drop_events_without_asof: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    if not rows:
        return [], {"events_total": len(events), "events_aligned": 0, "events_dropped": 0}

    dropped = 0
    usable_events: list[EventRecord] = []
    for event in events:
        if event.asof_timestamp_utc is None and drop_events_without_asof:
            dropped += 1
            continue
        usable_events.append(event)

    for row in rows:
        row.update(zero_event_features())

    aligned_count = 0
    suspicious_future_inclusions = 0
    max_pre_event_minutes = max(pre_windows_minutes or (0,))
    max_post_event_minutes = max(post_windows_minutes or (0,))
    for row in rows:
        ts = _row_ts(row)
        if ts is None:
            continue
        pre_known_events: list[EventRecord] = []
        future_scheduled_events: list[EventRecord] = []
        for event in usable_events:
            if not _applies(event, row, attach_macro_to_all_tickers=attach_macro_to_all_tickers):
                continue
            asof = event.asof_timestamp_utc or event.event_time_utc
            delta_minutes = (ts - event.event_time_utc).total_seconds() / 60.0
            in_pre = -max_pre_event_minutes <= delta_minutes < 0
            in_post = 0 < delta_minutes <= max_post_event_minutes
            in_bar = 0 <= delta_minutes < 5
            recent = 0 <= delta_minutes <= 1440
            if enforce_asof and asof > ts:
                if event.scheduled and event.event_time_utc > ts and in_pre:
                    future_scheduled_events.append(event)
                continue
            if in_pre or in_post or in_bar or recent:
                pre_known_events.append(event)
            elif delta_minutes < 0:
                suspicious_future_inclusions += 1
        if not pre_known_events and not future_scheduled_events:
            continue

        ticker_specific = [event for event in pre_known_events if event.event_scope == "ticker"]
        known_events = list(pre_known_events)
        pre_events = list(future_scheduled_events)
        nearest_next = min(((event.event_time_utc - ts).total_seconds() / 60.0 for event in pre_events if event.event_time_utc >= ts), default=1e6)
        nearest_last = min(((ts - event.event_time_utc).total_seconds() / 60.0 for event in known_events if event.event_time_utc <= ts), default=1e6)
        row["minutes_to_next_event"] = float(nearest_next)
        row["minutes_since_last_event"] = float(nearest_last)
        row["time_to_event_bucket"] = _bucket(float(nearest_next))
        row["time_since_event_bucket"] = _bucket(float(nearest_last))
        row["is_pre_event"] = 1.0 if pre_events else 0.0
        row["is_post_event"] = 1.0 if any(event.event_time_utc < ts and (ts - event.event_time_utc) <= timedelta(minutes=max_post_event_minutes) for event in known_events) else 0.0
        row["is_event_bar"] = 1.0 if any(0 <= (ts - event.event_time_utc).total_seconds() < 300 for event in known_events) else 0.0
        row["is_event_window"] = 1.0 if row["is_pre_event"] or row["is_post_event"] or row["is_event_bar"] else 0.0

        for window_name, minutes in (("30m", 30), ("1h", 60), ("1d", 1440)):
            recent_known = [
                event for event in known_events if 0 <= (ts - event.event_time_utc).total_seconds() / 60.0 <= minutes
            ]
            row[f"event_count_last_{window_name}"] = float(len(recent_known))
            if window_name == "1h":
                row["market_event_count_last_1h"] = float(sum(1 for event in recent_known if event.event_scope != "ticker"))
                row["ticker_event_count_last_1h"] = float(sum(1 for event in recent_known if event.event_scope == "ticker"))
                row["max_event_importance_last_1h"] = max((float(event.importance_score) for event in recent_known), default=0.0)
                sentiments = [float(event.sentiment_score) for event in recent_known]
                row["mean_event_sentiment_last_1h"] = sum(sentiments) / len(sentiments) if sentiments else 0.0
                row["sum_abs_event_sentiment_last_1h"] = sum(abs(value) for value in sentiments)

        active_events = [
            event for event in known_events if abs((ts - event.event_time_utc).total_seconds()) / 60.0 <= 60
        ]
        for event_type in EVENT_TYPES:
            column = f"{event_type}_event_active"
            if column in row:
                row[column] = 1.0 if any(event.event_type == event_type for event in active_events) else 0.0
        row["scheduled_event_active"] = 1.0 if any(event.scheduled for event in active_events) else 0.0
        row["unscheduled_event_active"] = 1.0 if any(not event.scheduled for event in active_events) else 0.0
        row["ticker_sec_filing_active"] = 1.0 if any(event.event_type == "sec_filing" for event in ticker_specific) else 0.0
        row["ticker_earnings_active"] = 1.0 if any(event.event_type == "earnings" for event in ticker_specific) else 0.0
        news_scores = [event.sentiment_score for event in known_events if event.event_type == "news"]
        row["ticker_news_sentiment_last_1h"] = sum(news_scores) / len(news_scores) if news_scores else 0.0
        row["macro_surprise_score"] = max((abs(event.surprise_score) for event in known_events if event.event_type == "macro"), default=0.0)
        geo_scores = [event.sentiment_score for event in known_events if event.event_type == "geopolitical"]
        row["geopolitical_tone_score"] = sum(geo_scores) / len(geo_scores) if geo_scores else 0.0

        for half_life in sentiment_decay_halflives_minutes:
            suffix = "30m" if int(half_life) == 30 else "2h" if int(half_life) == 120 else f"{int(half_life)}m"
            sentiment_sum = 0.0
            importance_sum = 0.0
            for event in known_events:
                age = max(0.0, (ts - event.event_time_utc).total_seconds() / 60.0)
                weight = math.exp(-math.log(2.0) * age / max(1.0, float(half_life)))
                sentiment_sum += float(event.sentiment_score) * weight
                importance_sum += float(event.importance_score) * weight
            row[f"event_sentiment_decay_{suffix}"] = sentiment_sum
            row[f"event_importance_decay_{suffix}"] = importance_sum

        most_important = max(active_events or pre_events or known_events, key=lambda event: float(event.importance_score), default=None)
        if most_important is not None:
            row["event_type_id"] = float(EVENT_TYPE_ID.get(most_important.event_type, 0))
            row["event_scope_id"] = float(EVENT_SCOPE_ID.get(most_important.event_scope, 0))
            row["event_scheduled_id"] = 1.0 if most_important.scheduled else 0.0

        news_recent = [event for event in known_events if event.event_type == "news"]
        if news_recent:
            latest_news = max(news_recent, key=lambda event: event.event_time_utc)
            neg, neu, pos = _sentiment_parts(float(latest_news.sentiment_score))
            row["news_sentiment_score"] = float(latest_news.sentiment_score)
            row["news_sentiment_confidence"] = abs(float(latest_news.sentiment_score))
            row["news_negative_score"] = neg
            row["news_neutral_score"] = neu
            row["news_positive_score"] = pos
            row["news_novelty_score"] = float(latest_news.novelty_score)
            row["news_cluster_size"] = float(len(news_recent))
            row["news_volume_spike"] = 1.0 if len(news_recent) >= 3 else 0.0
            row["news_tone_change_1h"] = row["mean_event_sentiment_last_1h"]
            row["news_tone_change_1d"] = row["mean_event_sentiment_last_1h"]

        aligned_count += 1

    return list(EVENT_FEATURE_COLUMNS), {
        "events_total": len(events),
        "events_usable": len(usable_events),
        "events_dropped": dropped,
        "rows_with_event_features": aligned_count,
        "suspicious_future_inclusions": suspicious_future_inclusions,
    }

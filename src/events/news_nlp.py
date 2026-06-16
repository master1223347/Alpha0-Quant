"""Optional news sentiment, novelty, and clustering helpers."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta

from src.events.schema import EventRecord


POSITIVE_WORDS = {"beat", "beats", "growth", "upgrade", "surge", "profit", "strong", "raises", "record"}
NEGATIVE_WORDS = {"miss", "misses", "lawsuit", "downgrade", "fall", "weak", "cuts", "probe", "loss"}


def lexicon_sentiment(text: str) -> tuple[float, str]:
    words = {part.strip(".,:;!?()[]{}\"'").lower() for part in text.split()}
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    if pos == neg:
        return 0.0, "neutral"
    score = (pos - neg) / max(1, pos + neg)
    return float(score), "positive" if score > 0 else "negative"


def enrich_news_events(events: list[EventRecord], *, novelty_window_minutes: int = 1440) -> dict[str, int]:
    recent_hashes: dict[str, deque[tuple[datetime, str]]] = defaultdict(deque)
    enriched = 0
    for event in sorted(events, key=lambda item: item.event_time_utc):
        if event.event_type != "news":
            continue
        text = f"{event.title} {event.summary}".strip()
        if event.sentiment_label == "unknown" and text:
            event.sentiment_score, event.sentiment_label = lexicon_sentiment(text)
        digest = hashlib.sha1(text.lower().encode("utf-8")).hexdigest() if text else ""
        key = event.ticker or "__market__"
        cutoff = event.event_time_utc - timedelta(minutes=int(novelty_window_minutes))
        while recent_hashes[key] and recent_hashes[key][0][0] < cutoff:
            recent_hashes[key].popleft()
        seen = {value for _, value in recent_hashes[key]}
        event.novelty_score = 0.0 if digest and digest in seen else 1.0
        if digest:
            recent_hashes[key].append((event.event_time_utc, digest))
        enriched += 1
    return {"news_events_enriched": enriched}

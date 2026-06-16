"""Event market-lab unit tests."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from src.events.align_events import align_events_to_feature_rows
from src.events.ingest_local_csv import ingest_local_events
from src.events.schema import EventRecord
from src.evaluation.backtest import run_backtest
from src.models.losses_prob import build_model_loss
from src.pipeline.train_model import build_model
from src.config.default_config import get_default_config


class EventMarketLabTest(unittest.TestCase):
    def test_local_csv_defaults_missing_asof_to_event_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["event_time_utc", "ticker", "event_type", "event_subtype", "title"])
                writer.writeheader()
                writer.writerow(
                    {
                        "event_time_utc": "2024-01-03T10:00:00",
                        "ticker": "AAA",
                        "event_type": "news",
                        "event_subtype": "headline",
                        "title": "AAA update",
                    }
                )
            events, report = ingest_local_events(path)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].asof_timestamp_utc, events[0].event_time_utc)
        self.assertTrue(report["warnings"])

    def test_alignment_blocks_unscheduled_future_news(self) -> None:
        rows = [
            {"timestamp": datetime(2024, 1, 3, 9, 55), "ticker": "AAA", "close": 10.0},
            {"timestamp": datetime(2024, 1, 3, 10, 0), "ticker": "AAA", "close": 10.1},
        ]
        events = [
            EventRecord(
                event_id="n1",
                event_time_utc=datetime(2024, 1, 3, 10, 0),
                event_date="2024-01-03",
                ticker="AAA",
                event_scope="ticker",
                event_type="news",
                event_subtype="headline",
                title="future news",
                scheduled=False,
                asof_timestamp_utc=datetime(2024, 1, 3, 10, 0),
            )
        ]
        align_events_to_feature_rows(rows, events, enforce_asof=True)
        self.assertEqual(rows[0]["news_event_active"], 0.0)
        self.assertEqual(rows[1]["news_event_active"], 1.0)

    def test_backtest_avoid_events_blocks_trade(self) -> None:
        report = run_backtest(
            probabilities=[0.9, 0.9],
            close=[10.0, 10.0],
            next_close=[10.1, 10.1],
            confidence_threshold=0.8,
            event_context={"is_event_window": [1.0, 0.0], "event_type_id": [4.0, 0.0]},
            event_mode="avoid_events",
        )
        self.assertGreaterEqual(report.trades_blocked_by_event_rules, 1)

    def test_event_model_and_loss_instantiate(self) -> None:
        config = get_default_config()
        config.model.include_event_heads = True
        config.training.event_opportunity_loss_weight = 0.1
        model = build_model(config, num_features=4, window_size=3)
        loss = build_model_loss(
            model,
            threshold_weight=0.0,
            regression_weight=0.0,
            event_opportunity_weight=0.1,
        )
        self.assertIsNotNone(loss)


if __name__ == "__main__":
    unittest.main()

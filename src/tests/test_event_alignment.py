"""Leakage-sensitive event calendar alignment tests."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from src.features.calendar_features import attach_event_calendar_features


class EventAlignmentTest(unittest.TestCase):
    def test_premarket_macro_and_earnings_snap_to_regular_open(self) -> None:
        rows = [
            {"timestamp": datetime(2024, 1, 3, 9, 30), "ticker": "ABC", "close": 10.0},
            {"timestamp": datetime(2024, 1, 3, 9, 35), "ticker": "ABC", "close": 10.1},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            macro_path = Path(temp_dir) / "macro.csv"
            sec_path = Path(temp_dir) / "sec.csv"
            with macro_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["timestamp", "event_type"])
                writer.writeheader()
                writer.writerow({"timestamp": "2024-01-03 08:30:00", "event_type": "cpi"})
            with sec_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["accepted_at", "ticker", "item"])
                writer.writeheader()
                writer.writerow({"accepted_at": "2024-01-03 07:40:00", "ticker": "ABC", "item": "2.02"})

            columns = attach_event_calendar_features(
                rows,
                macro_calendar_path=macro_path,
                sec_8k_events_path=sec_path,
                earnings_calendar_pit_path=None,
                enable_pre_earnings_flags_without_pit=False,
            )

        self.assertIn("macro_event_today", columns)
        self.assertEqual(rows[0]["macro_event_today"], 1.0)
        self.assertEqual(rows[0]["cpi_day"], 1.0)
        self.assertEqual(rows[0]["earnings_event_bar"], 1.0)
        self.assertEqual(rows[1]["earnings_event_bar"], 0.0)
        self.assertEqual(rows[1]["post_earnings_1bars"], 1.0)


if __name__ == "__main__":
    unittest.main()

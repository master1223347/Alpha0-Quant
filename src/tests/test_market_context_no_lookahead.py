"""Timestamp-purity tests for market-context features."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from src.features.market_context import compute_breadth_features


class MarketContextNoLookaheadTest(unittest.TestCase):
    def test_future_return_change_does_not_change_prior_breadth(self) -> None:
        start = datetime(2024, 1, 3, 9, 30)
        rows = []
        for idx in range(3):
            ts = start + timedelta(minutes=5 * idx)
            rows.extend(
                [
                    {"timestamp": ts, "ticker": "AAA", "log_return": 0.01, "relative_volume": 1.0, "close": 10.0 + idx},
                    {"timestamp": ts, "ticker": "BBB", "log_return": -0.02, "relative_volume": 1.0, "close": 20.0 - idx},
                ]
            )

        baseline, _ = compute_breadth_features(rows)
        mutated_rows = [dict(row) for row in rows]
        mutated_rows[-1]["log_return"] = 1.00
        mutated, _ = compute_breadth_features(mutated_rows)

        first_ts = start
        self.assertEqual(baseline[first_ts]["mctx_breadth_ad_share"], mutated[first_ts]["mctx_breadth_ad_share"])
        self.assertEqual(
            baseline[first_ts]["mctx_breadth_up_volume_share"],
            mutated[first_ts]["mctx_breadth_up_volume_share"],
        )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Fast synthetic checks for labeling, normalization, leakage, and backtest lag."""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.backtest import run_backtest
from src.features.base_features import build_base_features
from src.features.normalize import fit_feature_normalizer, transform_feature_rows
from src.features.volume_features import build_volume_features
from src.targets.labeling import label_sequence


REPORT_PATH = REPO_ROOT / "models" / "logs" / "tiny_pipeline_sanity.json"


def _make_ohlcv_rows(
    *,
    start: datetime,
    count: int,
    base_price: float = 100.0,
    base_volume: float = 1_000.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    price = base_price
    for index in range(count):
        timestamp = start + timedelta(minutes=index)
        drift = 1.0 + (0.002 if index % 2 == 0 else -0.0015) + (index * 0.0002)
        open_price = price
        close_price = price * drift
        high_price = max(open_price, close_price) * 1.01
        low_price = min(open_price, close_price) * 0.99
        volume = base_volume + (index * 37.0) + (15.0 if index % 3 == 0 else 0.0)
        rows.append(
            {
                "timestamp": timestamp,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            }
        )
        price = close_price
    return rows


def _check(name: str, passed: bool, details: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "details": details}


def _prefix_invariance(
    builder: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    *,
    prefix_lengths: list[int],
    feature_name: str,
) -> dict[str, Any]:
    full_features = builder(rows)
    comparisons: list[dict[str, Any]] = []
    for prefix_length in prefix_lengths:
        prefix_features = builder(rows[:prefix_length])
        expected = full_features[: len(prefix_features)]
        matched = prefix_features == expected
        comparisons.append(
            {
                "prefix_length": prefix_length,
                "feature_count": len(prefix_features),
                "matched": matched,
            }
        )
        if not matched:
            return _check(feature_name, False, {"comparisons": comparisons})
    return _check(feature_name, True, {"comparisons": comparisons})


def _lagged_positions(signals: list[int], lag_bars: int) -> list[int]:
    positions = [0] * len(signals)
    for index, signal in enumerate(signals):
        execution_index = index + lag_bars
        if execution_index < len(positions):
            positions[execution_index] = signal
    return positions


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _pairwise_overlaps(timestamp_sets: dict[str, set[Any]]) -> dict[str, int]:
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    return {
        f"{left}_{right}": len(timestamp_sets.get(left, set()).intersection(timestamp_sets.get(right, set())))
        for left, right in pairs
    }


def run_sanity_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    label_source_rows = _make_ohlcv_rows(start=datetime(2024, 1, 1, 9, 30), count=10)
    label_feature_rows = build_base_features(label_source_rows, momentum_lookback=3, volatility_window=5, relative_volume_window=5)
    for feature_row, source_row in zip(label_feature_rows, label_source_rows[5:]):
        feature_row["close"] = source_row["close"]
    label_rows = label_sequence(
        label_feature_rows,
        horizon=1,
        threshold=0.001,
        volatility_window=3,
        zscore_window=3,
    )
    label_mismatches = [
        {
            "timestamp": row["timestamp"],
            "label": int(row["label"]),
            "indicator": int(float(row["next_log_return"]) > 0.0),
        }
        for row in label_rows
        if int(row["label"]) != int(float(row["next_log_return"]) > 0.0)
    ]
    checks.append(
        _check(
            "label_alignment",
            not label_mismatches,
            {"rows_checked": len(label_rows), "mismatches": label_mismatches},
        )
    )

    train_rows = _make_ohlcv_rows(start=datetime(2024, 1, 2, 9, 30), count=14, base_price=120.0, base_volume=2_000.0)
    val_rows = _make_ohlcv_rows(start=datetime(2024, 1, 3, 9, 30), count=5, base_price=140.0, base_volume=1_400.0)
    test_rows = _make_ohlcv_rows(start=datetime(2024, 1, 4, 9, 30), count=5, base_price=160.0, base_volume=1_600.0)

    train_feature_rows = build_base_features(train_rows, momentum_lookback=3, volatility_window=5, relative_volume_window=5)
    normalizer = fit_feature_normalizer(train_feature_rows)
    transformed_train = transform_feature_rows(train_feature_rows, normalizer)
    train_means = {
        column: _mean([float(row[column]) for row in transformed_train])
        for column in normalizer.feature_columns
    }
    expected_train_means = {
        column: _mean([float(row[column]) for row in train_feature_rows])
        for column in normalizer.feature_columns
    }
    checks.append(
        _check(
            "normalization_train_only_fit",
            all(abs(normalizer.means[column] - expected_train_means[column]) < 1e-12 for column in normalizer.feature_columns),
            {
                "expected_train_means": expected_train_means,
                "normalizer_means": normalizer.means,
            },
        )
    )
    checks.append(
        _check(
            "normalization_train_moments",
            all(abs(value) < 1e-9 for value in train_means.values()),
            {"train_means": train_means},
        )
    )

    split_sequences = {
        "train": {
            "AAA": [[{"timestamp": row["timestamp"], "x": float(index)} for index, row in enumerate(train_rows)]],
        },
        "val": {
            "AAA": [[{"timestamp": row["timestamp"], "x": float(index)} for index, row in enumerate(val_rows)]],
        },
        "test": {
            "AAA": [[{"timestamp": row["timestamp"], "x": float(index)} for index, row in enumerate(test_rows)]],
        },
    }
    timestamp_sets = {
        split: {row["timestamp"] for sequences in ticker_map.values() for sequence in sequences for row in sequence}
        for split, ticker_map in split_sequences.items()
    }
    overlaps = _pairwise_overlaps(timestamp_sets)
    time_order_ok = bool(timestamp_sets["train"] and timestamp_sets["val"] and timestamp_sets["test"])
    time_order_ok = time_order_ok and max(timestamp_sets["train"]) < min(timestamp_sets["val"]) < min(timestamp_sets["test"])
    checks.append(
        _check(
            "global_time_no_overlap",
            all(count == 0 for count in overlaps.values()) and time_order_ok,
            {"overlap_counts": overlaps},
        )
    )

    base_rows = _make_ohlcv_rows(start=datetime(2024, 1, 5, 9, 30), count=18, base_price=90.0, base_volume=750.0)
    checks.append(
        _prefix_invariance(
            lambda rows: build_base_features(rows, momentum_lookback=3, volatility_window=5, relative_volume_window=5),
            base_rows,
            prefix_lengths=[8, 12, 18],
            feature_name="base_features_prefix_invariance",
        )
    )
    checks.append(
        _prefix_invariance(
            lambda rows: build_volume_features(rows, window=6),
            base_rows,
            prefix_lengths=[8, 12, 18],
            feature_name="volume_features_prefix_invariance",
        )
    )

    toy_probabilities = [0.9, 0.1, 0.7, 0.3, 0.8]
    toy_close = [100.0, 100.0, 100.0, 100.0, 100.0]
    toy_next_close = [110.0, 90.0, 120.0, 80.0, 130.0]
    lag_bars = 2
    raw_signals = [1 if probability >= 0.55 else -1 if probability <= 0.45 else 0 for probability in toy_probabilities]
    expected_positions = _lagged_positions(raw_signals, lag_bars)
    expected_gross_returns = [
        position * ((future_close - current_close) / current_close)
        for position, current_close, future_close in zip(expected_positions, toy_close, toy_next_close)
    ]
    expected_turnovers = [abs(current - previous) for previous, current in zip([0] + expected_positions[:-1], expected_positions)]
    expected_transaction_cost_pnl = -(25.0 / 10000.0) * sum(expected_turnovers)
    expected_slippage_pnl = -(15.0 / 10000.0) * sum(expected_turnovers)
    expected_pnl = sum(expected_gross_returns) + expected_transaction_cost_pnl + expected_slippage_pnl
    backtest = run_backtest(
        toy_probabilities,
        toy_close,
        toy_next_close,
        long_threshold=0.55,
        short_threshold=0.45,
        execution_lag_bars=lag_bars,
        cost_bps_per_trade=25.0,
        slippage_bps=15.0,
    )
    checks.append(
        _check(
            "backtest_execution_lag",
            all(
                [
                    backtest.trade_count == sum(1 for turnover in expected_turnovers if turnover > 0),
                    math.isclose(backtest.gross_pnl, sum(expected_gross_returns), rel_tol=0.0, abs_tol=1e-12),
                    math.isclose(backtest.transaction_cost_pnl, expected_transaction_cost_pnl, rel_tol=0.0, abs_tol=1e-12),
                    math.isclose(backtest.slippage_pnl, expected_slippage_pnl, rel_tol=0.0, abs_tol=1e-12),
                    math.isclose(backtest.pnl, expected_pnl, rel_tol=0.0, abs_tol=1e-12),
                    math.isclose(
                        (backtest.equity_curve or [1.0])[-1],
                        math.prod([1.0 + value - ((25.0 + 15.0) / 10000.0) * turnover for value, turnover in zip(expected_gross_returns, expected_turnovers)]),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ),
                ]
            ),
            {
                "expected_positions": expected_positions,
                "expected_gross_pnl": sum(expected_gross_returns),
                "expected_trade_count": sum(1 for turnover in expected_turnovers if turnover > 0),
                "expected_transaction_cost_pnl": expected_transaction_cost_pnl,
                "expected_slippage_pnl": expected_slippage_pnl,
                "expected_pnl": expected_pnl,
                "actual": {
                    "pnl": backtest.pnl,
                    "gross_pnl": backtest.gross_pnl,
                    "trade_count": backtest.trade_count,
                    "transaction_cost_pnl": backtest.transaction_cost_pnl,
                    "slippage_pnl": backtest.slippage_pnl,
                    "equity_curve": backtest.equity_curve,
                },
            },
        )
    )

    summary = {
        "checks_total": len(checks),
        "checks_passed": sum(1 for check in checks if check["passed"]),
        "checks_failed": sum(1 for check in checks if not check["passed"]),
        "all_passed": all(check["passed"] for check in checks),
    }
    return checks, summary


def main() -> int:
    checks, summary = run_sanity_checks()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "summary": summary,
        "checks": checks,
    }
    with REPORT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    print(json.dumps({"report_path": str(REPORT_PATH), "summary": summary}, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

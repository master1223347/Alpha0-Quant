from datetime import datetime, timedelta

from src.config.default_config import get_default_config
from src.pipeline.build_features import _merge_feature_rows


def _rows(count: int) -> list[dict]:
    start = datetime(2026, 1, 5, 14, 30)
    return [
        {
            "timestamp": start + timedelta(minutes=5 * index),
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.5 + index,
            "volume": 1000.0 + index,
        }
        for index in range(count)
    ]


def test_short_sequence_without_realized_vol_is_dropped() -> None:
    config = get_default_config()
    config.features.use_realized_volatility = True
    assert _merge_feature_rows(_rows(12), config) == []


def test_merged_rows_always_carry_realized_vol_columns() -> None:
    config = get_default_config()
    config.features.use_realized_volatility = True
    merged = _merge_feature_rows(_rows(40), config)
    assert merged
    assert all("rv_close_to_close" in row for row in merged)

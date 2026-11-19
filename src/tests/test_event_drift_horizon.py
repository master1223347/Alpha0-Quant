from datetime import datetime, timedelta

import pytest

from src.config.default_config import get_default_config
from src.dataset.panel_dataset import build_panel_dataset
from src.dataset.window_dataset import build_labeled_windows
from src.pipeline.build_dataset import _resolve_event_drift_key


def _sequences():
    start = datetime(2024, 1, 2, 14, 30)
    rows = [
        {
            "timestamp": start + timedelta(minutes=5 * index),
            "ticker": "AAA",
            "f": float(index),
            "label": 1.0,
            "close": 100.0,
            "next_close": 101.0,
            "post_event_drift_label_h1": 1,
            "post_event_drift_label_h3": 2,
        }
        for index in range(4)
    ]
    return {"AAA": [rows]}


def test_builders_use_configured_drift_label():
    panel = build_panel_dataset(_sequences(), context_size=1, feature_columns=["f"], event_drift_key="post_event_drift_label_h3")
    windows = build_labeled_windows(
        _sequences(), window_size=2, feature_columns=["f"], event_drift_key="post_event_drift_label_h3"
    )
    assert panel.event_drift_label.tolist() == [2, 2, 2, 2]
    assert set(windows.event_drift_label.tolist()) == {2}
    default_panel = build_panel_dataset(_sequences(), context_size=1, feature_columns=["f"])
    assert default_panel.event_drift_label.tolist() == [1, 1, 1, 1]


def test_resolve_event_drift_key():
    config = get_default_config()
    assert _resolve_event_drift_key(config) == "post_event_drift_label_h1"
    config.targets.event_drift_horizon = 6
    assert _resolve_event_drift_key(config) == "post_event_drift_label_h6"
    config.targets.event_drift_horizon = 5
    _resolve_event_drift_key(config)  # drift loss disabled: no validation
    config.training.event_drift_loss_weight = 0.2
    with pytest.raises(ValueError):
        _resolve_event_drift_key(config)
    config.targets.event_drift_horizon = 1
    config.targets.post_event_horizons = (3, 6)
    with pytest.raises(ValueError):
        _resolve_event_drift_key(config)

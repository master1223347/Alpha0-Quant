from src.config.default_config import get_default_config
from src.pipeline.build_dataset import _resolve_purge_bars


def test_default_purge_covers_longest_label_lookahead():
    config = get_default_config()
    config.targets.post_event_horizons = (1, 3, 6, 12)
    config.targets.event_reversal_horizon = 6
    config.targets.event_absorption_horizon = 6
    assert _resolve_purge_bars(config, label_horizon=1) == 12
    assert _resolve_purge_bars(config, label_horizon=20) == 20


def test_explicit_purge_overrides_default():
    config = get_default_config()
    config.dataset.purge_bars = 0
    assert _resolve_purge_bars(config, label_horizon=12) == 0
    config.dataset.purge_bars = 3
    assert _resolve_purge_bars(config, label_horizon=12) == 3

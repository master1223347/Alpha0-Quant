import math

import pytest

from src.targets.labeling import assign_cross_sectional_rank, label_sequence


def _sequence(closes):
    rows = []
    previous = closes[0]
    for index, close in enumerate(closes):
        rows.append({"timestamp": index, "close": close, "log_return": math.log(close / previous)})
        previous = close
    return rows


def test_next_log_return_and_threshold_labels():
    labeled = label_sequence(_sequence([100.0, 101.0, 100.0, 100.0]), horizon=1, threshold=0.001)
    assert len(labeled) == 3
    assert labeled[0]["next_log_return"] == pytest.approx(math.log(1.01))
    assert [row["threshold_label"] for row in labeled] == [2, 0, 1]
    assert [row["label"] for row in labeled] == [1, 0, 0]


def test_horizon_drops_tail_rows_and_uses_future_close():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    labeled = label_sequence(_sequence(closes), horizon=2)
    assert len(labeled) == 3
    assert labeled[0]["next_close"] == 102.0
    assert label_sequence(_sequence(closes[:2]), horizon=2) == []


def test_volatility_threshold_uses_only_past_returns():
    base = [100.0, 101.0, 99.0, 100.5, 100.0, 100.2]
    changed_future = base[:4] + [150.0, 80.0]
    original = label_sequence(_sequence(base), horizon=1, volatility_window=3)
    perturbed = label_sequence(_sequence(changed_future), horizon=1, volatility_window=3)
    for index in range(3):
        assert original[index]["vol_threshold"] == pytest.approx(perturbed[index]["vol_threshold"])
        assert original[index]["event_threshold"] == pytest.approx(perturbed[index]["event_threshold"])


def test_input_rows_are_not_mutated():
    sequence = _sequence([100.0, 101.0, 102.0])
    label_sequence(sequence, horizon=1)
    assert "next_log_return" not in sequence[0]


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"threshold": -1.0}, {"volatility_window": 0}])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        label_sequence(_sequence([100.0, 101.0, 102.0]), **kwargs)


def test_cross_sectional_rank_per_timestamp():
    ticker_sequences = {
        "AAA": [[{"timestamp": 0, "ticker": "AAA", "next_log_return": 0.02}]],
        "BBB": [[{"timestamp": 0, "ticker": "BBB", "next_log_return": -0.01}]],
        "CCC": [[{"timestamp": 0, "ticker": "CCC", "next_log_return": 0.00},
                 {"timestamp": 1, "ticker": "CCC", "next_log_return": 0.05}]],
    }
    assign_cross_sectional_rank(ticker_sequences)
    assert ticker_sequences["BBB"][0][0]["cross_sectional_rank"] == 0.0
    assert ticker_sequences["CCC"][0][0]["cross_sectional_rank"] == 0.5
    assert ticker_sequences["AAA"][0][0]["cross_sectional_rank"] == 1.0
    assert ticker_sequences["CCC"][0][1]["cross_sectional_rank"] == 0.5


def test_unobserved_event_drift_is_marked_ignored():
    from src.targets.labeling import DRIFT_LABEL_UNOBSERVED

    rows = _sequence([100.0, 101.0, 102.0, 103.0, 104.0])
    for row in rows:
        row["is_event_window"] = 1.0
    labeled = label_sequence(rows, horizon=1, post_event_horizons=(1, 3))
    assert [row["post_event_drift_label_h3"] == DRIFT_LABEL_UNOBSERVED for row in labeled] == [
        False,
        False,
        True,
        True,
    ]
    assert all(row["post_event_drift_label_h1"] != DRIFT_LABEL_UNOBSERVED for row in labeled)


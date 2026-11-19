import pytest

from src.dataset.sampler import flatten_sequences, limit_tickers, split_ticker_sequences


def _ticker_sequences(num_tickers=3, length=20):
    return {
        f"T{ticker}": [[{"timestamp": step, "ticker": f"T{ticker}"} for step in range(length)]]
        for ticker in range(num_tickers)
    }


def _timestamps(split_map):
    return {
        split: {row["timestamp"] for sequences in tickers.values() for row in flatten_sequences(sequences)}
        for split, tickers in split_map.items()
    }


@pytest.mark.parametrize("mode", ["global_time", "per_ticker"])
def test_splits_are_chronological_and_disjoint(mode):
    data = _ticker_sequences(num_tickers=2, length=10)
    if mode == "per_ticker":
        # per_ticker splits whole sequences, so give each bar its own sequence.
        data = {ticker: [[row] for row in sequences[0]] for ticker, sequences in data.items()}
    timestamps = _timestamps(split_ticker_sequences(data, split_mode=mode))
    assert timestamps["train"] and timestamps["val"] and timestamps["test"]
    assert max(timestamps["train"]) < min(timestamps["val"])
    assert max(timestamps["val"]) < min(timestamps["test"])


def test_global_time_split_ratios():
    timestamps = _timestamps(split_ticker_sequences(_ticker_sequences(), split_mode="global_time"))
    assert (len(timestamps["train"]), len(timestamps["val"]), len(timestamps["test"])) == (14, 3, 3)


def test_invalid_split_arguments():
    with pytest.raises(ValueError):
        split_ticker_sequences(_ticker_sequences(), train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)
    with pytest.raises(ValueError):
        split_ticker_sequences(_ticker_sequences(), split_mode="random")


def test_limit_tickers_is_deterministic():
    data = _ticker_sequences(num_tickers=10, length=1)
    first = limit_tickers(data, max_tickers=4, seed=1)
    second = limit_tickers(data, max_tickers=4, seed=1)
    assert len(first) == 4
    assert first.keys() == second.keys()
    assert limit_tickers(data, max_tickers=None) is data


def test_global_time_purge_drops_bars_before_each_boundary():
    data = _ticker_sequences(num_tickers=2, length=20)
    splits = split_ticker_sequences(data, split_mode="global_time", purge_bars=2)
    timestamps = _timestamps(splits)
    assert timestamps["train"] == set(range(12))
    assert timestamps["val"] == {14}
    assert timestamps["test"] == {17, 18, 19}
    for split in ("train", "val"):
        last = max(timestamps[split])
        next_start = min(timestamps["val" if split == "train" else "test"])
        assert next_start - last > 2


def test_purge_covers_sequences_ending_before_boundary():
    # Labeling trims a sequence's tail bars, so a labeled sequence ending just before the boundary
    # can still have targets priced in the next split.
    data = {"T0": [[{"timestamp": s} for s in range(14)], [{"timestamp": s} for s in range(14, 20)]]}
    timestamps = _timestamps(split_ticker_sequences(data, split_mode="global_time", purge_bars=2))
    assert timestamps["train"] == set(range(12))
    assert timestamps["val"] == {14}


def test_negative_purge_raises():
    with pytest.raises(ValueError):
        split_ticker_sequences(_ticker_sequences(), split_mode="global_time", purge_bars=-1)


def test_purge_skips_boundaries_without_later_split():
    data = _ticker_sequences(num_tickers=1, length=20)
    timestamps = _timestamps(
        split_ticker_sequences(
            data, split_mode="global_time", train_ratio=0.8, val_ratio=0.2, test_ratio=0.0, purge_bars=2
        )
    )
    assert timestamps["train"] == set(range(14))
    assert timestamps["val"] == set(range(16, 20))
    assert timestamps["test"] == set()

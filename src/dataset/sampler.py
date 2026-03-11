"""Sampling and split helpers for feature sequences."""

from __future__ import annotations

import random
from typing import Any


SplitMap = dict[str, dict[str, list[list[dict[str, Any]]]]]


def limit_tickers(
    ticker_sequences: dict[str, list[list[dict[str, Any]]]],
    *,
    max_tickers: int | None,
    seed: int = 42,
) -> dict[str, list[list[dict[str, Any]]]]:
    """Deterministically keep a subset of tickers for fast experiments."""
    if max_tickers is None or max_tickers <= 0 or len(ticker_sequences) <= max_tickers:
        return ticker_sequences

    tickers = sorted(ticker_sequences.keys())
    random.Random(seed).shuffle(tickers)
    selected = set(tickers[:max_tickers])
    return {ticker: sequences for ticker, sequences in ticker_sequences.items() if ticker in selected}


def split_ticker_sequences(
    ticker_sequences: dict[str, list[list[dict[str, Any]]]],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitMap:
    """Split each ticker's sequences chronologically by cumulative sequence length."""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    splits: SplitMap = {"train": {}, "val": {}, "test": {}}
    for ticker, sequences in ticker_sequences.items():
        sorted_sequences = sorted(sequences, key=lambda seq: seq[0]["timestamp"] if seq else 0)
        total_rows = sum(len(sequence) for sequence in sorted_sequences)
        train_target = int(total_rows * train_ratio)
        val_target = int(total_rows * (train_ratio + val_ratio))

        cumulative = 0
        train_sequences: list[list[dict[str, Any]]] = []
        val_sequences: list[list[dict[str, Any]]] = []
        test_sequences: list[list[dict[str, Any]]] = []

        for sequence in sorted_sequences:
            if not sequence:
                continue
            if cumulative < train_target:
                train_sequences.append(sequence)
            elif cumulative < val_target:
                val_sequences.append(sequence)
            else:
                test_sequences.append(sequence)
            cumulative += len(sequence)

        if train_sequences:
            splits["train"][ticker] = train_sequences
        if val_sequences:
            splits["val"][ticker] = val_sequences
        if test_sequences:
            splits["test"][ticker] = test_sequences

    return splits


def flatten_sequences(sequences: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten list-of-sequences into a single row list."""
    flattened: list[dict[str, Any]] = []
    for sequence in sequences:
        flattened.extend(sequence)
    return flattened

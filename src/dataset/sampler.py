"""Sampling and split helpers for feature sequences."""

from __future__ import annotations

import random
from collections import defaultdict
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
    split_mode: str = "per_ticker",
) -> SplitMap:
    """Split ticker sequences either per ticker or by global timestamp."""
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"split ratios must sum to 1.0, got {ratio_sum}")

    mode = split_mode.lower().strip()
    if mode not in {"per_ticker", "global_time"}:
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    if mode == "per_ticker":
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

    timestamp_to_split = _build_global_timestamp_split_map(ticker_sequences, train_ratio, val_ratio)
    return _split_by_timestamp(ticker_sequences, timestamp_to_split)


def _build_global_timestamp_split_map(
    ticker_sequences: dict[str, list[list[dict[str, Any]]]],
    train_ratio: float,
    val_ratio: float,
) -> dict[Any, str]:
    timestamps: list[Any] = []
    seen: set[Any] = set()
    for sequences in ticker_sequences.values():
        for sequence in sequences:
            for row in sequence:
                timestamp = row.get("timestamp")
                if timestamp is None:
                    raise ValueError("Global timestamp split requires timestamp values")
                if timestamp not in seen:
                    seen.add(timestamp)
                    timestamps.append(timestamp)

    timestamps.sort()
    total_timestamps = len(timestamps)
    if total_timestamps == 0:
        return {}

    train_target = int(total_timestamps * train_ratio)
    val_target = int(total_timestamps * (train_ratio + val_ratio))

    timestamp_to_split: dict[Any, str] = {}
    for index, timestamp in enumerate(timestamps):
        if index < train_target:
            split_name = "train"
        elif index < val_target:
            split_name = "val"
        else:
            split_name = "test"
        timestamp_to_split[timestamp] = split_name

    return timestamp_to_split


def _split_by_timestamp(
    ticker_sequences: dict[str, list[list[dict[str, Any]]]],
    timestamp_to_split: dict[Any, str],
) -> SplitMap:
    splits: SplitMap = {"train": {}, "val": {}, "test": {}}

    for ticker, sequences in ticker_sequences.items():
        split_sequences: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
        for sequence in sorted(sequences, key=lambda seq: seq[0]["timestamp"] if seq else 0):
            if not sequence:
                continue

            current_split: str | None = None
            current_split_sequence: list[dict[str, Any]] = []
            for row in sequence:
                timestamp = row.get("timestamp")
                if timestamp is None:
                    raise ValueError("Global timestamp split requires timestamp values")
                split_name = timestamp_to_split.get(timestamp)
                if split_name is None:
                    continue

                if current_split is None:
                    current_split = split_name
                if split_name != current_split:
                    if current_split_sequence:
                        split_sequences[current_split].append(current_split_sequence)
                    current_split = split_name
                    current_split_sequence = []

                current_split_sequence.append(row)

            if current_split_sequence and current_split is not None:
                split_sequences[current_split].append(current_split_sequence)

        for split_name, split_sequence_list in split_sequences.items():
            if split_sequence_list:
                splits[split_name][ticker] = split_sequence_list

    return splits


def flatten_sequences(sequences: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Flatten list-of-sequences into a single row list."""
    flattened: list[dict[str, Any]] = []
    for sequence in sequences:
        flattened.extend(sequence)
    return flattened

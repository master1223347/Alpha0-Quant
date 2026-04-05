"""Sliding window dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


RESERVED_COLUMNS = {"timestamp", "ticker", "label", "close", "next_close"}


@dataclass(slots=True)
class WindowDatasetArtifacts:
    X: "np.ndarray"
    y: "np.ndarray"
    close: "np.ndarray"
    next_close: "np.ndarray"
    timestamps: list[datetime]
    tickers: list[str]
    feature_columns: list[str]

    def __len__(self) -> int:
        return int(self.y.shape[0])


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("numpy is required for window dataset construction") from exc
    return np


def infer_feature_columns(rows: list[dict[str, Any]]) -> list[str]:
    """Infer numeric feature columns from one set of feature rows."""
    if not rows:
        return []
    columns: list[str] = []
    for key, value in rows[0].items():
        if key in RESERVED_COLUMNS:
            continue
        if isinstance(value, (int, float)):
            columns.append(key)
    return sorted(columns)


def build_labeled_windows(
    ticker_feature_sequences: dict[str, list[list[dict[str, Any]]]],
    *,
    window_size: int = 32,
    stride: int = 1,
    feature_columns: list[str] | None = None,
    label_key: str = "label",
) -> WindowDatasetArtifacts:
    """Build sliding windows and next-candle binary labels from feature sequences."""
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")

    np = _require_numpy()

    first_rows = []
    for sequences in ticker_feature_sequences.values():
        for sequence in sequences:
            if sequence:
                first_rows = sequence
                break
        if first_rows:
            break

    columns = feature_columns if feature_columns is not None else infer_feature_columns(first_rows)
    if not columns:
        raise ValueError("No feature columns available for window construction")

    X_values: list[list[list[float]]] = []
    y_values: list[int] = []
    close_values: list[float] = []
    next_close_values: list[float] = []
    timestamps: list[datetime] = []
    tickers: list[str] = []

    for ticker, sequences in sorted(ticker_feature_sequences.items()):
        for sequence in sequences:
            if len(sequence) < window_size:
                continue

            for end_index in range(window_size - 1, len(sequence), stride):
                target_row = sequence[end_index]
                if label_key not in target_row:
                    continue

                window_rows = sequence[end_index - window_size + 1 : end_index + 1]
                window_features = [[float(row[column]) for column in columns] for row in window_rows]

                X_values.append(window_features)
                y_values.append(int(target_row[label_key]))
                close_values.append(float(target_row["close"]))
                next_close_values.append(float(target_row["next_close"]))
                timestamps.append(target_row["timestamp"])
                tickers.append(ticker)

    if not X_values:
        empty_x = np.zeros((0, window_size, len(columns)), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.int64)
        empty_close = np.zeros((0,), dtype=np.float32)
        return WindowDatasetArtifacts(
            X=empty_x,
            y=empty_y,
            close=empty_close,
            next_close=empty_close.copy(),
            timestamps=[],
            tickers=[],
            feature_columns=columns,
        )

    return WindowDatasetArtifacts(
        X=np.asarray(X_values, dtype=np.float32),
        y=np.asarray(y_values, dtype=np.int64),
        close=np.asarray(close_values, dtype=np.float32),
        next_close=np.asarray(next_close_values, dtype=np.float32),
        timestamps=timestamps,
        tickers=tickers,
        feature_columns=columns,
    )


class WindowTensorDataset:
    """Torch dataset wrapper around window artifacts."""

    def __init__(self, artifacts: WindowDatasetArtifacts) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch is required for WindowTensorDataset") from exc

        self._torch = torch
        self.X = torch.tensor(artifacts.X, dtype=torch.float32)
        self.y = torch.tensor(artifacts.y, dtype=torch.float32)
        self.close = torch.tensor(artifacts.close, dtype=torch.float32)
        self.next_close = torch.tensor(artifacts.next_close, dtype=torch.float32)
        self.timestamps = artifacts.timestamps
        self.tickers = artifacts.tickers
        self.feature_columns = artifacts.feature_columns

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "X": self.X[index],
            "y": self.y[index],
            "close": self.close[index],
            "next_close": self.next_close[index],
        }

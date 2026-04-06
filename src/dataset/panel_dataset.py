"""Panel dataset construction for synchronized same-timestamp samples."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


RESERVED_COLUMNS = {
    "timestamp",
    "ticker",
    "label",
    "close",
    "next_close",
    "next_log_return",
    "vol_target",
    "vol_target_clipped",
    "vol_threshold",
    "z_return",
    "threshold_up",
    "threshold_down",
    "threshold_no_move",
    "threshold_label",
    "vol_direction_up",
    "vol_direction_down",
    "vol_direction_neutral",
    "vol_direction_label",
    "cross_sectional_rank",
}


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("numpy is required for panel dataset construction") from exc
    return np


def infer_feature_columns(rows: list[dict[str, Any]], *, exclude: set[str] | None = None) -> list[str]:
    """Infer numeric feature columns from panel rows."""
    if not rows:
        return []
    excluded = RESERVED_COLUMNS.copy()
    if exclude is not None:
        excluded.update(exclude)

    columns: list[str] = []
    for key, value in rows[0].items():
        if key in excluded:
            continue
        if isinstance(value, (int, float)):
            columns.append(key)
    return sorted(columns)


@dataclass(slots=True)
class PanelDatasetArtifacts:
    """Tensor-backed panel samples aligned by timestamp."""

    X: "np.ndarray"
    y: "np.ndarray"
    close: "np.ndarray"
    next_close: "np.ndarray"
    timestamps: list[datetime]
    tickers: list[str]
    feature_columns: list[str]
    context_size: int
    target_return: "np.ndarray | None" = None
    direction_label: "np.ndarray | None" = None
    threshold_label: "np.ndarray | None" = None
    rank_target: "np.ndarray | None" = None

    def __len__(self) -> int:
        return int(self.y.shape[0])


def _pad_feature_vector(length: int, fill_value: float) -> list[float]:
    return [float(fill_value)] * length


def build_panel_dataset(
    ticker_feature_sequences: dict[str, list[list[dict[str, Any]]]],
    *,
    context_size: int = 32,
    feature_columns: list[str] | None = None,
    label_key: str = "label",
    return_key: str = "next_log_return",
    direction_key: str = "label",
    threshold_key: str = "threshold_label",
    rank_key: str = "cross_sectional_rank",
    fill_value: float = 0.0,
) -> PanelDatasetArtifacts:
    """Build synchronized panel samples with same-timestamp neighborhood context."""
    if context_size <= 0:
        raise ValueError("context_size must be > 0")

    np = _require_numpy()

    first_rows: list[dict[str, Any]] = []
    for sequences in ticker_feature_sequences.values():
        for sequence in sequences:
            if sequence:
                first_rows = sequence
                break
        if first_rows:
            break

    columns = feature_columns if feature_columns is not None else infer_feature_columns(first_rows)
    if not columns:
        raise ValueError("No feature columns available for panel dataset construction")

    rows_by_timestamp: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for ticker, sequences in sorted(ticker_feature_sequences.items()):
        for sequence in sequences:
            for row in sequence:
                timestamp = row.get("timestamp")
                if timestamp is None:
                    raise ValueError("Panel dataset construction requires timestamp values")
                rows_by_timestamp[timestamp].append(row)

    X_values: list[list[list[float]]] = []
    y_values: list[float] = []
    close_values: list[float] = []
    next_close_values: list[float] = []
    target_return_values: list[float] = []
    direction_values: list[float] = []
    threshold_values: list[float] = []
    rank_values: list[float] = []
    timestamps: list[datetime] = []
    tickers: list[str] = []

    left_context = context_size // 2
    right_context = context_size - left_context - 1
    pad_row = _pad_feature_vector(len(columns), fill_value)

    for timestamp in sorted(rows_by_timestamp.keys()):
        timestamp_rows = sorted(
            rows_by_timestamp[timestamp],
            key=lambda row: (str(row.get("ticker", "")), row.get("timestamp")),
        )
        if not timestamp_rows:
            continue

        for center_index, center_row in enumerate(timestamp_rows):
            if label_key not in center_row:
                raise ValueError(f"Missing label column {label_key!r} in panel sample")
            if "close" not in center_row or "next_close" not in center_row:
                raise ValueError("Panel samples require close and next_close values")

            start_index = center_index - left_context
            end_index = center_index + right_context
            context_rows: list[list[float]] = []
            for position in range(start_index, end_index + 1):
                if 0 <= position < len(timestamp_rows):
                    source_row = timestamp_rows[position]
                    context_rows.append([float(source_row[column]) for column in columns])
                else:
                    context_rows.append(pad_row.copy())

            X_values.append(context_rows)
            y_values.append(float(center_row[label_key]))
            close_values.append(float(center_row["close"]))
            next_close_values.append(float(center_row["next_close"]))
            fallback_return = (float(center_row["next_close"]) - float(center_row["close"])) / max(
                float(center_row["close"]),
                1e-8,
            )
            target_return_values.append(float(center_row.get(return_key, fallback_return)))
            direction_values.append(float(center_row.get(direction_key, center_row.get("label", 0.0))))
            threshold_values.append(float(center_row.get(threshold_key, 1.0)))
            rank_values.append(float(center_row.get(rank_key, 0.5)))
            timestamps.append(center_row["timestamp"])
            tickers.append(str(center_row.get("ticker", "")))

    if not X_values:
        empty_x = np.zeros((0, context_size, len(columns)), dtype=np.float32)
        empty_y = np.zeros((0,), dtype=np.float32)
        empty_close = np.zeros((0,), dtype=np.float32)
        empty_aux = np.zeros((0,), dtype=np.float32)
        return PanelDatasetArtifacts(
            X=empty_x,
            y=empty_y,
            close=empty_close,
            next_close=empty_close.copy(),
            timestamps=[],
            tickers=[],
            feature_columns=columns,
            context_size=context_size,
            target_return=empty_aux,
            direction_label=empty_aux.copy(),
            threshold_label=empty_aux.copy(),
            rank_target=empty_aux.copy(),
        )

    return PanelDatasetArtifacts(
        X=np.asarray(X_values, dtype=np.float32),
        y=np.asarray(y_values, dtype=np.float32),
        close=np.asarray(close_values, dtype=np.float32),
        next_close=np.asarray(next_close_values, dtype=np.float32),
        timestamps=timestamps,
        tickers=tickers,
        feature_columns=columns,
        context_size=context_size,
        target_return=np.asarray(target_return_values, dtype=np.float32),
        direction_label=np.asarray(direction_values, dtype=np.float32),
        threshold_label=np.asarray(threshold_values, dtype=np.int64),
        rank_target=np.asarray(rank_values, dtype=np.float32),
    )


class PanelTensorDataset:
    """Torch dataset wrapper for panel artifacts."""

    def __init__(self, artifacts: PanelDatasetArtifacts) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch is required for PanelTensorDataset") from exc

        self._torch = torch
        self.X = torch.tensor(artifacts.X, dtype=torch.float32)
        self.y = torch.tensor(artifacts.y, dtype=torch.float32)
        self.close = torch.tensor(artifacts.close, dtype=torch.float32)
        self.next_close = torch.tensor(artifacts.next_close, dtype=torch.float32)
        self.target_return = (
            torch.tensor(artifacts.target_return, dtype=torch.float32) if artifacts.target_return is not None else None
        )
        self.direction_label = (
            torch.tensor(artifacts.direction_label, dtype=torch.float32) if artifacts.direction_label is not None else None
        )
        self.threshold_label = (
            torch.tensor(artifacts.threshold_label, dtype=torch.long) if artifacts.threshold_label is not None else None
        )
        self.rank_target = torch.tensor(artifacts.rank_target, dtype=torch.float32) if artifacts.rank_target is not None else None
        self.timestamps = artifacts.timestamps
        self.tickers = artifacts.tickers
        self.feature_columns = artifacts.feature_columns
        self.context_size = artifacts.context_size

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = {
            "X": self.X[index],
            "y": self.y[index],
            "close": self.close[index],
            "next_close": self.next_close[index],
        }
        if self.target_return is not None:
            sample["target_return"] = self.target_return[index]
        if self.direction_label is not None:
            sample["direction_label"] = self.direction_label[index]
        if self.threshold_label is not None:
            sample["threshold_label"] = self.threshold_label[index]
        if self.rank_target is not None:
            sample["rank_target"] = self.rank_target[index]
        return sample

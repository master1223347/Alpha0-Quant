"""Torch DataLoader creation helpers."""

from __future__ import annotations

from typing import Any

from src.dataset.window_dataset import WindowDatasetArtifacts


def _pick_value(source: Any, names: tuple[str, ...]) -> Any | None:
    if isinstance(source, dict):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
        return None

    for name in names:
        if hasattr(source, name):
            value = getattr(source, name)
            if value is not None:
                return value
    return None


class ArtifactTensorDataset:
    """Wrap window-like or panel-like artifacts into a torch Dataset."""

    def __init__(self, artifacts: Any) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch is required for ArtifactTensorDataset") from exc

        self._torch = torch
        self.X = self._to_tensor(
            _pick_value(artifacts, ("X", "inputs", "features", "panel_X", "panel_features", "panel_inputs")),
            dtype=torch.float32,
        )
        self.y = self._to_tensor(
            _pick_value(artifacts, ("y", "target", "targets", "label", "labels", "panel_y", "panel_targets")),
            dtype=torch.float32,
        )
        self.close = self._to_tensor(_pick_value(artifacts, ("close", "price", "prices", "panel_close")), dtype=torch.float32)
        self.next_close = self._to_tensor(
            _pick_value(artifacts, ("next_close", "future_close", "next_price", "panel_next_close")),
            dtype=torch.float32,
        )
        self.target_return = self._to_tensor(
            _pick_value(artifacts, ("target_return", "forward_return", "y_return", "return_target")),
            dtype=torch.float32,
        )
        self.direction_label = self._to_tensor(
            _pick_value(artifacts, ("direction_label", "binary_label", "label")),
            dtype=torch.float32,
        )
        self.threshold_label = self._to_tensor(
            _pick_value(artifacts, ("threshold_label", "threshold_target")),
            dtype=torch.long,
        )
        self.rank_target = self._to_tensor(
            _pick_value(artifacts, ("rank_target", "cross_sectional_rank", "rank_label")),
            dtype=torch.float32,
        )
        self.timestamps = _pick_value(artifacts, ("timestamps",))
        self.tickers = _pick_value(artifacts, ("tickers",))
        self.feature_columns = _pick_value(artifacts, ("feature_columns",))
        self.artifacts = artifacts

        if self.X is None or self.y is None:
            raise ValueError("Artifact dataset must provide X and y values")
        if self.close is None:
            self.close = torch.zeros_like(self.y, dtype=torch.float32)
        if self.next_close is None:
            self.next_close = self.close.clone()

    def _to_tensor(self, value: Any | None, *, dtype: Any | None = None) -> Any | None:
        if value is None:
            return None
        if self._torch.is_tensor(value):
            return value.to(dtype=dtype) if dtype is not None else value
        tensor = self._torch.as_tensor(value)
        return tensor.to(dtype=dtype) if dtype is not None else tensor

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        def _normalize_timestamp(value: Any) -> Any:
            if hasattr(value, "timestamp"):
                try:
                    return float(value.timestamp())
                except Exception:
                    return value
            return value

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
        if self.timestamps is not None:
            sample["timestamp"] = _normalize_timestamp(self.timestamps[index])
        if self.tickers is not None:
            sample["ticker"] = self.tickers[index]
        return sample


def _wrap_dataset(artifacts: Any) -> Any:
    try:
        from torch.utils.data import Dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for create_dataloaders") from exc

    if isinstance(artifacts, Dataset):
        return artifacts
    if isinstance(artifacts, WindowDatasetArtifacts):
        return ArtifactTensorDataset(artifacts)
    if isinstance(artifacts, dict) or any(hasattr(artifacts, name) for name in ("X", "y", "inputs", "features", "panel_X")):
        return ArtifactTensorDataset(artifacts)
    raise TypeError(f"Unsupported dataset artifact type: {type(artifacts)!r}")


def create_dataloaders(
    datasets: dict[str, Any],
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> dict[str, Any]:
    """Build split-aware torch DataLoaders from window or panel artifacts."""
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for create_dataloaders") from exc

    loaders: dict[str, Any] = {}
    for split_name, artifacts in datasets.items():
        dataset = _wrap_dataset(artifacts)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_train and split_name == "train",
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders

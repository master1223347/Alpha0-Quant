"""Torch DataLoader creation helpers."""

from __future__ import annotations

from typing import Any

from src.dataset.window_dataset import WindowDatasetArtifacts, WindowTensorDataset


def create_dataloaders(
    datasets: dict[str, WindowDatasetArtifacts],
    *,
    batch_size: int = 256,
    num_workers: int = 0,
    shuffle_train: bool = True,
) -> dict[str, Any]:
    """Build split-aware torch DataLoaders from window artifacts."""
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch is required for create_dataloaders") from exc

    loaders: dict[str, Any] = {}
    for split_name, artifacts in datasets.items():
        tensor_dataset = WindowTensorDataset(artifacts)
        loaders[split_name] = DataLoader(
            tensor_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train and split_name == "train",
            num_workers=num_workers,
            drop_last=False,
        )
    return loaders

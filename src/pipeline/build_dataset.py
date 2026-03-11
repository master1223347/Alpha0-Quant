"""Dataset build pipeline from feature sequences."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.default_config import ExperimentConfig, get_default_config
from src.dataset.sampler import split_ticker_sequences
from src.dataset.window_dataset import WindowDatasetArtifacts, build_labeled_windows
from src.features.normalize import FeatureNormalizer, fit_feature_normalizer, transform_feature_rows
from src.pipeline.build_features import FeatureBuildArtifacts, build_feature_store
from src.utils.logger import get_logger
from src.utils.paths import ensure_parent


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class BuildDatasetArtifacts:
    datasets: dict[str, WindowDatasetArtifacts]
    feature_columns: list[str]
    normalizer: FeatureNormalizer | None
    split_row_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        serializable = asdict(self)
        serializable["datasets"] = {
            split: {
                "samples": len(artifacts),
                "feature_columns": artifacts.feature_columns,
            }
            for split, artifacts in self.datasets.items()
        }
        return serializable


def _label_sequence(sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach next-close binary labels to each row except the terminal candle."""
    labeled: list[dict[str, Any]] = []
    for index in range(len(sequence) - 1):
        current = dict(sequence[index])
        next_close = float(sequence[index + 1]["close"])
        current_close = float(current["close"])
        current["next_close"] = next_close
        current["label"] = 1 if next_close > current_close else 0
        labeled.append(current)
    return labeled


def _label_ticker_sequences(ticker_sequences: dict[str, list[list[dict[str, Any]]]]) -> dict[str, list[list[dict[str, Any]]]]:
    labeled: dict[str, list[list[dict[str, Any]]]] = {}
    for ticker, sequences in ticker_sequences.items():
        labeled_sequences = []
        for sequence in sequences:
            labeled_sequence = _label_sequence(sequence)
            if labeled_sequence:
                labeled_sequences.append(labeled_sequence)
        if labeled_sequences:
            labeled[ticker] = labeled_sequences
    return labeled


def _flatten_rows(ticker_sequences: dict[str, list[list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return [row for sequences in ticker_sequences.values() for sequence in sequences for row in sequence]


def _normalize_split_sequences(
    split_sequences: dict[str, dict[str, list[list[dict[str, Any]]]]],
    feature_columns: list[str],
) -> tuple[dict[str, dict[str, list[list[dict[str, Any]]]]], FeatureNormalizer | None]:
    train_rows = _flatten_rows(split_sequences["train"])
    if not train_rows:
        return split_sequences, None

    normalizer = fit_feature_normalizer(train_rows, feature_columns=feature_columns)
    normalized: dict[str, dict[str, list[list[dict[str, Any]]]]] = {"train": {}, "val": {}, "test": {}}

    for split_name, ticker_map in split_sequences.items():
        normalized_tickers: dict[str, list[list[dict[str, Any]]]] = {}
        for ticker, sequences in ticker_map.items():
            normalized_sequences = []
            for sequence in sequences:
                normalized_sequences.append(transform_feature_rows(sequence, normalizer))
            normalized_tickers[ticker] = normalized_sequences
        normalized[split_name] = normalized_tickers
    return normalized, normalizer


def _write_table(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = ensure_parent(path)
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_parquet(output_path, index=False)
        return output_path
    except Exception:
        fallback = output_path.with_suffix(".json")
        with fallback.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, default=str)
        return fallback


def build_dataset(
    config: ExperimentConfig | None = None,
    *,
    exchange: str | None = None,
    asset_type: str | None = None,
) -> BuildDatasetArtifacts:
    """Build normalized train/val/test window datasets."""
    config = config or get_default_config()
    feature_artifacts: FeatureBuildArtifacts = build_feature_store(config, exchange=exchange, asset_type=asset_type)
    labeled_sequences = _label_ticker_sequences(feature_artifacts.ticker_sequences)

    split_sequences = split_ticker_sequences(
        labeled_sequences,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
    )

    if config.features.normalize:
        split_sequences, normalizer = _normalize_split_sequences(split_sequences, feature_artifacts.feature_columns)
    else:
        normalizer = None

    datasets: dict[str, WindowDatasetArtifacts] = {}
    split_row_counts: dict[str, int] = {}
    for split_name, ticker_sequences in split_sequences.items():
        split_row_counts[split_name] = sum(len(sequence) for sequences in ticker_sequences.values() for sequence in sequences)
        datasets[split_name] = build_labeled_windows(
            ticker_sequences,
            window_size=config.dataset.window_size,
            stride=config.dataset.stride,
            feature_columns=feature_artifacts.feature_columns,
            label_key="label",
        )

    all_rows = []
    label_rows = []
    for split_name, ticker_map in split_sequences.items():
        for ticker, sequences in ticker_map.items():
            for sequence in sequences:
                for row in sequence:
                    record = dict(row)
                    record["split"] = split_name
                    all_rows.append(record)
                    label_rows.append(
                        {
                            "timestamp": row["timestamp"],
                            "ticker": ticker,
                            "close": row["close"],
                            "next_close": row["next_close"],
                            "label": row["label"],
                            "split": split_name,
                        }
                    )

    dataset_output = _write_table(config.data.dataset_path, all_rows)
    labels_output = _write_table(config.data.labels_path, label_rows)
    LOGGER.info("Dataset rows written to %s", dataset_output)
    LOGGER.info("Label rows written to %s", labels_output)

    info_path = Path(config.data.metadata_dir) / "dataset_info.json"
    info_path.parent.mkdir(parents=True, exist_ok=True)
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "feature_columns": feature_artifacts.feature_columns,
                "split_row_counts": split_row_counts,
                "window_size": config.dataset.window_size,
                "stride": config.dataset.stride,
            },
            handle,
            indent=2,
        )

    return BuildDatasetArtifacts(
        datasets=datasets,
        feature_columns=feature_artifacts.feature_columns,
        normalizer=normalizer,
        split_row_counts=split_row_counts,
    )

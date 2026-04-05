"""Dataset build pipeline from feature sequences."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.default_config import ExperimentConfig, get_default_config
from src.dataset.panel_dataset import PanelDatasetArtifacts, build_panel_dataset
from src.dataset.sampler import split_ticker_sequences
from src.dataset.window_dataset import WindowDatasetArtifacts, build_labeled_windows
from src.features.normalize import FeatureNormalizer, fit_feature_normalizer, transform_feature_rows
from src.pipeline.build_features import FeatureBuildArtifacts, build_feature_store
from src.targets.labeling import TARGET_COLUMNS, assign_cross_sectional_rank, label_ticker_sequences
from src.utils.logger import get_logger
from src.utils.paths import ensure_parent


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class BuildDatasetArtifacts:
    datasets: dict[str, WindowDatasetArtifacts | PanelDatasetArtifacts]
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


def _resolve_dataset_type(config: ExperimentConfig) -> str:
    configured = config.dataset.dataset_type.lower().strip()
    if configured in {"window", "panel"}:
        return configured
    if configured not in {"auto", ""}:
        raise ValueError(f"Unsupported dataset_type: {config.dataset.dataset_type}")

    model_name = config.model.model_name.lower()
    if any(marker in model_name for marker in ("panel", "cross_sectional", "cross-sectional")):
        return "panel"
    return "window"


def _resolve_label_horizon(config: ExperimentConfig) -> int:
    return int(config.targets.horizon if config.targets.horizon is not None else config.dataset.label_horizon)


def _resolve_panel_context_size(config: ExperimentConfig) -> int:
    context_size = config.dataset.panel_context_size
    if context_size is None:
        context_size = config.dataset.window_size
    if context_size <= 0:
        raise ValueError("panel_context_size/window_size must be > 0")
    return int(context_size)


def _resolve_return_target_key(config: ExperimentConfig) -> str:
    preferred = str(config.targets.primary_target).strip()
    if preferred in {"next_log_return", "vol_target", "z_return", "cross_sectional_rank"}:
        return preferred
    return "next_log_return"


def _flatten_rows(ticker_sequences: dict[str, list[list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return [row for sequences in ticker_sequences.values() for sequence in sequences for row in sequence]


def _split_timestamp_sets(
    split_sequences: dict[str, dict[str, list[list[dict[str, Any]]]]],
) -> dict[str, set[Any]]:
    timestamp_sets: dict[str, set[Any]] = {}
    for split_name, ticker_map in split_sequences.items():
        timestamps: set[Any] = set()
        for sequences in ticker_map.values():
            for sequence in sequences:
                for row in sequence:
                    timestamp = row.get("timestamp")
                    if timestamp is not None:
                        timestamps.add(timestamp)
        timestamp_sets[split_name] = timestamps
    return timestamp_sets


def _timestamp_overlap_counts(timestamp_sets: dict[str, set[Any]]) -> dict[str, int]:
    overlaps: dict[str, int] = {}
    pairs = (("train", "val"), ("train", "test"), ("val", "test"))
    for left, right in pairs:
        left_set = timestamp_sets.get(left, set())
        right_set = timestamp_sets.get(right, set())
        overlaps[f"{left}_{right}"] = len(left_set.intersection(right_set))
    return overlaps


def _validate_split_integrity(
    *,
    split_mode: str,
    split_sequences: dict[str, dict[str, list[list[dict[str, Any]]]]],
    feature_columns: list[str],
) -> dict[str, int]:
    timestamp_sets = _split_timestamp_sets(split_sequences)
    overlap_counts = _timestamp_overlap_counts(timestamp_sets)
    has_overlap = any(count > 0 for count in overlap_counts.values())
    has_cross_sectional_features = any(column.startswith("cs_") for column in feature_columns)

    if split_mode == "global_time":
        if has_overlap:
            raise ValueError(
                f"Leakage risk: global_time split produced overlapping timestamps across splits: {overlap_counts}"
            )

        train_timestamps = timestamp_sets.get("train", set())
        val_timestamps = timestamp_sets.get("val", set())
        test_timestamps = timestamp_sets.get("test", set())
        if train_timestamps and val_timestamps and max(train_timestamps) > min(val_timestamps):
            raise ValueError("Leakage risk: train timestamps extend past validation start under global_time split")
        if val_timestamps and test_timestamps and max(val_timestamps) > min(test_timestamps):
            raise ValueError("Leakage risk: validation timestamps extend past test start under global_time split")
        return overlap_counts

    if has_overlap and has_cross_sectional_features:
        raise ValueError(
            "Leakage risk: per_ticker split with cross-sectional features causes same-timestamp contamination "
            f"across splits ({overlap_counts}). Use dataset.split_mode='global_time'."
        )
    if has_overlap:
        LOGGER.warning(
            "Potential temporal leakage: %s split has overlapping timestamps across splits: %s",
            split_mode,
            overlap_counts,
        )
    return overlap_counts


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
    if not feature_artifacts.feature_columns:
        raise ValueError(
            "No usable feature rows were built. Try increasing data coverage and/or relaxing sequence constraints "
            "(for example: set data.max_tickers higher or null, lower data.min_sequence_length, and lower "
            "dataset.window_size)."
        )

    target_horizon = _resolve_label_horizon(config)
    labeled_sequences = label_ticker_sequences(
        feature_artifacts.ticker_sequences,
        horizon=target_horizon,
        threshold=float(config.targets.threshold),
        volatility_window=int(config.targets.volatility_window),
        zscore_window=int(config.targets.zscore_window),
    )
    assign_cross_sectional_rank(labeled_sequences, target_column="next_log_return")

    split_mode = config.dataset.split_mode.lower().strip()
    split_sequences = split_ticker_sequences(
        labeled_sequences,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
        split_mode=split_mode,
    )
    overlap_counts = _validate_split_integrity(
        split_mode=split_mode,
        split_sequences=split_sequences,
        feature_columns=feature_artifacts.feature_columns,
    )

    if config.features.normalize:
        split_sequences, normalizer = _normalize_split_sequences(split_sequences, feature_artifacts.feature_columns)
    else:
        normalizer = None

    dataset_type = _resolve_dataset_type(config)
    if dataset_type == "panel":
        panel_context_size = _resolve_panel_context_size(config)
        if config.dataset.panel_context_size is not None and config.dataset.window_size != panel_context_size:
            # Keep downstream model construction aligned with the panel context tensor.
            config.dataset.window_size = panel_context_size
    else:
        panel_context_size = config.dataset.window_size

    return_target_key = _resolve_return_target_key(config)
    datasets: dict[str, WindowDatasetArtifacts | PanelDatasetArtifacts] = {}
    split_row_counts: dict[str, int] = {}
    for split_name, ticker_sequences in split_sequences.items():
        split_row_counts[split_name] = sum(len(sequence) for sequences in ticker_sequences.values() for sequence in sequences)
        if dataset_type == "panel":
            datasets[split_name] = build_panel_dataset(
                ticker_sequences,
                context_size=panel_context_size,
                feature_columns=feature_artifacts.feature_columns,
                label_key=config.targets.primary_target,
                return_key=return_target_key,
            )
        else:
            datasets[split_name] = build_labeled_windows(
                ticker_sequences,
                window_size=config.dataset.window_size,
                stride=config.dataset.stride,
                feature_columns=feature_artifacts.feature_columns,
                label_key=config.targets.primary_target,
                return_key=return_target_key,
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
                    label_record = {
                        "timestamp": row["timestamp"],
                        "ticker": ticker,
                        "close": row["close"],
                        "next_close": row["next_close"],
                        "split": split_name,
                    }
                    for column in TARGET_COLUMNS:
                        if column not in row:
                            raise ValueError(f"Missing target column {column!r} when building label rows")
                        label_record[column] = row[column]
                    label_rows.append(label_record)

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
                "dataset_type": dataset_type,
                "split_mode": split_mode,
                "primary_target": config.targets.primary_target,
                "return_target": return_target_key,
                "target_columns": list(TARGET_COLUMNS),
                "context_size": panel_context_size if dataset_type == "panel" else None,
                "timestamp_overlap_counts": overlap_counts,
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

"""Audit event feature/target separation for the synthetic event config."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.events.registry import EVENT_FEATURE_COLUMNS, EVENT_ID_COLUMNS, EVENT_TARGET_COLUMNS
from src.features.normalize import PASSTHROUGH_FEATURE_COLUMNS
from src.pipeline.build_dataset import build_dataset
from src.pipeline.build_features import build_feature_store
from src.pipeline.run_experiment import load_experiment_config
from src.targets.labeling import TARGET_COLUMNS


OUTPUT_PATH = Path("models/logs/event_feature_matrix_audit.json")
SUSPICIOUS_TOKENS = (
    "label",
    "target",
    "future",
    "next",
    "drift",
    "reversal",
    "absorption",
    "realized",
    "forward",
)


def _ensure_fixture() -> None:
    subprocess.run([sys.executable, "scripts/create_synthetic_event_fixture.py"], check=True)


def _suspicious(columns: list[str]) -> dict[str, list[str]]:
    return {
        token: sorted([column for column in columns if token in column.lower()])
        for token in SUSPICIOUS_TOKENS
    }


def run_feature_matrix_audit(config_path: str = "experiments/exp_event_market_lab.yaml") -> dict[str, Any]:
    _ensure_fixture()
    config = load_experiment_config(config_path)
    feature_artifacts = build_feature_store(config)
    dataset_artifacts = build_dataset(config)
    feature_cols = list(feature_artifacts.feature_columns)
    target_cols = list(TARGET_COLUMNS)
    event_feature_cols = [column for column in feature_cols if column in EVENT_FEATURE_COLUMNS]
    event_label_cols = list(EVENT_TARGET_COLUMNS)
    normalized_feature_cols = (
        list(dataset_artifacts.normalizer.feature_columns)
        if dataset_artifacts.normalizer is not None
        else []
    )

    overlap_feature_target = sorted(set(feature_cols).intersection(target_cols))
    overlap_feature_event_target = sorted(set(feature_cols).intersection(event_label_cols))
    overlap_normalized_id = sorted(set(normalized_feature_cols).intersection(EVENT_ID_COLUMNS))

    for column in target_cols:
        assert column not in feature_cols, f"target column leaked into feature matrix: {column}"
    for column in event_label_cols:
        assert column not in feature_cols, f"event target column leaked into feature matrix: {column}"

    report = {
        "config": config_path,
        "feature_columns": feature_cols,
        "target_columns": target_cols,
        "event_feature_columns": event_feature_cols,
        "event_label_columns": event_label_cols,
        "overlap_feature_target": overlap_feature_target,
        "overlap_feature_event_target": overlap_feature_event_target,
        "normalized_feature_columns": normalized_feature_cols,
        "passthrough_feature_columns": sorted(PASSTHROUGH_FEATURE_COLUMNS),
        "overlap_normalized_id_columns": overlap_normalized_id,
        "suspicious_columns": _suspicious(feature_cols),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {OUTPUT_PATH}")
    return report


def main() -> None:
    run_feature_matrix_audit()


if __name__ == "__main__":
    main()

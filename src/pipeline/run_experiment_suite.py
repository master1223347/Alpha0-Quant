"""Run a batch of experiments while reusing dataset builds when possible."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.config.default_config import config_to_dict
from src.pipeline.build_dataset import build_dataset
from src.pipeline.evaluate_model import run_evaluation_pipeline
from src.pipeline.run_experiment import load_experiment_config
from src.pipeline.train_model import run_training_pipeline
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _stable(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str)


def _dataset_signature(config: Any) -> str:
    payload = {
        "data": config_to_dict(config).get("data", {}),
        "universe": config_to_dict(config).get("universe", {}),
        "targets": config_to_dict(config).get("targets", {}),
        "features": config_to_dict(config).get("features", {}),
        "dataset": config_to_dict(config).get("dataset", {}),
    }
    return _stable(payload)


def _resolve_config_paths(values: list[str]) -> list[str]:
    paths: list[str] = []
    for value in values:
        candidate = Path(value)
        if candidate.exists():
            if candidate.is_dir():
                paths.extend(str(path) for path in sorted(candidate.glob("*.yaml")))
            else:
                paths.append(str(candidate))
            continue
        matches = sorted(Path(".").glob(value))
        paths.extend(str(path) for path in matches if path.is_file())
    if not paths:
        raise ValueError("No config files matched the provided patterns")
    seen: set[str] = set()
    deduped: list[str] = []
    for path in paths:
        resolved = str(Path(path))
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _resolve_result_path(
    *,
    config: Any,
    config_path: str,
    used_paths: set[Path],
) -> Path:
    base_path = Path(config.training.metrics_path).with_name("experiment_result.json")
    if base_path not in used_paths:
        used_paths.add(base_path)
        return base_path

    stem = Path(config_path).stem
    suffix_index = 1
    while True:
        if suffix_index == 1:
            candidate = base_path.with_name(f"experiment_result_{stem}.json")
        else:
            candidate = base_path.with_name(f"experiment_result_{stem}_{suffix_index}.json")
        if candidate not in used_paths:
            used_paths.add(candidate)
            LOGGER.warning(
                "Duplicate metrics_path detected; writing suite result for %s to %s",
                config.name,
                candidate,
            )
            return candidate
        suffix_index += 1


def run_experiment_suite(config_paths: list[str]) -> dict[str, Any]:
    configs = [(path, load_experiment_config(path)) for path in config_paths]
    groups: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    for path, config in configs:
        groups[_dataset_signature(config)].append((path, config))

    suite_results: list[dict[str, Any]] = []
    used_result_paths: set[Path] = set()
    for group_index, grouped_configs in enumerate(groups.values(), start=1):
        base_path, base_config = grouped_configs[0]
        LOGGER.info(
            "Building dataset for group %s using %s (%s configs)",
            group_index,
            base_path,
            len(grouped_configs),
        )
        dataset_artifacts = build_dataset(base_config)

        for config_path, config in grouped_configs:
            LOGGER.info("Running experiment %s", config.name)
            train_artifacts = run_training_pipeline(
                config=config,
                dataset_artifacts=dataset_artifacts,
            )
            eval_artifacts = run_evaluation_pipeline(
                config=config,
                model=train_artifacts.model,
                dataset_artifacts=dataset_artifacts,
                checkpoint_path=train_artifacts.training.best_checkpoint_path,
            )
            result = {
                "config_path": config_path,
                "config": config_to_dict(config),
                "training": train_artifacts.training.to_dict(),
                "evaluation_report_path": eval_artifacts.report_path,
            }
            result_path = _resolve_result_path(
                config=config,
                config_path=config_path,
                used_paths=used_result_paths,
            )
            result_path.parent.mkdir(parents=True, exist_ok=True)
            with result_path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
            suite_results.append(result)

    return {"runs": suite_results, "run_count": len(suite_results), "dataset_group_count": len(groups)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment suite with dataset reuse")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Config paths, directories, or glob patterns",
    )
    parser.add_argument(
        "--summary",
        default="models/logs/experiment_suite_summary.json",
        help="Path to write suite summary JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_paths = _resolve_config_paths(args.configs)
    summary = run_experiment_suite(config_paths)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    LOGGER.info("Suite summary written to %s", summary_path)


if __name__ == "__main__":
    main()

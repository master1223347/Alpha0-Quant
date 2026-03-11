"""Run full experiment lifecycle from YAML config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config.default_config import apply_overrides, config_to_dict, get_default_config
from src.pipeline.evaluate_model import run_evaluation_pipeline
from src.pipeline.train_model import run_training_pipeline
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyYAML is required to load experiment YAML files") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def load_experiment_config(path: str | Path) -> Any:
    """Load YAML experiment config and merge with defaults."""
    overrides = _load_yaml(path)
    experiment_name = str(overrides.get("name", Path(path).stem))
    config = get_default_config(name=experiment_name)
    return apply_overrides(config, overrides)


def run_experiment(config_path: str | Path) -> dict[str, Any]:
    """Run training + evaluation and return serializable artifacts."""
    config = load_experiment_config(config_path)
    LOGGER.info("Running experiment %s", config.name)
    train_artifacts = run_training_pipeline(config=config)
    eval_artifacts = run_evaluation_pipeline(
        config=config,
        model=train_artifacts.model,
        dataset_artifacts=train_artifacts.dataset,
        checkpoint_path=train_artifacts.training.best_checkpoint_path,
    )

    result = {
        "config": config_to_dict(config),
        "training": train_artifacts.training.to_dict(),
        "evaluation_report_path": eval_artifacts.report_path,
    }

    result_path = Path(config.training.metrics_path).with_name("experiment_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    LOGGER.info("Experiment result written to %s", result_path)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alpha0 experiment from YAML config")
    parser.add_argument("--config", required=True, help="Path to experiment YAML file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_experiment(args.config)


if __name__ == "__main__":
    main()

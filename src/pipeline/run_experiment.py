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
    except ModuleNotFoundError:
        return _load_simple_yaml(path)

    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def _parse_simple_yaml_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"", "null", "Null", "NULL", "~"}:
        return None
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(part.strip()) for part in inner.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(marker in value for marker in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Minimal fallback parser for the repo's simple experiment YAML files."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            line = raw_line.rstrip("\n")
            indent = len(line) - len(line.lstrip(" "))
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            key = key.strip()
            while stack and indent <= stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if value.strip() == "":
                child: dict[str, Any] = {}
                parent[key] = child
                stack.append((indent, child))
            else:
                parent[key] = _parse_simple_yaml_scalar(value)
    return root


def load_experiment_config(path: str | Path) -> Any:
    """Load YAML experiment config and merge with defaults."""
    overrides = _load_yaml(path)
    experiment_name = str(overrides.get("name", Path(path).stem))
    config = get_default_config(name=experiment_name)
    return apply_overrides(config, overrides)


def run_experiment(
    config_path: str | Path,
    *,
    split_mode: str | None = None,
    walk_forward: bool | None = None,
    embargo_bars: int | None = None,
) -> dict[str, Any]:
    """Run training + evaluation and return serializable artifacts."""
    config = load_experiment_config(config_path)
    if split_mode is not None:
        config.dataset.split_mode = str(split_mode)
    if walk_forward is not None:
        config.evaluation.walk_forward_enabled = bool(walk_forward)
    if embargo_bars is not None:
        config.evaluation.walk_forward_embargo_bars = int(max(0, embargo_bars))
    LOGGER.info("Running experiment %s", config.name)
    train_artifacts = run_training_pipeline(config=config)
    eval_artifacts = run_evaluation_pipeline(
        config=config,
        model=train_artifacts.model,
        dataset_artifacts=train_artifacts.dataset,
        checkpoint_path=train_artifacts.training.best_checkpoint_path,
    )
    walk_forward_retrain_result = None
    if bool(getattr(config.walk_forward_retrain, "enabled", False)):
        from src.pipeline.walk_forward_retrain import run_walk_forward_retrain

        walk_forward_retrain_result = run_walk_forward_retrain(config=config).to_dict()

    result = {
        "config": config_to_dict(config),
        "training": train_artifacts.training.to_dict(),
        "evaluation_report_path": eval_artifacts.report_path,
        "walk_forward_retrain": walk_forward_retrain_result,
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
    parser.add_argument("--split_mode", default=None, choices=["global_time", "per_ticker"], help="Optional split mode override")
    parser.add_argument(
        "--walk_forward",
        default=None,
        choices=["true", "false"],
        help="Enable/disable walk-forward evaluation override",
    )
    parser.add_argument("--embargo_bars", type=int, default=None, help="Optional walk-forward embargo bars override")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    walk_forward = None
    if args.walk_forward is not None:
        walk_forward = str(args.walk_forward).strip().lower() == "true"
    run_experiment(
        args.config,
        split_mode=args.split_mode,
        walk_forward=walk_forward,
        embargo_bars=args.embargo_bars,
    )


if __name__ == "__main__":
    main()

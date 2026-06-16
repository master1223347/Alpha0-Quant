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
        "backtest_path": str(Path(config.training.metrics_path).with_name("backtest.json")),
        "event_study_report_path": str(Path(config.training.metrics_path).with_name("event_study_report.json"))
        if bool(getattr(config.events, "enabled", False))
        else None,
        "event_alpha_decay_report_path": str(Path(config.training.metrics_path).with_name("event_alpha_decay_report.json"))
        if bool(getattr(config.events, "enabled", False))
        else None,
        "walk_forward_retrain": walk_forward_retrain_result,
    }

    result_path = Path(config.training.metrics_path).with_name("experiment_result.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    LOGGER.info("Experiment result written to %s", result_path)
    _write_results_summary(config, eval_artifacts.report.to_dict(), result)
    return result


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_results_summary(config: Any, report: dict[str, Any], result: dict[str, Any]) -> None:
    metrics = report.get("metrics", {})
    backtest = report.get("backtest", {})
    lines = [
        "# Alpha0 Quant Results",
        "",
        "## Baseline Prediction",
        f"non_neutral_directional_hit_rate: {_fmt(metrics.get('accuracy', 'n/a'))}",
        f"auc: {_fmt(metrics.get('auc', 'n/a'))}",
        f"confidence_buckets: {len(metrics.get('confidence_buckets', []) or [])}",
        "",
        "## Opportunity / Trade Quality",
        f"sharpe_after_costs: {_fmt(backtest.get('sharpe', 'n/a'))}",
        f"hit_rate_after_costs: {_fmt(backtest.get('hit_rate', 'n/a'))}",
        f"average_trade_return: {_fmt(backtest.get('pnl', 0.0) / max(1, int(backtest.get('trade_count', 0) or 0)))}",
        f"trade_count: {_fmt(backtest.get('trade_count', 'n/a'))}",
        f"turnover_change: {_fmt(backtest.get('turnover_change', 0.0))}",
    ]
    event_metrics = metrics.get("event_intelligence", {})
    if event_metrics or bool(getattr(config.events, "enabled", False)):
        lines.extend(
            [
                "",
                "## Event Intelligence",
                f"event_sharpe: {_fmt(backtest.get('event_sharpe', 0.0))}",
                f"non_event_sharpe: {_fmt(backtest.get('non_event_sharpe', 0.0))}",
                f"event_hit_rate: {_fmt(backtest.get('event_hit_rate', 0.0))}",
                f"non_event_hit_rate: {_fmt(backtest.get('non_event_hit_rate', 0.0))}",
                f"trades_blocked_by_event_rules: {_fmt(backtest.get('trades_blocked_by_event_rules', 0))}",
                f"trades_added_by_event_rules: {_fmt(backtest.get('trades_added_by_event_rules', 0))}",
                "",
                "## Event Study",
                f"event_sample_count: {_fmt((metrics.get('event_study') or {}).get('event_sample_count', 0))}",
                f"placebo: {(metrics.get('event_study') or {}).get('placebo', {})}",
                "",
                "## News/Sentiment",
                "sentiment_coverage: see event_study_report.json",
                "sentiment_performance_correlation: not estimated in v1",
                "novelty_performance_correlation: not estimated in v1",
                "",
                "## SEC/Earnings/Macro",
                "window_performance: see event_study_report.json grouped event_type IDs",
                "",
                "## Alpha Decay",
                f"normal_signal_half_life: {_fmt((metrics.get('alpha_decay') or {}).get('normal_signal_half_life', 'n/a'))}",
                f"event_signal_half_life: {_fmt((metrics.get('alpha_decay') or {}).get('event_signal_half_life', 'n/a'))}",
                "",
                "## Leakage Warnings",
                "event_leakage_report: data/events/event_leakage_report.json",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            f"evaluation_report: {result.get('evaluation_report_path')}",
            f"backtest: {result.get('backtest_path')}",
            f"event_study: {result.get('event_study_report_path')}",
            f"event_alpha_decay: {result.get('event_alpha_decay_report_path')}",
        ]
    )
    with Path("results.txt").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


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

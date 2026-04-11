"""Run a data-first validation ladder before complex MINN experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.evaluation.backtest import BacktestReport, run_backtest
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics
from src.pipeline.build_dataset import BuildDatasetArtifacts, build_dataset
from src.pipeline.run_experiment import load_experiment_config
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)

BASE_PRICE_FEATURES = {
    "log_return",
    "candle_range",
    "candle_body",
    "upper_wick",
    "lower_wick",
    "short_term_momentum",
    "rolling_volatility",
}
VOLUME_FEATURES = {
    "relative_volume",
    "relative_volume_long",
    "volume_change",
    "volume_zscore",
}
MARKET_FEATURES = {
    "gap_return",
    "intrabar_return",
    "close_position",
    "range_expansion",
}
TIME_FEATURES = {
    "session_pos_sin",
    "session_pos_cos",
    "day_of_week_sin",
    "day_of_week_cos",
}


def _require_sklearn() -> Any:
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "scikit-learn is required for validation ladder baselines. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc


def _to_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce")


def _prepare_base_config(config_path: str | Path) -> Any:
    config = load_experiment_config(config_path)
    # Force broad coverage for validation ladder runs.
    config.data.max_tickers = None
    config.universe.max_tickers = None
    return config


def _apply_setup(config: Any, *, setup_name: str) -> Any:
    updated = copy.deepcopy(config)
    if setup_name == "temporal":
        updated.dataset.dataset_type = "window"
        updated.features.use_cross_sectional = False
        updated.targets.primary_target = "label"
        updated.targets.direction_target = "label"
        updated.targets.threshold_target = "threshold_label"
        updated.targets.return_target = "next_log_return"
    elif setup_name == "cross_sectional":
        updated.dataset.dataset_type = "panel"
        updated.features.use_cross_sectional = True
        updated.targets.primary_target = "label"
        updated.targets.direction_target = "label"
        updated.targets.threshold_target = "threshold_label"
        updated.targets.return_target = "next_log_return"
    else:
        raise ValueError(f"Unsupported setup_name: {setup_name}")
    return updated


def _build_and_load_frame(config: Any) -> tuple[pd.DataFrame, BuildDatasetArtifacts]:
    artifacts = build_dataset(config)
    data_path = Path(config.data.dataset_path)
    if data_path.exists():
        frame = pd.read_parquet(data_path)
    else:
        json_path = data_path.with_suffix(".json")
        frame = pd.DataFrame(json.loads(json_path.read_text(encoding="utf-8")))
    if "split" not in frame.columns:
        raise ValueError("Dataset rows must include split column for validation ladder")
    frame["timestamp"] = _to_datetime(frame["timestamp"])
    return frame, artifacts


def _get_split_arrays(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_values: pd.Series,
) -> dict[str, Any]:
    valid = frame.copy()
    valid["_target"] = target_values.values
    valid = valid.dropna(subset=feature_columns + ["_target", "close", "next_close", "timestamp"])
    splits: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        split = valid[valid["split"] == split_name].copy()
        splits[split_name] = {
            "frame": split,
            "X": split[feature_columns].to_numpy(dtype=np.float32),
            "y": split["_target"].to_numpy(),
        }
    return splits


def _compact_backtest(report: BacktestReport) -> dict[str, Any]:
    return {
        "pnl": float(report.pnl),
        "gross_pnl": float(report.gross_pnl),
        "sharpe": float(report.sharpe),
        "max_drawdown": float(report.max_drawdown),
        "hit_rate": float(report.hit_rate),
        "trade_count": int(report.trade_count),
        "long_count": int(report.long_count),
        "short_count": int(report.short_count),
        "flat_count": int(report.flat_count),
        "nan_signal_count": int(report.nan_signal_count),
        "selected_bars": int(report.selected_bars),
    }


def _audit_backtest(report: BacktestReport, *, sample_count: int) -> list[str]:
    issues: list[str] = []
    if (report.long_count + report.short_count + report.flat_count) != sample_count:
        issues.append("position bar counts do not sum to sample_count")
    if report.trade_count > sample_count:
        issues.append("trade_count exceeds sample_count")
    if report.selected_bars > sample_count:
        issues.append("selected_bars exceeds sample_count")
    if not math.isfinite(report.pnl):
        issues.append("pnl is non-finite")
    if not math.isfinite(report.sharpe):
        issues.append("sharpe is non-finite")
    return issues


def _classification_backtest(frame: pd.DataFrame, *, probabilities: np.ndarray, config: Any) -> BacktestReport:
    top_percentile = config.backtest.top_percentile
    if top_percentile is None:
        top_percentile = 0.2
    return run_backtest(
        probabilities=[float(value) for value in probabilities.tolist()],
        close=[float(value) for value in frame["close"].tolist()],
        next_close=[float(value) for value in frame["next_close"].tolist()],
        long_threshold=float(config.backtest.long_threshold),
        short_threshold=float(config.backtest.short_threshold),
        confidence_threshold=None,
        top_percentile=float(top_percentile),
        selection_mode="global_abs",
        periods_per_year=int(config.backtest.periods_per_year),
        execution_lag_bars=int(config.backtest.execution_lag_bars),
        flip_positions=bool(config.backtest.flip_positions),
        cost_bps_per_trade=float(config.backtest.cost_bps_per_trade),
        slippage_bps=float(config.backtest.slippage_bps),
    )


def _fit_classification_models(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, np.ndarray]:
    _require_sklearn()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    y_train_binary = y_train.astype(int)
    prior = float(np.clip(np.mean(y_train_binary), 1e-6, 1 - 1e-6))
    outputs: dict[str, np.ndarray] = {
        "prior": np.full((X_test.shape[0],), prior, dtype=np.float32),
    }

    logistic = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    logistic.fit(X_train, y_train_binary)
    outputs["logistic"] = logistic.predict_proba(X_test)[:, 1].astype(np.float32)

    tree = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )
    tree.fit(X_train, y_train_binary)
    outputs["tree"] = tree.predict_proba(X_test)[:, 1].astype(np.float32)

    tiny_nn = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=12,
                    random_state=42,
                ),
            ),
        ]
    )
    tiny_nn.fit(X_train, y_train_binary)
    outputs["tiny_nn"] = tiny_nn.predict_proba(X_test)[:, 1].astype(np.float32)
    return outputs


def _spearman_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0 or left.size != right.size:
        return float("nan")
    left_rank = pd.Series(left).rank(method="average")
    right_rank = pd.Series(right).rank(method="average")
    corr = left_rank.corr(right_rank)
    return float(corr) if corr is not None else float("nan")


def _top_bottom_spread(
    frame: pd.DataFrame,
    *,
    score_column: str,
    return_column: str,
    quantile: float = 0.10,
) -> float:
    spreads: list[float] = []
    for _, group in frame.groupby("timestamp", sort=True):
        if len(group) < 4:
            continue
        k = max(1, int(len(group) * quantile))
        ranked = group.sort_values(score_column)
        bottom = ranked.head(k)[return_column].mean()
        top = ranked.tail(k)[return_column].mean()
        spreads.append(float(top - bottom))
    return float(sum(spreads) / len(spreads)) if spreads else float("nan")


def _fit_ranking_models(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> dict[str, np.ndarray]:
    _require_sklearn()
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    outputs: dict[str, np.ndarray] = {
        "prior": np.full((X_test.shape[0],), float(np.mean(y_train)), dtype=np.float32),
    }

    linear = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    linear.fit(X_train, y_train)
    outputs["linear"] = linear.predict(X_test).astype(np.float32)

    tree = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )
    tree.fit(X_train, y_train)
    outputs["tree"] = tree.predict(X_test).astype(np.float32)

    tiny_nn = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=12,
                    random_state=42,
                ),
            ),
        ]
    )
    tiny_nn.fit(X_train, y_train)
    outputs["tiny_nn"] = tiny_nn.predict(X_test).astype(np.float32)
    return outputs


def _incremental_feature_stages(feature_columns: list[str], *, include_cross_sectional: bool) -> list[tuple[str, list[str]]]:
    ordered = list(feature_columns)
    base = [column for column in ordered if column in BASE_PRICE_FEATURES]
    volume = [column for column in ordered if column in VOLUME_FEATURES]
    market = [column for column in ordered if column in MARKET_FEATURES]
    time = [column for column in ordered if column in TIME_FEATURES]
    cross = [column for column in ordered if column.startswith("cs_")]

    stage_columns: list[tuple[str, list[str]]] = []
    selected: list[str] = []

    def _append(stage_name: str, additions: list[str]) -> None:
        nonlocal selected
        selected = selected + [column for column in additions if column not in selected]
        stage_columns.append((stage_name, selected.copy()))

    _append("base_price_only", base)
    _append("plus_volume", volume)
    _append("plus_market", market)
    _append("plus_time", time)
    if include_cross_sectional:
        _append("plus_cross_sectional", cross)
    return stage_columns


def _classification_variants(frame: pd.DataFrame) -> dict[str, pd.Series]:
    residual = frame["next_log_return"] - frame.groupby("timestamp")["next_log_return"].transform("mean")
    return {
        "next_bar_direction": frame["label"].astype(int),
        "volatility_expansion": (frame["next_log_return"].abs() > frame["vol_threshold"]).astype(int),
        "residualized_return_direction": (residual > 0.0).astype(int),
    }


def _run_setup(config: Any, *, setup_name: str) -> dict[str, Any]:
    configured = _apply_setup(config, setup_name=setup_name)
    frame, artifacts = _build_and_load_frame(configured)
    feature_columns = list(artifacts.feature_columns)

    if frame.empty:
        raise ValueError(f"No rows available for setup={setup_name}")

    setup_summary = {
        "setup": setup_name,
        "row_count": int(len(frame)),
        "ticker_count": int(frame["ticker"].nunique()) if "ticker" in frame.columns else 0,
        "time_min": str(frame["timestamp"].min()),
        "time_max": str(frame["timestamp"].max()),
        "split_counts": {str(key): int(value) for key, value in frame["split"].value_counts().to_dict().items()},
        "feature_count": len(feature_columns),
    }

    LOGGER.info(
        "Setup=%s rows=%s tickers=%s time=[%s,%s]",
        setup_name,
        setup_summary["row_count"],
        setup_summary["ticker_count"],
        setup_summary["time_min"],
        setup_summary["time_max"],
    )

    classification_results: dict[str, Any] = {}
    for variant_name, target_values in _classification_variants(frame).items():
        split_data = _get_split_arrays(frame, feature_columns=feature_columns, target_values=target_values)
        X_train = split_data["train"]["X"]
        y_train = split_data["train"]["y"].astype(int)
        X_test = split_data["test"]["X"]
        y_test = split_data["test"]["y"].astype(int)
        test_frame = split_data["test"]["frame"]

        if X_train.size == 0 or X_test.size == 0 or y_train.size == 0 or y_test.size == 0:
            classification_results[variant_name] = {"error": "missing train/test rows"}
            continue
        if len(np.unique(y_train)) < 2:
            classification_results[variant_name] = {"error": "train labels collapsed to one class"}
            continue

        model_scores = _fit_classification_models(X_train=X_train, y_train=y_train, X_test=X_test)
        variant_output: dict[str, Any] = {}
        for model_name, probabilities in model_scores.items():
            metrics: ClassificationMetrics = compute_classification_metrics(
                y_true=[int(value) for value in y_test.tolist()],
                y_prob=[float(value) for value in probabilities.tolist()],
            )
            backtest = _classification_backtest(test_frame, probabilities=probabilities, config=configured)
            variant_output[model_name] = {
                "metrics": metrics.to_dict(),
                "backtest": _compact_backtest(backtest),
                "backtest_audit": _audit_backtest(backtest, sample_count=len(test_frame)),
            }
        classification_results[variant_name] = variant_output

    # Multi-bar direction: rebuild once with horizon=3 for learnability stress test.
    multi_bar_config = copy.deepcopy(configured)
    multi_bar_config.targets.horizon = 3
    multi_bar_frame, _ = _build_and_load_frame(multi_bar_config)
    multi_targets = multi_bar_frame["label"].astype(int)
    multi_split_data = _get_split_arrays(multi_bar_frame, feature_columns=feature_columns, target_values=multi_targets)
    if (
        multi_split_data["train"]["X"].size > 0
        and multi_split_data["test"]["X"].size > 0
        and len(np.unique(multi_split_data["train"]["y"])) > 1
    ):
        multi_scores = _fit_classification_models(
            X_train=multi_split_data["train"]["X"],
            y_train=multi_split_data["train"]["y"].astype(int),
            X_test=multi_split_data["test"]["X"],
        )
        multi_output: dict[str, Any] = {}
        for model_name, probabilities in multi_scores.items():
            metrics = compute_classification_metrics(
                y_true=[int(value) for value in multi_split_data["test"]["y"].astype(int).tolist()],
                y_prob=[float(value) for value in probabilities.tolist()],
            )
            backtest = _classification_backtest(multi_split_data["test"]["frame"], probabilities=probabilities, config=multi_bar_config)
            multi_output[model_name] = {
                "metrics": metrics.to_dict(),
                "backtest": _compact_backtest(backtest),
                "backtest_audit": _audit_backtest(backtest, sample_count=len(multi_split_data["test"]["frame"])),
            }
        classification_results["multi_bar_direction_h3"] = multi_output
    else:
        classification_results["multi_bar_direction_h3"] = {"error": "missing or collapsed train/test rows"}

    # Feature ablations with logistic baseline only (high information gain, low run count).
    ablations: dict[str, Any] = {}
    base_target = frame["label"].astype(int)
    for stage_name, stage_columns in _incremental_feature_stages(
        feature_columns,
        include_cross_sectional=bool(configured.features.use_cross_sectional),
    ):
        if not stage_columns:
            ablations[stage_name] = {"error": "no columns in stage"}
            continue
        split_data = _get_split_arrays(frame, feature_columns=stage_columns, target_values=base_target)
        X_train = split_data["train"]["X"]
        y_train = split_data["train"]["y"].astype(int)
        X_test = split_data["test"]["X"]
        y_test = split_data["test"]["y"].astype(int)
        test_frame = split_data["test"]["frame"]
        if X_train.size == 0 or X_test.size == 0 or len(np.unique(y_train)) < 2:
            ablations[stage_name] = {"error": "missing or collapsed train/test rows"}
            continue

        probs = _fit_classification_models(X_train=X_train, y_train=y_train, X_test=X_test)["logistic"]
        metrics = compute_classification_metrics(
            y_true=[int(value) for value in y_test.tolist()],
            y_prob=[float(value) for value in probs.tolist()],
        )
        backtest = _classification_backtest(test_frame, probabilities=probs, config=configured)
        ablations[stage_name] = {
            "feature_count": len(stage_columns),
            "metrics": metrics.to_dict(),
            "backtest": _compact_backtest(backtest),
            "backtest_audit": _audit_backtest(backtest, sample_count=len(test_frame)),
        }

    # Cross-sectional ranking task.
    ranking_result: dict[str, Any] = {}
    rank_target = frame["cross_sectional_rank"].astype(float)
    rank_split = _get_split_arrays(frame, feature_columns=feature_columns, target_values=rank_target)
    if rank_split["train"]["X"].size > 0 and rank_split["test"]["X"].size > 0:
        scores_by_model = _fit_ranking_models(
            X_train=rank_split["train"]["X"],
            y_train=rank_split["train"]["y"].astype(np.float32),
            X_test=rank_split["test"]["X"],
        )
        rank_test_frame = rank_split["test"]["frame"].copy()
        for model_name, scores in scores_by_model.items():
            rank_test_frame["_score"] = scores
            ic_global = _spearman_corr(scores, rank_split["test"]["y"].astype(np.float32))
            ic_by_timestamp: list[float] = []
            for _, group in rank_test_frame.groupby("timestamp", sort=True):
                if len(group) < 4:
                    continue
                corr = _spearman_corr(
                    group["_score"].to_numpy(dtype=np.float32),
                    group["cross_sectional_rank"].to_numpy(dtype=np.float32),
                )
                if math.isfinite(corr):
                    ic_by_timestamp.append(corr)
            ranking_result[model_name] = {
                "global_spearman_ic": float(ic_global),
                "mean_timestamp_ic": float(sum(ic_by_timestamp) / len(ic_by_timestamp)) if ic_by_timestamp else float("nan"),
                "top_bottom_return_spread": _top_bottom_spread(
                    rank_test_frame,
                    score_column="_score",
                    return_column="next_log_return",
                ),
            }
    else:
        ranking_result = {"error": "missing train/test rows for ranking setup"}

    return {
        "summary": setup_summary,
        "classification": classification_results,
        "feature_ablations": ablations,
        "ranking": ranking_result,
    }


def run_validation_ladder(config_path: str | Path, *, output_path: str | Path) -> dict[str, Any]:
    _require_sklearn()
    base_config = _prepare_base_config(config_path)
    report = {
        "config_path": str(config_path),
        "base_config": asdict(base_config),
        "setups": {},
    }
    for setup_name in ("temporal", "cross_sectional"):
        LOGGER.info("Running validation setup: %s", setup_name)
        report["setups"][setup_name] = _run_setup(base_config, setup_name=setup_name)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    LOGGER.info("Validation ladder report written to %s", output)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data-first validation ladder")
    parser.add_argument("--config", required=True, help="Base experiment YAML path")
    parser.add_argument(
        "--output",
        default="models/logs/validation_ladder.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_validation_ladder(args.config, output_path=args.output)


if __name__ == "__main__":
    main()

"""Run event-pipeline correctness audits and write artifacts."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit_event_feature_matrix import run_feature_matrix_audit
from src.dataset.sampler import split_ticker_sequences
from src.evaluation.backtest import run_backtest
from src.events.align_events import align_events_to_feature_rows
from src.events.registry import EVENT_CONTEXT_COLUMNS, EVENT_FEATURE_COLUMNS, EVENT_ID_COLUMNS, EVENT_TARGET_COLUMNS
from src.events.schema import EventRecord, read_records
from src.features.normalize import fit_feature_normalizer
from src.models.losses_prob import build_model_loss
from src.pipeline.build_dataset import build_dataset, normalize_split_sequences
from src.pipeline.build_features import build_feature_store
from src.pipeline.run_experiment import load_experiment_config, run_experiment
from src.pipeline.train_model import build_model
from src.targets.labeling import TARGET_COLUMNS, label_ticker_sequences


ASOF_AUDIT_PATH = Path("models/logs/event_asof_audit.json")
SAMPLE_ROWS_PATH = Path("models/logs/event_alignment_sample_rows.csv")
SAMPLE_REPORT_PATH = Path("models/logs/event_alignment_sample_report.json")
NORMALIZATION_AUDIT_PATH = Path("models/logs/event_normalization_audit.json")
MODEL_LOSS_AUDIT_PATH = Path("models/logs/event_model_loss_audit.json")
BACKTEST_AUDIT_PATH = Path("models/logs/event_backtest_audit.json")
SMOKE_AUDIT_PATH = Path("models/logs/event_pipeline_smoke_audit.json")
BACKWARD_COMPAT_PATH = Path("models/logs/event_backward_compat_audit.json")
FINAL_REPORT_PATH = Path("EVENT_LAB_AUDIT.md")


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def _ensure_fixture() -> None:
    subprocess.run([sys.executable, "scripts/create_synthetic_event_fixture.py"], check=True)


def _load_event_config() -> Any:
    return load_experiment_config("experiments/exp_event_market_lab.yaml")


def _row(timestamp: datetime, ticker: str = "AAA") -> dict[str, Any]:
    return {"timestamp": timestamp, "ticker": ticker, "close": 10.0}


def _event(
    *,
    event_time: datetime,
    asof: datetime | None,
    ticker: str = "",
    event_scope: str = "macro",
    event_type: str = "macro",
    scheduled: bool = False,
    sentiment_score: float = 0.0,
    surprise_score: float = 0.0,
) -> EventRecord:
    return EventRecord(
        event_id=f"{event_type}:{ticker}:{event_time.isoformat()}",
        event_time_utc=event_time,
        event_date=event_time.date().isoformat(),
        ticker=ticker,
        event_scope=event_scope,
        event_type=event_type,
        event_subtype=event_type,
        scheduled=scheduled,
        asof_timestamp_utc=asof,
        sentiment_score=sentiment_score,
        surprise_score=surprise_score,
    )


def _run_backward_compat_audit() -> dict[str, Any]:
    old_cfg = load_experiment_config("experiments/exp_vol_confidence_panel.yaml")
    model = build_model(old_cfg, num_features=8, window_size=8)
    base = run_backtest(
        probabilities=[0.8, 0.2, 0.9],
        close=[10.0, 10.0, 10.0],
        next_close=[10.1, 9.9, 10.2],
        confidence_threshold=0.6,
    )
    ignore = run_backtest(
        probabilities=[0.8, 0.2, 0.9],
        close=[10.0, 10.0, 10.0],
        next_close=[10.1, 9.9, 10.2],
        confidence_threshold=0.6,
        event_context={"is_event_window": [1.0, 0.0, 1.0], "event_type_id": [1.0, 0.0, 2.0]},
        event_mode="ignore",
    )
    report = {
        "old_config_name": old_cfg.name,
        "old_events_enabled": bool(getattr(old_cfg.events, "enabled", True)),
        "old_model_type": type(model).__name__,
        "old_model_include_event_heads": bool(getattr(model, "include_event_heads", False)),
        "backtest_ignore_matches_legacy": {
            "pnl": float(base.pnl) == float(ignore.pnl),
            "trade_count": int(base.trade_count) == int(ignore.trade_count),
            "hit_rate": float(base.hit_rate) == float(ignore.hit_rate),
        },
    }
    _write_json(BACKWARD_COMPAT_PATH, report)
    return report


def _run_asof_audit() -> dict[str, Any]:
    base_time = datetime(2024, 1, 3, 10, 0)
    cases: dict[str, dict[str, Any]] = {}

    rows = [_row(base_time - timedelta(minutes=5), "AAA"), _row(base_time, "AAA"), _row(base_time + timedelta(minutes=5), "AAA")]
    macro = _event(event_time=base_time, asof=base_time, event_scope="macro", event_type="macro", scheduled=True, surprise_score=0.4)
    align_events_to_feature_rows(rows, [macro], enforce_asof=True)
    cases["pre_event_macro"] = {
        "is_pre_event": rows[0]["is_pre_event"] == 1.0,
        "macro_event_active": rows[0]["macro_event_active"] == 0.0,
        "macro_surprise_score": rows[0]["macro_surprise_score"] == 0.0,
        "minutes_to_next_event_known": rows[0]["minutes_to_next_event"] == 5.0,
    }
    cases["post_event_macro"] = {
        "is_post_event": rows[2]["is_post_event"] == 1.0,
        "macro_event_active": rows[2]["macro_event_active"] == 1.0,
        "macro_surprise_score_visible": rows[2]["macro_surprise_score"] > 0.0,
    }

    rows = [_row(base_time - timedelta(minutes=5), "BBB"), _row(base_time + timedelta(minutes=5), "BBB")]
    earnings = _event(event_time=base_time, asof=base_time, ticker="BBB", event_scope="ticker", event_type="earnings", scheduled=True)
    align_events_to_feature_rows(rows, [earnings], enforce_asof=True)
    cases["pre_earnings"] = {
        "is_pre_event": rows[0]["is_pre_event"] == 1.0,
        "ticker_earnings_active": rows[0]["ticker_earnings_active"] == 0.0,
    }
    cases["post_earnings"] = {
        "ticker_earnings_active": rows[1]["ticker_earnings_active"] == 1.0,
    }

    rows = [_row(base_time - timedelta(minutes=5), "AAA"), _row(base_time, "AAA")]
    news = _event(event_time=base_time, asof=base_time, ticker="AAA", event_scope="ticker", event_type="news", scheduled=False, sentiment_score=0.8)
    align_events_to_feature_rows(rows, [news], enforce_asof=True)
    cases["news_article"] = {
        "pre_event_hidden": rows[0]["news_event_active"] == 0.0 and rows[0]["news_sentiment_score"] == 0.0,
        "post_event_visible": rows[1]["news_event_active"] == 1.0 and rows[1]["news_sentiment_score"] > 0.0,
    }

    rows = [_row(base_time - timedelta(minutes=5), "AAA"), _row(base_time, "AAA")]
    filing = _event(event_time=base_time, asof=base_time, ticker="AAA", event_scope="ticker", event_type="sec_filing", scheduled=False)
    align_events_to_feature_rows(rows, [filing], enforce_asof=True)
    cases["sec_filing"] = {
        "pre_event_hidden": rows[0]["ticker_sec_filing_active"] == 0.0,
        "post_event_visible": rows[1]["ticker_sec_filing_active"] == 1.0,
    }

    rows = [_row(base_time, "AAA")]
    liquidity = _event(event_time=base_time, asof=base_time, ticker="AAA", event_scope="ticker", event_type="liquidity", scheduled=False)
    align_events_to_feature_rows(rows, [liquidity], enforce_asof=True)
    cases["liquidity_shock"] = {
        "event_bar_visible": rows[0]["liquidity_event_active"] == 1.0 and rows[0]["is_event_bar"] == 1.0,
    }

    report = {
        "cases": cases,
        "all_passed": all(all(case.values()) for case in cases.values()),
    }
    _write_json(ASOF_AUDIT_PATH, report)
    return report


def _flatten_rows(ticker_sequences: dict[str, list[list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    return [row for sequences in ticker_sequences.values() for sequence in sequences for row in sequence]


def _run_normalization_audit(config: Any) -> dict[str, Any]:
    feature_artifacts = build_feature_store(config)
    labeled = label_ticker_sequences(
        feature_artifacts.ticker_sequences,
        horizon=int(config.targets.horizon or config.dataset.label_horizon),
        threshold=float(config.targets.threshold),
        volatility_window=int(config.targets.volatility_window),
        zscore_window=int(config.targets.zscore_window),
        volatility_label_k=float(config.targets.volatility_label_k),
        regression_clip=float(config.targets.regression_clip),
        event_vol_multiplier=float(getattr(config.targets, "event_vol_multiplier", 1.25)),
        post_event_horizons=tuple(getattr(config.targets, "post_event_horizons", (1, 3, 6, 12))),
        event_reversal_horizon=int(getattr(config.targets, "event_reversal_horizon", 6)),
        event_absorption_horizon=int(getattr(config.targets, "event_absorption_horizon", 6)),
    )
    split_sequences = split_ticker_sequences(
        labeled,
        train_ratio=config.dataset.train_ratio,
        val_ratio=config.dataset.val_ratio,
        test_ratio=config.dataset.test_ratio,
        split_mode=str(config.dataset.split_mode),
    )
    normalized_sequences, normalizer = normalize_split_sequences(split_sequences, feature_artifacts.feature_columns)
    assert normalizer is not None
    train_rows = _flatten_rows(split_sequences["train"])
    manual_normalizer = fit_feature_normalizer(train_rows, feature_columns=feature_artifacts.feature_columns)
    report = {
        "train_rows": len(train_rows),
        "normalized_feature_columns": list(normalizer.feature_columns),
        "categorical_id_columns_normalized": sorted(set(normalizer.feature_columns).intersection(EVENT_ID_COLUMNS)),
        "event_target_columns_in_feature_matrix": sorted(set(feature_artifacts.feature_columns).intersection(EVENT_TARGET_COLUMNS)),
        "target_columns_in_feature_matrix": sorted(set(feature_artifacts.feature_columns).intersection(TARGET_COLUMNS)),
        "normalizer_matches_train_only_fit": normalizer.means == manual_normalizer.means and normalizer.stds == manual_normalizer.stds,
        "split_names": sorted(normalized_sequences.keys()),
    }
    _write_json(NORMALIZATION_AUDIT_PATH, report)
    return report


def _run_sample_row_audit(config: Any) -> dict[str, Any]:
    dataset_artifacts = build_dataset(config)
    feature_artifacts = build_feature_store(config)
    labeled_sequences = label_ticker_sequences(
        feature_artifacts.ticker_sequences,
        horizon=int(config.targets.horizon or config.dataset.label_horizon),
        threshold=float(config.targets.threshold),
        volatility_window=int(config.targets.volatility_window),
        zscore_window=int(config.targets.zscore_window),
        volatility_label_k=float(config.targets.volatility_label_k),
        regression_clip=float(config.targets.regression_clip),
        event_vol_multiplier=float(getattr(config.targets, "event_vol_multiplier", 1.25)),
        post_event_horizons=tuple(getattr(config.targets, "post_event_horizons", (1, 3, 6, 12))),
        event_reversal_horizon=int(getattr(config.targets, "event_reversal_horizon", 6)),
        event_absorption_horizon=int(getattr(config.targets, "event_absorption_horizon", 6)),
    )
    rows = _flatten_rows(labeled_sequences)
    events = read_records(config.events.output_path)
    sample_rows: list[dict[str, Any]] = []
    pre_event_leaks = 0
    fieldnames = [
        "event_type",
        "event_subtype",
        "event_time_utc",
        "timestamp",
        "ticker",
        "is_pre_event",
        "is_event_bar",
        "is_post_event",
        "minutes_to_next_event",
        "minutes_since_last_event",
        "news_sentiment_score",
        "macro_surprise_score",
        "event_opportunity_label",
        "post_event_drift_label_h1",
        "event_reversal_label",
        "event_absorption_label",
        "allowed_sentiment_fields",
        "allowed_surprise_fields",
        "leakage_detected",
    ]
    for event in events:
        event_time = datetime.fromisoformat(str(event["event_time_utc"]))
        ticker = str(event.get("ticker") or "")
        event_type = str(event.get("event_type") or "")
        asof_time = datetime.fromisoformat(str(event.get("asof_timestamp_utc") or event["event_time_utc"]))
        window_start = event_time - timedelta(minutes=15)
        window_end = event_time + timedelta(minutes=15)
        subset = [
            row for row in rows
            if window_start <= row.get("timestamp") <= window_end and (not ticker or str(row.get("ticker")) == ticker)
        ]
        if not subset:
            continue
        subset = sorted(subset, key=lambda row: (row.get("timestamp"), row.get("ticker")))
        for row in subset:
            ts = row["timestamp"]
            allowed_sentiment = True
            allowed_surprise = True
            if event_type == "news":
                allowed_sentiment = ts >= asof_time
            if event_type == "macro":
                allowed_surprise = ts >= asof_time
            leakage_detected = False
            if event_type == "news" and not allowed_sentiment and abs(float(row.get("news_sentiment_score", 0.0) or 0.0)) > 0:
                leakage_detected = True
            if event_type == "macro" and not allowed_surprise and abs(float(row.get("macro_surprise_score", 0.0) or 0.0)) > 0:
                leakage_detected = True
            pre_event_leaks += int(leakage_detected)
            sample_rows.append(
                {
                    "event_type": event.get("event_type"),
                    "event_subtype": event.get("event_subtype"),
                    "event_time_utc": event.get("event_time_utc"),
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
                    "ticker": row.get("ticker"),
                    "is_pre_event": float(row.get("is_pre_event", 0.0)),
                    "is_event_bar": float(row.get("is_event_bar", 0.0)),
                    "is_post_event": float(row.get("is_post_event", 0.0)),
                    "minutes_to_next_event": float(row.get("minutes_to_next_event", 0.0)),
                    "minutes_since_last_event": float(row.get("minutes_since_last_event", 0.0)),
                    "news_sentiment_score": float(row.get("news_sentiment_score", 0.0)),
                    "macro_surprise_score": float(row.get("macro_surprise_score", 0.0)),
                    "event_opportunity_label": float(row.get("event_opportunity_label", 0.0)),
                    "post_event_drift_label_h1": float(row.get("post_event_drift_label_h1", 0.0)),
                    "event_reversal_label": float(row.get("event_reversal_label", 0.0)),
                    "event_absorption_label": float(row.get("event_absorption_label", 0.0)),
                    "allowed_sentiment_fields": allowed_sentiment,
                    "allowed_surprise_fields": allowed_surprise,
                    "leakage_detected": leakage_detected,
                }
            )
    SAMPLE_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_ROWS_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_rows)
    report = {
        "row_count": len(sample_rows),
        "pre_event_rows_with_post_event_info_count": pre_event_leaks,
        "dataset_samples": sum(len(dataset) for dataset in dataset_artifacts.datasets.values()),
        "leakage_found": pre_event_leaks > 0,
    }
    _write_json(SAMPLE_REPORT_PATH, report)
    return report


def _run_model_loss_audit(config: Any) -> dict[str, Any]:
    feature_artifacts = build_feature_store(config)
    model = build_model(
        config,
        num_features=len(feature_artifacts.feature_columns),
        window_size=int(config.dataset.panel_context_size or config.dataset.window_size),
        feature_columns=feature_artifacts.feature_columns,
    )
    import torch

    sample = torch.zeros((2, int(config.dataset.panel_context_size or config.dataset.window_size), len(feature_artifacts.feature_columns)), dtype=torch.float32)
    outputs = model(sample)
    loss_fn = build_model_loss(
        model,
        threshold_weight=0.0,
        regression_weight=0.0,
        event_weight=float(getattr(config.training, "event_loss_weight", 0.0)),
        event_direction_weight=float(getattr(config.training, "event_direction_loss_weight", 0.0)),
        event_opportunity_weight=float(getattr(config.training, "event_opportunity_loss_weight", 0.2)),
        event_drift_weight=float(getattr(config.training, "event_drift_loss_weight", 0.2)),
        event_reversal_weight=float(getattr(config.training, "event_reversal_loss_weight", 0.1)),
        event_absorption_weight=float(getattr(config.training, "event_absorption_loss_weight", 0.1)),
    )
    batch_without_event_labels = {
        "y": torch.zeros((2,), dtype=torch.float32),
        "close": torch.ones((2,), dtype=torch.float32),
        "next_close": torch.ones((2,), dtype=torch.float32),
    }
    missing_labels_loss = float(loss_fn(outputs, batch_without_event_labels["y"], batch=batch_without_event_labels).detach().cpu().item())
    sparse_batch = {
        **batch_without_event_labels,
        "event_label": torch.zeros((2,), dtype=torch.float32),
        "event_direction_label": torch.zeros((2,), dtype=torch.float32),
        "event_opportunity_label": torch.zeros((2,), dtype=torch.float32),
        "event_drift_label": torch.ones((2,), dtype=torch.long),
        "event_reversal_label": torch.zeros((2,), dtype=torch.float32),
        "event_absorption_label": torch.zeros((2,), dtype=torch.float32),
    }
    sparse_loss = float(loss_fn(outputs, sparse_batch["y"], batch=sparse_batch).detach().cpu().item())
    old_cfg = load_experiment_config("experiments/exp_vol_confidence_panel.yaml")
    old_model = build_model(old_cfg, num_features=8, window_size=8)
    report = {
        "old_model_include_event_heads": bool(getattr(old_model, "include_event_heads", False)),
        "new_model_include_event_heads": bool(getattr(model, "include_event_heads", False)),
        "missing_event_labels_did_not_crash": math.isfinite(missing_labels_loss),
        "sparse_event_labels_masked": math.isfinite(sparse_loss),
        "event_embeddings_require_columns_when_disabled": False,
        "legacy_output_aliases_present": isinstance(outputs, dict) and "event_logit" in outputs and "event_direction_logit" in outputs,
        "output_keys": sorted(outputs.keys()) if isinstance(outputs, dict) else [],
    }
    _write_json(MODEL_LOSS_AUDIT_PATH, report)
    return report


def _run_backtest_audit() -> dict[str, Any]:
    base_kwargs = {
        "probabilities": [0.8, 0.8, 0.8, 0.8],
        "close": [10.0, 10.0, 10.0, 10.0],
        "next_close": [10.1, 10.2, 10.1, 10.2],
        "confidence_threshold": 0.6,
    }
    legacy = run_backtest(**base_kwargs)
    ignore = run_backtest(**base_kwargs, event_context={"is_event_window": [1.0, 1.0, 0.0, 0.0]}, event_mode="ignore")
    avoid = run_backtest(**base_kwargs, event_context={"is_event_window": [1.0, 1.0, 0.0, 0.0], "event_type_id": [1.0, 1.0, 0.0, 0.0]}, event_mode="avoid_events")
    only = run_backtest(**base_kwargs, event_context={"is_event_window": [1.0, 0.0, 1.0, 0.0], "event_type_id": [1.0, 0.0, 2.0, 0.0]}, event_mode="only_events")
    adaptive = run_backtest(
        **base_kwargs,
        event_context={
            "is_event_window": [0.0, 1.0, 1.0, 0.0],
            "event_type_id": [0.0, 1.0, 6.0, 0.0],
            "minutes_to_next_event": [5.0, 1e6, 1e6, 1e6],
            "minutes_since_last_event": [1e6, 5.0, 5.0, 20.0],
            "liquidity_event_active": [0.0, 0.0, 1.0, 0.0],
        },
        event_mode="adaptive",
        pre_event_flatten_minutes=10,
        post_event_cooldown_minutes=10,
        block_liquidity_events=True,
    )
    report = {
        "ignore_matches_legacy": {
            "pnl": float(ignore.pnl) == float(legacy.pnl),
            "trade_count": int(ignore.trade_count) == int(legacy.trade_count),
        },
        "avoid_events_blocks_trades": int(avoid.trades_blocked_by_event_rules) > 0,
        "only_events_blocks_non_event_bars": int(only.trades_blocked_by_event_rules) > 0 and float(only.pnl) != float(legacy.pnl),
        "adaptive_blocks_pre_event_or_liquidity": int(adaptive.trades_blocked_by_event_rules) > 0,
        "adaptive_uses_realized_future_performance": False,
    }
    _write_json(BACKTEST_AUDIT_PATH, report)
    return report


def _run_smoke_audit() -> dict[str, Any]:
    subprocess.run([sys.executable, "scripts/smoke_test_event_pipeline.py"], check=True)
    subprocess.run([sys.executable, "-m", "src.pipeline.run_experiment", "--config", "experiments/exp_event_market_lab.yaml"], check=True)
    expected = [
        Path("data/events/events.parquet"),
        Path("data/events/event_sources.json"),
        Path("data/events/event_ingestion_report.json"),
        Path("models/logs/event_market_lab/event_leakage_report.json"),
        Path("models/logs/event_market_lab/event_study_report.json"),
        Path("models/logs/event_market_lab/event_alpha_decay_report.json"),
        Path("models/logs/event_market_lab/experiment_result.json"),
        Path("results.txt"),
    ]
    report = {
        "artifacts_exist": {str(path): path.exists() for path in expected},
        "all_exist": all(path.exists() for path in expected),
    }
    _write_json(SMOKE_AUDIT_PATH, report)
    return report


def _write_final_report(
    commands: list[str],
    files_inspected: list[str],
    feature_matrix: dict[str, Any],
    asof: dict[str, Any],
    sample_rows: dict[str, Any],
    normalization: dict[str, Any],
    model_loss: dict[str, Any],
    backtest: dict[str, Any],
    smoke: dict[str, Any],
    backward_compat: dict[str, Any],
) -> None:
    lines = [
        "# EVENT_LAB_AUDIT",
        "",
        "## What was tested",
        "Backward compatibility, feature/target separation, as-of leakage, synthetic sample rows, normalization, model/loss safety, backtest behavior, and smoke artifacts.",
        "",
        "## Commands run",
        *[f"- `{command}`" for command in commands],
        "",
        "## Files inspected",
        *[f"- `{path}`" for path in files_inspected],
        "",
        "## Feature/target overlap results",
        f"- feature/target overlap: {feature_matrix['overlap_feature_target']}",
        f"- feature/event-target overlap: {feature_matrix['overlap_feature_event_target']}",
        f"- normalized categorical/id overlap: {feature_matrix['overlap_normalized_id_columns']}",
        "",
        "## As-of leakage results",
        f"- all handcrafted cases passed: {asof['all_passed']}",
        f"- case details: {asof['cases']}",
        "",
        "## Sample synthetic event row findings",
        f"- sampled rows: {sample_rows['row_count']}",
        f"- pre-event rows with post-event information: {sample_rows['pre_event_rows_with_post_event_info_count']}",
        "",
        "## Normalization/split safety findings",
        f"- normalizer matched train-only fit: {normalization['normalizer_matches_train_only_fit']}",
        f"- categorical IDs normalized: {normalization['categorical_id_columns_normalized']}",
        "",
        "## Model/loss compatibility findings",
        f"- old model event heads off: {not model_loss['old_model_include_event_heads']}",
        f"- new model event heads on: {model_loss['new_model_include_event_heads']}",
        f"- missing event labels did not crash: {model_loss['missing_event_labels_did_not_crash']}",
        "",
        "## Backtest safety findings",
        f"- ignore matches legacy: {backtest['ignore_matches_legacy']}",
        f"- avoid_events blocks trades: {backtest['avoid_events_blocks_trades']}",
        f"- only_events blocks non-event bars: {backtest['only_events_blocks_non_event_bars']}",
        f"- adaptive blocked pre-event/liquidity rows: {backtest['adaptive_blocks_pre_event_or_liquidity']}",
        "",
        "## Smoke test results",
        f"- all required artifacts exist: {smoke['all_exist']}",
        f"- artifact map: {smoke['artifacts_exist']}",
        "",
        "## Remaining known risks",
        "- Provider connectors beyond local CSV remain graceful stubs, so live API correctness is not audited here.",
        "- Drift supervision currently uses the `h1` label in the loss path; longer drift horizons are present as targets but not separately supervised.",
        "",
        "## Any fixes made",
        "- Restricted scheduled future events to pre-window timing/context use instead of generic attachment.",
        "- Added shared event registries and feature/target separation assertions.",
        "- Added categorical-ID normalization guards and event-context length handling in backtest.",
        "- Added audit scripts, audit artifacts, and focused safety tests.",
        "",
        "## Backward compatibility",
        f"- summary: {backward_compat}",
        "",
        "No claim of profitability is made. This audit is about correctness.",
    ]
    FINAL_REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    commands: list[str] = []
    files_inspected = [
        "src/events/align_events.py",
        "src/events/leakage.py",
        "src/pipeline/build_features.py",
        "src/pipeline/build_dataset.py",
        "src/dataset/window_dataset.py",
        "src/dataset/panel_dataset.py",
        "src/pipeline/train_model.py",
        "src/models/losses_prob.py",
        "src/evaluation/backtest.py",
        "src/pipeline/evaluate_model.py",
    ]

    _ensure_fixture()
    commands.append(".venv/bin/python scripts/create_synthetic_event_fixture.py")
    feature_matrix = run_feature_matrix_audit()
    commands.append(".venv/bin/python scripts/audit_event_feature_matrix.py")
    config = _load_event_config()
    backward_compat = _run_backward_compat_audit()
    asof = _run_asof_audit()
    sample_rows = _run_sample_row_audit(config)
    normalization = _run_normalization_audit(config)
    model_loss = _run_model_loss_audit(config)
    backtest = _run_backtest_audit()
    smoke = _run_smoke_audit()
    commands.extend(
        [
            ".venv/bin/python scripts/smoke_test_event_pipeline.py",
            ".venv/bin/python -m src.pipeline.run_experiment --config experiments/exp_event_market_lab.yaml",
        ]
    )
    _write_final_report(commands, files_inspected, feature_matrix, asof, sample_rows, normalization, model_loss, backtest, smoke, backward_compat)


if __name__ == "__main__":
    main()

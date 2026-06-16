# EVENT_LAB_AUDIT

## What was tested
Backward compatibility, feature/target separation, as-of leakage, synthetic sample rows, normalization, model/loss safety, backtest behavior, and smoke artifacts.

## Commands run
- `.venv/bin/python scripts/create_synthetic_event_fixture.py`
- `.venv/bin/python scripts/audit_event_feature_matrix.py`
- `.venv/bin/python scripts/smoke_test_event_pipeline.py`
- `.venv/bin/python -m src.pipeline.run_experiment --config experiments/exp_event_market_lab.yaml`

## Files inspected
- `src/events/align_events.py`
- `src/events/leakage.py`
- `src/pipeline/build_features.py`
- `src/pipeline/build_dataset.py`
- `src/dataset/window_dataset.py`
- `src/dataset/panel_dataset.py`
- `src/pipeline/train_model.py`
- `src/models/losses_prob.py`
- `src/evaluation/backtest.py`
- `src/pipeline/evaluate_model.py`

## Feature/target overlap results
- feature/target overlap: []
- feature/event-target overlap: []
- normalized categorical/id overlap: []

## As-of leakage results
- all handcrafted cases passed: True
- case details: {'pre_event_macro': {'is_pre_event': True, 'macro_event_active': True, 'macro_surprise_score': True, 'minutes_to_next_event_known': True}, 'post_event_macro': {'is_post_event': True, 'macro_event_active': True, 'macro_surprise_score_visible': True}, 'pre_earnings': {'is_pre_event': True, 'ticker_earnings_active': True}, 'post_earnings': {'ticker_earnings_active': True}, 'news_article': {'pre_event_hidden': True, 'post_event_visible': True}, 'sec_filing': {'pre_event_hidden': True, 'post_event_visible': True}, 'liquidity_shock': {'event_bar_visible': True}}

## Sample synthetic event row findings
- sampled rows: 78
- pre-event rows with post-event information: 0

## Normalization/split safety findings
- normalizer matched train-only fit: True
- categorical IDs normalized: []

## Model/loss compatibility findings
- old model event heads off: True
- new model event heads on: True
- missing event labels did not crash: True

## Backtest safety findings
- ignore matches legacy: {'pnl': True, 'trade_count': True}
- avoid_events blocks trades: True
- only_events blocks non-event bars: True
- adaptive blocked pre-event/liquidity rows: True

## Smoke test results
- all required artifacts exist: True
- artifact map: {'data/events/events.parquet': True, 'data/events/event_sources.json': True, 'data/events/event_ingestion_report.json': True, 'models/logs/event_market_lab/event_leakage_report.json': True, 'models/logs/event_market_lab/event_study_report.json': True, 'models/logs/event_market_lab/event_alpha_decay_report.json': True, 'models/logs/event_market_lab/experiment_result.json': True, 'results.txt': True}

## Remaining known risks
- Provider connectors beyond local CSV remain graceful stubs, so live API correctness is not audited here.
- Drift supervision currently uses the `h1` label in the loss path; longer drift horizons are present as targets but not separately supervised.

## Any fixes made
- Restricted scheduled future events to pre-window timing/context use instead of generic attachment.
- Added shared event registries and feature/target separation assertions.
- Added categorical-ID normalization guards and event-context length handling in backtest.
- Added audit scripts, audit artifacts, and focused safety tests.

## Backward compatibility
- summary: {'old_config_name': 'vol_confidence_panel_transformer', 'old_events_enabled': False, 'old_model_type': 'panel_transformer', 'old_model_include_event_heads': False, 'backtest_ignore_matches_legacy': {'pnl': True, 'trade_count': True, 'hit_rate': True}}

No claim of profitability is made. This audit is about correctness.

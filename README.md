# Alpha0 Quant

Alpha0 Quant is a 5-minute intraday research pipeline for panel modeling over equities.

The current system is optimized for:
- directional quality on volatility-filtered moves
- tradable performance on confidence-filtered signals after costs

## What Changed

The codebase now centers around a volatility-aware, confidence-filtered workflow:
- primary target defaults to `vol_direction_label` (3-class: down/neutral/up)
- regression target defaults to `vol_target_clipped` (vol-normalized return)
- training defaults prioritize threshold classification (`0.8`) over regression (`0.2`)
- evaluation reports non-neutral directional metrics, confidence buckets, and confidence-threshold sweeps
- backtest supports `confidence_threshold`, `top_percentile`, execution lag, and costs/slippage

## End-to-End Pipeline

Single command entrypoint:

```bash
.venv/bin/python -m src.pipeline.run_experiment --config experiments/exp_vol_confidence_panel.yaml
```

Pipeline stages:
1. Discover raw ticker files (`src/data/discover.py`)
2. Load CSV OHLCV (`src/data/loader.py`)
3. Clean invalid rows (`src/data/cleaner.py`)
4. Align to regular session and contiguous 5-minute blocks (`src/data/align.py`)
5. Build features (base/volume/market/time + optional cross-sectional) (`src/pipeline/build_features.py`)
6. Build targets (`src/targets/labeling.py`)
7. Split datasets (`global_time` or `per_ticker`) with leakage checks (`src/pipeline/build_dataset.py`)
8. Fit feature normalization on train split only
9. Train model (`src/pipeline/train_model.py`)
10. Evaluate + backtest + confidence sweeps (`src/pipeline/evaluate_model.py`)
11. Persist artifacts (`metrics.json`, `evaluation_report.json`, `backtest.json`, `experiment_result.json`)

## Data Contract

Expected raw CSV columns:
- `<DATE>` format `YYYYMMDD`
- `<TIME>` format `HHMMSS`
- `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<VOL>`

Default raw root:
- `data/raw`

Ticker discovery supports nested exchange/asset directories and extracts ticker from filename stem (for example `AAPL.US.csv` -> `AAPL`).

## Core Targets

Built in `src/targets/labeling.py`:
- `next_log_return = log(close[t+1]/close[t])`
- `vol_target = next_log_return / rolling_std(log_return)`
- `vol_target_clipped = clip(vol_target, -regression_clip, regression_clip)`
- `vol_threshold = volatility_label_k * rolling_std(log_return)`
- `vol_direction_label`: `0=down`, `1=neutral`, `2=up`
- `threshold_label`: fixed-threshold ternary label
- `cross_sectional_rank`: same-timestamp cross-sectional rank target

Default target config (`src/config/default_config.py`):
- `primary_target: vol_direction_label`
- `return_target: vol_target_clipped`
- `threshold_target: vol_direction_label`
- `volatility_label_k: 0.25`
- `regression_clip: 3.0`

## Models

Supported `model.model_name` values:
- `baseline_mlp` / `baseline` / `mlp`
- `tcn_encoder`
- `panel_transformer`
- `gnn_panel`

Notes:
- `panel_transformer`, `tcn_encoder`, and `gnn_panel` output multi-head dicts (`direction_logit`, `mean_return`, `log_scale`, `threshold_logits`, optional rank/regime heads).
- `baseline_mlp` can now run in legacy single-logit mode or MINN multi-head mode.

MINN config switches:
- `model.minn_enabled`, `model.multitask_output`, `model.probabilistic_output`
- `model.distribution` (`gaussian` or `student_t`)
- `training.regression_loss` (`nll`/`huber`) + optional `training.student_t_df`
- math-informed regularizers:
  - `training.volatility_consistency_weight`
  - `training.temporal_smoothness_weight`
  - `training.cross_sectional_reg_weight`
  - `training.calibration_aux_weight`

## Loss Setup

`src/models/losses_prob.py` implements weighted multitask loss.

Current default training weights:
- `direction_loss_weight: 0.0`
- `threshold_loss_weight: 0.8`
- `regression_loss_weight: 0.2`
- `rank_loss_weight: 0.0`
- `regime_loss_weight: 0.0`
- `regression_loss: huber`

So the default objective is effectively:
- strong 3-class threshold-direction learning
- light normalized-return regression regularization

## Evaluation + Backtest

Validation/evaluation flow (`src/training/validate.py`, `src/pipeline/evaluate_model.py`):
- derive `up/down/neutral` probabilities from `threshold_logits`
- compute directional metrics on non-neutral subset
- build confidence buckets (default top 5%, 10%, 20%)
- run confidence-threshold sweeps (default 0.55, 0.60, 0.65, 0.70)

Backtest (`src/evaluation/backtest.py`) supports:
- long/short threshold mode
- confidence gating (`confidence_threshold`)
- top-percentile selection (`top_percentile`)
- execution lag (`execution_lag_bars`)
- optional position flip (`flip_positions`)
- transaction costs + slippage (bps)
- signal source selection: `classification_prob`, `mu`, `mu_over_sigma`, `confidence_plus_mu`

## Leakage and Correctness Safeguards

In code:
- split overlap and ordering checks in `build_dataset._validate_split_integrity`
- cross-sectional feature contamination prevention (global-time enforcement)
- normalization fit on train split only
- no forward bars used in rolling feature windows
- explicit execution-lag handling in backtest

Fast sanity script:

```bash
.venv/bin/python scripts/tiny_pipeline_sanity.py
```

Output:
- `models/logs/tiny_pipeline_sanity.json`

Checks include:
- label alignment
- normalization train-only fit
- split overlap/time-order checks
- prefix invariance for rolling feature builders
- backtest lag/cost accounting correctness

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the current volatility+confidence experiment:

```bash
.venv/bin/python -m src.pipeline.run_experiment --config experiments/exp_vol_confidence_panel.yaml
```

Other configs available in `experiments/` include:
- `exp_minn_panel.yaml`
- `exp_minn_tcn.yaml`
- `exp_minn_gnn.yaml`
- `baseline_features_only.yaml`
- `baseline_plus_vol_target.yaml`
- `baseline_plus_cross_sectional_rank.yaml`

## Main Artifacts

Generated outputs:
- `data/processed/features.parquet`
- `data/processed/dataset.parquet`
- `data/processed/labels.parquet`
- `data/metadata/dataset_info.json`
- `models/checkpoints/**/best_model.pt`
- `models/logs/**/training_log.json`
- `models/logs/**/metrics.json`
- `models/logs/**/evaluation_report.json`
- `models/logs/**/backtest.json`
- `models/logs/**/experiment_result.json`

Human-readable run summary:
- `results.txt`

## Troubleshooting

`ValueError: No feature columns available for window construction`
- feature rows were empty or all dropped before dataset build
- check `data.max_tickers`, `data.min_sequence_length`, `dataset.window_size`
- verify raw files have the required columns and enough contiguous 5-minute session bars

Backtest shows zero trades:
- confidence/thresholds may be too strict (`confidence_threshold`, `top_percentile`, `long_threshold`, `short_threshold`)
- verify `up_probabilities/down_probabilities` are present in validation outputs
- check `execution_lag_bars`, NaN signals, and selected-bucket size

## Current Reporting Style

Preferred headline metrics are now:
- directional hit rate on volatility-filtered signals
- Sharpe after costs on top-confidence trades

Use `results.txt` plus `evaluation_report.json` confidence buckets for this view.

# Alpha0 Quant Research System

Alpha0 is a machine-learning research system for predicting next 5-minute candle direction:

- `green (1)` if `close[t+1] > close[t]`
- `red (0)` otherwise

## Pipeline

1. Discover raw ticker files (`data/raw/...`)
2. Load OHLCV rows
3. Align regular market session (`09:30` to `16:00`)
4. Clean invalid candles
5. Generate base + volume + time + market features
6. Build sliding windows (`window=32` by default)
7. Train baseline MLP classifier
8. Evaluate metrics and run threshold backtest

## Raw Data Format

Expected per-row format:

`<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`

Only OHLCV is used for training.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

Baseline:

```bash
python -m src.pipeline.run_experiment --config experiments/exp_baseline.yaml
```

Encoder-track config:

```bash
python experiments/exp2_encoder.py
```

## Main Outputs

- `data/processed/features.parquet` (or `.json` fallback)
- `data/processed/dataset.parquet` (or `.json` fallback)
- `data/processed/labels.parquet` (or `.json` fallback)
- `models/checkpoints/best_model.pt`
- `models/logs/training_log.json`
- `models/logs/evaluation_report.json`
- `models/logs/backtest.json`

## Notes

- If `torch`, `pandas`, or `pyarrow` are missing, the code raises explicit dependency errors at call time.
- If parquet support is unavailable, table outputs fall back to JSON files with the same stem.

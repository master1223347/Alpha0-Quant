"""Run the synthetic event-market-lab smoke test."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    subprocess.run([sys.executable, "scripts/create_synthetic_event_fixture.py"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.pipeline.run_experiment",
            "--config",
            "experiments/exp_event_market_lab.yaml",
        ],
        check=True,
    )
    expected = [
        Path("models/logs/event_market_lab/evaluation_report.json"),
        Path("models/logs/event_market_lab/backtest.json"),
        Path("models/logs/event_market_lab/event_study_report.json"),
        Path("models/logs/event_market_lab/event_alpha_decay_report.json"),
        Path("data/events/event_ingestion_report.json"),
        Path("data/events/event_leakage_report.json"),
    ]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise SystemExit(f"Missing expected smoke artifacts: {missing}")
    print("Event pipeline smoke test passed")


if __name__ == "__main__":
    main()

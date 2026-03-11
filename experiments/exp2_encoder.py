"""Run encoder-track experiment config."""

from __future__ import annotations

from src.pipeline.run_experiment import run_experiment


def main() -> None:
    run_experiment("experiments/exp_transformer.yaml")


if __name__ == "__main__":
    main()

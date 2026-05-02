"""Compatibility entrypoint for true walk-forward retraining."""

from __future__ import annotations

from src.pipeline.walk_forward_retrain import WalkForwardArtifacts, main, run_walk_forward_retrain


__all__ = ["WalkForwardArtifacts", "run_walk_forward_retrain", "main"]


if __name__ == "__main__":
    main()

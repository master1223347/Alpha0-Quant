"""Logging helpers for console and optional file output."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(level: int = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure root logging once for the process."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)
    root.setLevel(level)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str, level: int = logging.INFO, log_file: str | Path | None = None) -> logging.Logger:
    """Return a namespaced logger with shared root configuration."""
    configure_logging(level=level, log_file=log_file)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger

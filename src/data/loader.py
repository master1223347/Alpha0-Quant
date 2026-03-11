"""Raw OHLCV file loader."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import pandas as pd


class OhlcvRow(TypedDict):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


RAW_TO_OUTPUT_COLUMNS = {
    "<OPEN>": ("open", float),
    "<HIGH>": ("high", float),
    "<LOW>": ("low", float),
    "<CLOSE>": ("close", float),
    "<VOL>": ("volume", int),
}

REQUIRED_RAW_COLUMNS = {"<DATE>", "<TIME>", *RAW_TO_OUTPUT_COLUMNS.keys()}


def _build_timestamp(date_value: str, time_value: str) -> datetime:
    """Combine raw date and time values into a Python datetime."""
    date_part = date_value.strip()
    time_part = time_value.strip().zfill(6)
    return datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M%S")


def load_ticker_file(path: str | Path) -> list[OhlcvRow]:
    """Load one raw ticker file into a standardized OHLCV row list."""
    file_path = Path(path)
    with file_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {file_path}")

        missing_columns = REQUIRED_RAW_COLUMNS.difference(reader.fieldnames)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required columns in {file_path}: {missing}")

        rows: list[OhlcvRow] = []
        for raw_row in reader:
            row: OhlcvRow = {
                "timestamp": _build_timestamp(raw_row["<DATE>"], raw_row["<TIME>"]),
                "open": float(raw_row["<OPEN>"]),
                "high": float(raw_row["<HIGH>"]),
                "low": float(raw_row["<LOW>"]),
                "close": float(raw_row["<CLOSE>"]),
                "volume": int(raw_row["<VOL>"]),
            }
            rows.append(row)

    return sorted(rows, key=lambda row: row["timestamp"])


def load_ticker_frame(path: str | Path) -> "pd.DataFrame":
    """Load one raw ticker file into a pandas DataFrame if pandas is installed."""
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required for load_ticker_frame(); use load_ticker_file() otherwise."
        ) from exc

    return pd.DataFrame(load_ticker_file(path), columns=["timestamp", "open", "high", "low", "close", "volume"])

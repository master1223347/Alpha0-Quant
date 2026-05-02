"""Ticker-to-sector mapping helpers."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


def _canonical_ticker(value: Any) -> str:
    return str(value).strip().upper()


def load_ticker_sector_map(path: str | Path | None) -> dict[str, str]:
    """Load a ticker -> sector map from csv/parquet/json-like tabular files."""
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        LOGGER.warning("Sector map path %s not found; sector features will be zero-filled", file_path)
        return {}

    records: list[dict[str, Any]] = []
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            with file_path.open("r", newline="", encoding="utf-8") as handle:
                records = [dict(row) for row in csv.DictReader(handle)]
        else:
            import pandas as pd

            if suffix in {".parquet", ".pq"}:
                frame = pd.read_parquet(file_path)
            elif suffix in {".json", ".jsonl"}:
                frame = pd.read_json(file_path, lines=suffix == ".jsonl")
            else:
                frame = pd.read_csv(file_path)
            records = frame.to_dict(orient="records")
    except Exception as exc:
        LOGGER.warning("Failed to load sector map %s: %s", file_path, exc)
        return {}

    output: dict[str, str] = {}
    for record in records:
        ticker = record.get("ticker") or record.get("symbol") or record.get("Ticker") or record.get("Symbol")
        sector = record.get("sector") or record.get("gics_sector") or record.get("Sector")
        if ticker is None or sector is None:
            continue
        ticker_key = _canonical_ticker(ticker)
        sector_value = str(sector).strip()
        if ticker_key and sector_value:
            output[ticker_key] = sector_value
    return output


def infer_sector_by_etf_affinity(
    rows: list[dict[str, Any]],
    *,
    sector_etf_tickers: tuple[str, ...],
) -> dict[str, str]:
    """Cheap fallback sector assignment.

    If no static PIT sector map exists, this returns a deterministic "unknown"
    bucket per ticker. It intentionally does not infer from future returns; a
    true ETF-affinity assignment needs a separate rolling beta job.
    """
    del sector_etf_tickers
    tickers = sorted({_canonical_ticker(row.get("ticker", "")) for row in rows if row.get("ticker")})
    return {ticker: "UNKNOWN" for ticker in tickers}


def attach_sector_id_feature(
    rows: list[dict[str, Any]],
    *,
    sector_map_path: str | Path | None,
    infer_if_missing: bool,
    sector_etf_tickers: tuple[str, ...] = (),
) -> list[str]:
    """Attach stable numeric sector IDs in place.

    The integer code is deterministic across runs and safe for normal feature
    normalization. A missing map produces sector ID 0 instead of leaking from
    ex-post affinity calculations.
    """
    if not rows:
        return []

    sector_map = load_ticker_sector_map(sector_map_path)
    if not sector_map and infer_if_missing:
        sector_map = infer_sector_by_etf_affinity(rows, sector_etf_tickers=sector_etf_tickers)

    sectors = sorted(set(sector_map.values()))
    sector_to_id = {sector: index + 1 for index, sector in enumerate(sectors)}
    for row in rows:
        ticker = _canonical_ticker(row.get("ticker", ""))
        sector = sector_map.get(ticker)
        if sector is None:
            row["mctx_sector_id"] = 0.0
            row["mctx_sector_hash"] = 0.0
            continue
        row["mctx_sector_id"] = float(sector_to_id.get(sector, 0))
        digest = hashlib.blake2b(sector.encode("utf-8"), digest_size=4).hexdigest()
        row["mctx_sector_hash"] = float(int(digest, 16) % 1000) / 1000.0
    return ["mctx_sector_id", "mctx_sector_hash"]

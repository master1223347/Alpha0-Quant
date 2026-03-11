"""Raw ticker file discovery utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


ASSET_TYPES = {"stocks", "etfs"}


@dataclass(frozen=True, slots=True)
class TickerFile:
    ticker: str
    path: Path
    exchange_group: str
    exchange: str
    asset_type: str


def _extract_ticker(path: Path) -> str:
    stem = path.stem.strip()
    if not stem:
        raise ValueError(f"Cannot extract ticker from empty filename: {path}")
    # Strip market suffixes such as AAPL.US -> AAPL.
    return stem.split(".")[0].upper()


def _parse_exchange_group(group_name: str) -> tuple[str, str]:
    tokens = [token for token in re.split(r"[_\s-]+", group_name.lower().strip()) if token]
    if not tokens:
        return "unknown", "unknown"

    if tokens[-1] in ASSET_TYPES and len(tokens) >= 2:
        return "_".join(tokens[:-1]), tokens[-1]
    return "_".join(tokens), "unknown"


def _find_exchange_group(relative_parent: Path) -> str:
    for part in reversed(relative_parent.parts):
        lowered = part.lower()
        if "_stocks" in lowered or "_etfs" in lowered:
            return part
        if " stocks" in lowered or " etfs" in lowered:
            return part
    if relative_parent.parts:
        return relative_parent.parts[-1]
    return "unknown"


def discover_tickers(
    raw_root: str | Path = "data/raw",
    *,
    exchange: str | None = None,
    asset_type: str | None = None,
) -> list[TickerFile]:
    """Discover ticker files under the raw root and return structured metadata."""
    root = Path(raw_root)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Raw data path is not a directory: {root}")

    exchange_filter = exchange.lower() if exchange else None
    asset_type_filter = asset_type.lower() if asset_type else None

    discovered: list[TickerFile] = []
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue

        relative_path = file_path.relative_to(root)
        group_name = _find_exchange_group(relative_path.parent)
        exchange_name, asset_type_name = _parse_exchange_group(group_name)

        record = TickerFile(
            ticker=_extract_ticker(file_path),
            path=file_path,
            exchange_group=group_name,
            exchange=exchange_name,
            asset_type=asset_type_name,
        )

        if exchange_filter and record.exchange != exchange_filter:
            continue
        if asset_type_filter and record.asset_type != asset_type_filter:
            continue
        discovered.append(record)

    return discovered

"""Feature-building pipeline from raw OHLCV files."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config.default_config import ExperimentConfig, get_default_config, guess_raw_root
from src.data.align import align_ticker_rows
from src.data.cleaner import clean_ohlcv_rows
from src.data.corporate_actions import apply_corporate_actions
from src.data.discover import TickerFile, discover_tickers
from src.data.loader import OhlcvRow, load_ticker_file
from src.data.validator import validate_feature_rows, validate_ohlcv_rows
from src.features.base_features import build_base_features
from src.features.cross_sectional import apply_cross_sectional_features
from src.features.cross_sectional import cross_sectional_feature_columns
from src.features.factor_cointegration import apply_factor_cointegration_features
from src.features.market_features import build_market_features
from src.features.advanced_volatility import build_realized_volatility_features
from src.features.time_features import build_time_features
from src.features.volume_features import build_volume_features
from src.utils.logger import get_logger
from src.utils.paths import ensure_parent


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class FeatureBuildArtifacts:
    ticker_sequences: dict[str, list[list[dict[str, Any]]]]
    feature_columns: list[str]
    ticker_count: int
    sequence_count: int
    row_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _index_rows_by_timestamp(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {row["timestamp"]: row for row in rows}


def _merge_feature_rows(rows: list[OhlcvRow], config: ExperimentConfig) -> list[dict[str, Any]]:
    base = build_base_features(
        rows,
        momentum_lookback=config.features.momentum_lookback,
        volatility_window=config.features.volatility_window,
        relative_volume_window=config.features.relative_volume_window,
    )
    volume = build_volume_features(rows, window=config.features.volume_window)
    market = build_market_features(rows)
    time_rows = build_time_features(rows)
    if bool(getattr(config.features, "use_realized_volatility", True)):
        realized_vol = build_realized_volatility_features(
            rows,
            window=int(getattr(config.features, "realized_vol_window", 20)),
        )
    else:
        realized_vol = []

    if not base or not volume or not market or not time_rows:
        return []

    base_map = _index_rows_by_timestamp(base)
    volume_map = _index_rows_by_timestamp(volume)
    market_map = _index_rows_by_timestamp(market)
    time_map = _index_rows_by_timestamp(time_rows)
    realized_vol_map = _index_rows_by_timestamp(realized_vol) if realized_vol else {}
    ohlcv_map = {row["timestamp"]: row for row in rows}

    common_timestamps = (
        set(base_map.keys())
        .intersection(volume_map.keys())
        .intersection(market_map.keys())
        .intersection(time_map.keys())
        .intersection(ohlcv_map.keys())
    )
    if realized_vol_map:
        common_timestamps = common_timestamps.intersection(realized_vol_map.keys())
    merged: list[dict[str, Any]] = []
    for timestamp in sorted(common_timestamps):
        ohlcv = ohlcv_map[timestamp]
        merged.append(
            {
                "timestamp": timestamp,
                "close": float(ohlcv["close"]),
                **{k: v for k, v in base_map[timestamp].items() if k != "timestamp"},
                **{k: v for k, v in volume_map[timestamp].items() if k != "timestamp"},
                **{k: v for k, v in market_map[timestamp].items() if k != "timestamp"},
                **{k: v for k, v in time_map[timestamp].items() if k != "timestamp"},
                **({k: v for k, v in realized_vol_map[timestamp].items() if k != "timestamp"} if realized_vol_map else {}),
            }
        )
    return merged


def _feature_columns_from_sequence(sequence: list[dict[str, Any]]) -> list[str]:
    if not sequence:
        return []
    return sorted(key for key in sequence[0].keys() if key not in {"timestamp", "ticker", "close"})


def _use_cross_sectional_features(config: ExperimentConfig) -> bool:
    return bool(getattr(config.features, "use_cross_sectional", True))


def _load_membership_tickers(path: str | None) -> set[str] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        LOGGER.warning("Universe membership path %s not found; skipping membership filter", file_path)
        return None

    records: list[dict[str, Any]] = []
    try:
        if file_path.suffix.lower() == ".csv":
            import csv

            with file_path.open("r", newline="", encoding="utf-8") as handle:
                records = [row for row in csv.DictReader(handle)]
        else:
            import pandas as pd

            frame = pd.read_parquet(file_path) if file_path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(file_path)
            records = frame.to_dict(orient="records")
    except Exception as exc:
        LOGGER.warning("Failed to read universe membership file %s: %s", file_path, exc)
        return None

    tickers: set[str] = set()
    for record in records:
        value = record.get("ticker") or record.get("symbol")
        if value is None:
            continue
        active = record.get("active")
        if active is not None:
            active_value = str(active).strip().lower()
            if active_value in {"0", "false", "no"}:
                continue
        tickers.add(str(value).upper())
    return tickers or None


def build_features_for_ticker(ticker_file: TickerFile, config: ExperimentConfig) -> list[list[dict[str, Any]]]:
    """Build merged feature sequences for a single ticker file."""
    raw_rows = load_ticker_file(ticker_file.path)
    validate_ohlcv_rows(raw_rows, stage=f"{ticker_file.ticker}_loaded", raise_on_error=True)

    cleaned_rows = clean_ohlcv_rows(raw_rows)
    cleaned_rows = apply_corporate_actions(
        cleaned_rows,
        ticker=ticker_file.ticker,
        actions_path=getattr(config.data, "corporate_actions_path", None),
    )
    validate_ohlcv_rows(cleaned_rows, stage=f"{ticker_file.ticker}_cleaned", raise_on_error=True)

    aligned_sequences = align_ticker_rows(
        cleaned_rows,
        source_timezone=str(getattr(config.data, "source_timezone", "UTC")),
        market_timezone=str(getattr(config.data, "market_timezone", "America/New_York")),
    )
    min_sequence_length = (
        config.universe.min_sequence_length
        if config.universe.min_sequence_length is not None
        else config.data.min_sequence_length
    )
    ticker_sequences: list[list[dict[str, Any]]] = []
    for sequence in aligned_sequences:
        if len(sequence) < min_sequence_length:
            continue
        feature_sequence = _merge_feature_rows(sequence, config)
        if not feature_sequence:
            continue

        for row in feature_sequence:
            row["ticker"] = ticker_file.ticker

        columns = _feature_columns_from_sequence(feature_sequence)
        validate_feature_rows(
            feature_sequence,
            feature_columns=columns,
            stage=f"{ticker_file.ticker}_features",
            raise_on_error=True,
        )
        ticker_sequences.append(feature_sequence)

    return ticker_sequences


def _write_feature_rows(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    output_path = ensure_parent(path)
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        frame.to_parquet(output_path, index=False)
    except ModuleNotFoundError:
        fallback = output_path.with_suffix(".json")
        with fallback.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, default=str)
        return fallback
    except Exception:
        fallback = output_path.with_suffix(".json")
        with fallback.open("w", encoding="utf-8") as handle:
            json.dump(rows, handle, default=str)
        return fallback
    return output_path


def _write_ticker_list(metadata_dir: str | Path, tickers: list[str]) -> Path:
    output_path = ensure_parent(Path(metadata_dir) / "ticker_list.json")
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(sorted(tickers), handle, indent=2)
    return output_path


def build_feature_store(
    config: ExperimentConfig | None = None,
    *,
    exchange: str | None = None,
    asset_type: str | None = None,
) -> FeatureBuildArtifacts:
    """Discover tickers and build feature sequences for each ticker."""
    config = config or get_default_config()
    raw_root = guess_raw_root(config)
    resolved_exchange = exchange if exchange is not None else config.universe.exchange
    resolved_asset_type = asset_type if asset_type is not None else config.universe.asset_type
    ticker_files = discover_tickers(raw_root=raw_root, exchange=resolved_exchange, asset_type=resolved_asset_type)
    membership_tickers = _load_membership_tickers(getattr(config.universe, "membership_path", None))
    if membership_tickers:
        ticker_files = [ticker_file for ticker_file in ticker_files if ticker_file.ticker.upper() in membership_tickers]

    if config.universe.tickers:
        allowed = {ticker.upper() for ticker in config.universe.tickers}
        ticker_files = [ticker_file for ticker_file in ticker_files if ticker_file.ticker.upper() in allowed]

    max_tickers = config.universe.max_tickers if config.universe.max_tickers is not None else config.data.max_tickers
    if max_tickers is not None:
        ticker_files = ticker_files[: int(max_tickers)]

    ticker_sequences: dict[str, list[list[dict[str, Any]]]] = {}
    base_feature_columns: list[str] = []
    feature_columns: list[str] = []
    total_sequences = 0
    total_rows = 0

    for ticker_file in ticker_files:
        try:
            sequences = build_features_for_ticker(ticker_file, config)
        except Exception as exc:
            LOGGER.warning("Skipping ticker %s due to feature error: %s", ticker_file.ticker, exc)
            continue
        if not sequences:
            continue
        ticker_sequences[ticker_file.ticker] = sequences
        total_sequences += len(sequences)
        total_rows += sum(len(sequence) for sequence in sequences)
        if not base_feature_columns:
            base_feature_columns = _feature_columns_from_sequence(sequences[0])

    flattened_rows = [row for sequences in ticker_sequences.values() for sequence in sequences for row in sequence]
    if flattened_rows and (
        bool(getattr(config.features, "use_factor_features", True))
        or bool(getattr(config.features, "use_cointegration_features", True))
    ):
        apply_factor_cointegration_features(
            flattened_rows,
            use_factor_features=bool(getattr(config.features, "use_factor_features", True)),
            use_cointegration_features=bool(getattr(config.features, "use_cointegration_features", True)),
            factor_window=int(getattr(config.features, "factor_window", 78)),
            cointegration_window=int(getattr(config.features, "cointegration_window", 78)),
            min_samples=int(getattr(config.features, "cointegration_min_samples", 40)),
            half_life_clip=float(getattr(config.features, "cointegration_half_life_clip", 200.0)),
        )
        base_feature_columns = _feature_columns_from_sequence(flattened_rows[:1])

    use_cross_sectional = _use_cross_sectional_features(config)
    if flattened_rows and base_feature_columns and use_cross_sectional:
        apply_cross_sectional_features(flattened_rows, feature_columns=base_feature_columns)
        feature_columns = base_feature_columns + cross_sectional_feature_columns(base_feature_columns)
        for ticker, sequences in ticker_sequences.items():
            for index, sequence in enumerate(sequences):
                validate_feature_rows(
                    sequence,
                    feature_columns=feature_columns,
                    stage=f"{ticker}_cross_sectional_{index}",
                    raise_on_error=True,
                )
    elif flattened_rows:
        feature_columns = base_feature_columns or _feature_columns_from_sequence(flattened_rows[:1])

    output_path = _write_feature_rows(config.data.features_path, flattened_rows)
    ticker_list_path = _write_ticker_list(config.data.metadata_dir, list(ticker_sequences.keys()))
    LOGGER.info("Feature rows written to %s", output_path)
    LOGGER.info("Ticker list written to %s", ticker_list_path)

    return FeatureBuildArtifacts(
        ticker_sequences=ticker_sequences,
        feature_columns=feature_columns,
        ticker_count=len(ticker_sequences),
        sequence_count=total_sequences,
        row_count=total_rows,
    )

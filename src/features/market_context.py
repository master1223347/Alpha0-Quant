"""Market-context features: benchmark ETF state, breadth, correlation, and gap regime.

Phase-1 of the design report. All features are timestamp-global (broadcast to
every ticker at the same timestamp), so they are joined to the flattened panel
after per-ticker base features and cross-sectional features have been built.

Calendar features and static sector IDs live in sibling modules. HMM regime
inference remains in evaluation/regime.py; this module supplies its state
features without peeking forward.

The module is gated by config.market_context.enabled; when disabled it is a
no-op and returns an empty list of new column names.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from src.data.align import align_ticker_rows
from src.data.cleaner import clean_ohlcv_rows
from src.data.loader import OhlcvRow, load_ticker_file
from src.utils.logger import get_logger


LOGGER = get_logger(__name__)


# ----- benchmark ETF features ------------------------------------------------


def _find_benchmark_file(raw_root: Path, ticker: str) -> Path | None:
    """Locate a benchmark ticker file under the raw data tree.

    Searches for `<ticker>.us.txt` (case-insensitive) anywhere under raw_root.
    """
    target_lower = f"{ticker.lower()}.us"
    for candidate in raw_root.rglob("*"):
        if not candidate.is_file():
            continue
        stem_lower = candidate.stem.lower()
        if stem_lower == target_lower or stem_lower.split(".")[0] == ticker.lower():
            return candidate
    return None


def _load_benchmark_sequences(
    raw_root: Path,
    ticker: str,
    *,
    source_timezone: str,
    market_timezone: str,
) -> list[list[OhlcvRow]]:
    path = _find_benchmark_file(raw_root, ticker)
    if path is None:
        LOGGER.warning("Benchmark file for %s not found under %s", ticker, raw_root)
        return []
    rows = load_ticker_file(path)
    cleaned = clean_ohlcv_rows(rows)
    return align_ticker_rows(
        cleaned,
        source_timezone=source_timezone,
        market_timezone=market_timezone,
    )


def _benchmark_features_for_sequence(
    sequence: list[OhlcvRow],
    *,
    return_lookbacks: tuple[int, ...],
    rv_window: int,
    rv_baseline_window: int,
) -> dict[datetime, dict[str, float]]:
    """Compute per-bar timestamp -> feature dict for a single session sequence."""
    out: dict[datetime, dict[str, float]] = {}
    if not sequence:
        return out

    # log returns over the sequence (within session only; first bar = 0).
    log_returns: list[float] = [0.0]
    closes = [row["close"] for row in sequence]
    for i in range(1, len(sequence)):
        prev = closes[i - 1]
        cur = closes[i]
        if prev > 0 and cur > 0:
            log_returns.append(math.log(cur / prev))
        else:
            log_returns.append(0.0)

    # cumulative session VWAP for vwap deviation (uses only past bars).
    cum_pv = 0.0
    cum_v = 0.0

    # session high/low running for intraday range position.
    session_high = -math.inf
    session_low = math.inf

    # rolling realized vol: trailing rv_window bar log-return std, plus a
    # longer baseline (median per-bar abs return is a cheap stand-in for
    # the ratio "rv / median daily rv" called out in the report).
    abs_returns: list[float] = []

    for i, row in enumerate(sequence):
        ts = row["timestamp"]
        feats: dict[str, float] = {}

        # multi-horizon returns from past bars only.
        for lookback in return_lookbacks:
            if i >= lookback:
                feats[f"ret_{lookback}"] = sum(log_returns[i - lookback + 1 : i + 1])
            else:
                feats[f"ret_{lookback}"] = 0.0

        # vwap deviation from running session vwap (uses bars up to and
        # including current; deviation is from current close to vwap).
        typical = (row["high"] + row["low"] + row["close"]) / 3.0
        cum_pv += typical * row["volume"]
        cum_v += row["volume"]
        vwap = cum_pv / cum_v if cum_v > 0 else row["close"]
        feats["vwap_dev"] = (row["close"] / vwap) - 1.0 if vwap > 0 else 0.0

        # intraday range position [0,1].
        session_high = max(session_high, row["high"])
        session_low = min(session_low, row["low"])
        span = session_high - session_low
        feats["intraday_range_pos"] = (
            (row["close"] - session_low) / span if span > 1e-12 else 0.5
        )

        # rolling realized vol over trailing rv_window bars.
        abs_r = abs(log_returns[i])
        abs_returns.append(abs_r)
        rv_start = max(0, i - rv_window + 1)
        rv_slice = log_returns[rv_start : i + 1]
        rv = _std(rv_slice)
        baseline_start = max(0, i - rv_baseline_window + 1)
        baseline_slice = abs_returns[baseline_start : i + 1]
        baseline = (
            sum(baseline_slice) / len(baseline_slice) if baseline_slice else 0.0
        )
        feats["rv_ratio"] = rv / baseline if baseline > 1e-12 else 0.0

        out[ts] = feats

    return out


def compute_benchmark_state_features(
    raw_root: Path,
    *,
    benchmarks: Iterable[str],
    source_timezone: str,
    market_timezone: str,
    return_lookbacks: tuple[int, ...] = (1, 3, 12),
    rv_window: int = 12,
    rv_baseline_window: int = 78,
) -> tuple[dict[datetime, dict[str, float]], list[str]]:
    """Return (timestamp -> {col: value}) and the list of new column names.

    Column naming: `mctx_{ticker_lower}_{feat}` (e.g. `mctx_spy_ret_3`).
    """
    feature_cols: list[str] = []
    by_ts: dict[datetime, dict[str, float]] = defaultdict(dict)

    feat_names = (
        [f"ret_{lb}" for lb in return_lookbacks]
        + ["vwap_dev", "intraday_range_pos", "rv_ratio"]
    )

    for benchmark in benchmarks:
        prefix = f"mctx_{benchmark.lower()}_"
        cols_for_bench = [prefix + name for name in feat_names]
        feature_cols.extend(cols_for_bench)
        sequences = _load_benchmark_sequences(
            raw_root,
            benchmark,
            source_timezone=source_timezone,
            market_timezone=market_timezone,
        )
        for sequence in sequences:
            per_ts = _benchmark_features_for_sequence(
                sequence,
                return_lookbacks=return_lookbacks,
                rv_window=rv_window,
                rv_baseline_window=rv_baseline_window,
            )
            for ts, feats in per_ts.items():
                target = by_ts[ts]
                for name in feat_names:
                    target[prefix + name] = float(feats.get(name, 0.0))

    return by_ts, feature_cols


# ----- breadth features ------------------------------------------------------


def compute_breadth_features(
    rows: list[dict[str, Any]],
    *,
    return_column: str = "log_return",
    ema_windows: tuple[int, ...] = (3, 12),
) -> tuple[dict[datetime, dict[str, float]], list[str]]:
    """Per-timestamp cross-sectional breadth from already-built feature rows.

    Uses only same-bar values, so safe to run after per-ticker features but
    before labels are attached.
    """
    columns = [
        "mctx_breadth_ad_share",
        "mctx_breadth_up_volume_share",
        "mctx_breadth_signed_volume",
        "mctx_breadth_median_ret",
        "mctx_breadth_dispersion_ret",
        "mctx_breadth_cum_ad_open",
        "mctx_breadth_new_high_share_20d",
        "mctx_breadth_new_low_share_20d",
    ]
    for window in ema_windows:
        columns.extend(
            [
                f"mctx_breadth_ad_share_ema_{int(window)}",
                f"mctx_breadth_up_volume_share_ema_{int(window)}",
            ]
        )

    by_ts_rows: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ts = row.get("timestamp")
        if ts is None:
            continue
        by_ts_rows[ts].append(row)

    new_high_flags, new_low_flags = _compute_close_breakout_flags(rows, window_days=20)
    out: dict[datetime, dict[str, float]] = {}
    running_cum_ad: dict[Any, float] = defaultdict(float)
    ema_state: dict[str, float] = {}
    for ts in sorted(by_ts_rows.keys()):
        group = by_ts_rows[ts]
        rets: list[float] = []
        signed_volume_up = 0.0
        signed_volume_total = 0.0
        signed_volume_net = 0.0
        advancers = 0
        decliners = 0
        high_count = 0
        low_count = 0
        for r in group:
            ret = r.get(return_column)
            if not isinstance(ret, (int, float)) or math.isnan(float(ret)):
                continue
            ret_f = float(ret)
            rets.append(ret_f)
            ticker = str(r.get("ticker", ""))
            high_count += int(new_high_flags.get((ts, ticker), 0))
            low_count += int(new_low_flags.get((ts, ticker), 0))
            # Use relative_volume as a volume proxy if available; else 1.0.
            vol_proxy = r.get("relative_volume", 1.0)
            try:
                vol_proxy_f = float(vol_proxy)
            except (TypeError, ValueError):
                vol_proxy_f = 1.0
            if not math.isfinite(vol_proxy_f) or vol_proxy_f < 0:
                vol_proxy_f = 0.0
            signed_volume_total += vol_proxy_f
            if ret_f > 0:
                advancers += 1
                signed_volume_up += vol_proxy_f
                signed_volume_net += vol_proxy_f
            elif ret_f < 0:
                decliners += 1
                signed_volume_net -= vol_proxy_f

        n = len(rets)
        if n == 0:
            out[ts] = {c: 0.0 for c in columns}
            continue
        ad_total = advancers + decliners
        ad_share = (advancers - decliners) / ad_total if ad_total > 0 else 0.0
        up_vol_share = (
            signed_volume_up / signed_volume_total if signed_volume_total > 0 else 0.0
        )
        signed_volume_breadth = (
            signed_volume_net / signed_volume_total if signed_volume_total > 0 else 0.0
        )
        median = _median(rets)
        dispersion = _std(rets)
        session_date = ts.date() if isinstance(ts, datetime) else None
        if session_date is not None:
            running_cum_ad[session_date] += advancers - decliners
        row_features = {
            "mctx_breadth_ad_share": ad_share,
            "mctx_breadth_up_volume_share": up_vol_share,
            "mctx_breadth_signed_volume": signed_volume_breadth,
            "mctx_breadth_median_ret": median,
            "mctx_breadth_dispersion_ret": dispersion,
            "mctx_breadth_cum_ad_open": running_cum_ad.get(session_date, 0.0) if session_date is not None else 0.0,
            "mctx_breadth_new_high_share_20d": high_count / n if n > 0 else 0.0,
            "mctx_breadth_new_low_share_20d": low_count / n if n > 0 else 0.0,
        }
        for window in ema_windows:
            alpha = 2.0 / (float(window) + 1.0)
            ad_key = f"ad_{int(window)}"
            uv_key = f"uv_{int(window)}"
            ema_state[ad_key] = ad_share if ad_key not in ema_state else alpha * ad_share + (1.0 - alpha) * ema_state[ad_key]
            ema_state[uv_key] = (
                up_vol_share
                if uv_key not in ema_state
                else alpha * up_vol_share + (1.0 - alpha) * ema_state[uv_key]
            )
            row_features[f"mctx_breadth_ad_share_ema_{int(window)}"] = ema_state[ad_key]
            row_features[f"mctx_breadth_up_volume_share_ema_{int(window)}"] = ema_state[uv_key]
        out[ts] = row_features

    return out, columns


def compute_realized_correlation_features(
    rows: list[dict[str, Any]],
    *,
    windows_bars: tuple[int, ...] = (78, 390),
    liquid_subset_size: int = 300,
    return_column: str = "log_return",
) -> tuple[dict[datetime, dict[str, float]], list[str]]:
    """Rolling scalar correlation/common-factor summaries.

    This intentionally avoids full-N storage. It keeps the first
    `liquid_subset_size` tickers by observed row count as a deterministic
    liquidity proxy when ADV is unavailable.
    """
    if not rows:
        return {}, []

    cols: list[str] = []
    for window in windows_bars:
        suffix = f"{int(window)}b"
        cols.extend(
            [
                f"mctx_avg_corr_{suffix}",
                f"mctx_corr_dispersion_{suffix}",
                f"mctx_market_mode_share_{suffix}",
                f"mctx_resid_avg_corr_{suffix}",
                f"mctx_common_factor_share_{suffix}",
            ]
        )

    counts: dict[str, int] = defaultdict(int)
    by_ts: dict[datetime, dict[str, float]] = defaultdict(dict)
    for row in rows:
        ticker = str(row.get("ticker", ""))
        if not ticker:
            continue
        counts[ticker] += 1
        ts = row.get("timestamp")
        if ts is None:
            continue
        try:
            by_ts[ts][ticker] = float(row.get(return_column, 0.0))
        except (TypeError, ValueError):
            by_ts[ts][ticker] = 0.0

    liquid_tickers = [
        ticker
        for ticker, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: max(2, int(liquid_subset_size))]
    ]
    ordered_ts = sorted(by_ts.keys())
    returns_by_ticker = {ticker: [] for ticker in liquid_tickers}
    out: dict[datetime, dict[str, float]] = {}

    for ts in ordered_ts:
        row_returns = by_ts.get(ts, {})
        for ticker in liquid_tickers:
            returns_by_ticker[ticker].append(float(row_returns.get(ticker, 0.0)))

        feats: dict[str, float] = {}
        for window in windows_bars:
            window = int(window)
            suffix = f"{window}b"
            if len(next(iter(returns_by_ticker.values()), [])) < max(3, window):
                for col in cols:
                    if col.endswith(suffix):
                        feats[col] = 0.0
                continue

            series = [returns_by_ticker[ticker][-window:] for ticker in liquid_tickers]
            series = [values for values in series if _std(values) > 1e-12]
            if len(series) < 2:
                for col in cols:
                    if col.endswith(suffix):
                        feats[col] = 0.0
                continue
            avg_corr, corr_dispersion = _pairwise_corr_summary(series)
            market_mode_share = _market_mode_share(series)
            resid_series = _cross_sectional_residualize(series)
            resid_avg_corr, _ = _pairwise_corr_summary(resid_series)
            common_factor_share = _common_factor_share(series)
            feats[f"mctx_avg_corr_{suffix}"] = avg_corr
            feats[f"mctx_corr_dispersion_{suffix}"] = corr_dispersion
            feats[f"mctx_market_mode_share_{suffix}"] = market_mode_share
            feats[f"mctx_resid_avg_corr_{suffix}"] = resid_avg_corr
            feats[f"mctx_common_factor_share_{suffix}"] = common_factor_share
        out[ts] = feats

    return out, cols


# ----- gap regime features ---------------------------------------------------


def compute_gap_regime_features(
    raw_root: Path,
    *,
    benchmarks: Iterable[str],
    source_timezone: str,
    market_timezone: str,
    z_window_days: int = 60,
) -> tuple[dict[Any, dict[str, float]], list[str]]:
    """Daily overnight gap features per benchmark, keyed by session date.

    Returns mapping date -> {col: val}. Caller broadcasts to every intraday
    bar that falls on the same session date.
    """
    cols = [f"mctx_{b.lower()}_gap_z" for b in benchmarks]

    by_date: dict[Any, dict[str, float]] = defaultdict(dict)

    for benchmark in benchmarks:
        sequences = _load_benchmark_sequences(
            raw_root,
            benchmark,
            source_timezone=source_timezone,
            market_timezone=market_timezone,
        )
        if not sequences:
            continue
        # Use first/last bars of each session to form an overnight gap series.
        sessions: list[tuple[Any, float, float]] = []  # (date, first_open, last_close)
        for seq in sequences:
            if not seq:
                continue
            sessions.append(
                (seq[0]["timestamp"].date(), seq[0]["open"], seq[-1]["close"])
            )
        sessions.sort(key=lambda x: x[0])

        gaps: list[tuple[Any, float]] = []
        for i in range(1, len(sessions)):
            prev_close = sessions[i - 1][2]
            cur_open = sessions[i][1]
            if prev_close > 0 and cur_open > 0:
                gaps.append((sessions[i][0], math.log(cur_open / prev_close)))

        col_name = f"mctx_{benchmark.lower()}_gap_z"
        for i, (date, gap) in enumerate(gaps):
            window_start = max(0, i - z_window_days + 1)
            window = [g for _, g in gaps[window_start : i + 1]]
            mean = sum(window) / len(window) if window else 0.0
            std = _std(window)
            z = (gap - mean) / std if std > 1e-12 else 0.0
            by_date[date][col_name] = z

    # Ensure every date present has the full column set (zero-fill missing).
    for date, feats in by_date.items():
        for c in cols:
            feats.setdefault(c, 0.0)

    return by_date, cols


# ----- attach to flattened panel --------------------------------------------


def attach_market_context_features(
    rows: list[dict[str, Any]],
    *,
    raw_root: Path,
    benchmarks: Iterable[str],
    enable_breadth: bool,
    enable_gap_regime: bool,
    source_timezone: str,
    market_timezone: str,
    realized_corr_enabled: bool = False,
    return_lookbacks: tuple[int, ...] = (1, 3, 12),
    rv_window: int = 12,
    rv_baseline_window: int = 78,
    gap_z_window_days: int = 60,
    breadth_ema_windows: tuple[int, ...] = (3, 12),
    corr_liquid_subset_size: int = 300,
    corr_windows_bars: tuple[int, ...] = (78, 390),
) -> list[str]:
    """Attach broadcast market-context features in place. Returns new columns."""
    if not rows:
        return []

    new_cols: list[str] = []
    benchmarks_t = tuple(b.upper() for b in benchmarks)

    bench_by_ts: dict[datetime, dict[str, float]] = {}
    bench_cols: list[str] = []
    if benchmarks_t:
        bench_by_ts, bench_cols = compute_benchmark_state_features(
            raw_root,
            benchmarks=benchmarks_t,
            source_timezone=source_timezone,
            market_timezone=market_timezone,
            return_lookbacks=return_lookbacks,
            rv_window=rv_window,
            rv_baseline_window=rv_baseline_window,
        )
    new_cols.extend(bench_cols)

    breadth_by_ts: dict[datetime, dict[str, float]] = {}
    breadth_cols: list[str] = []
    if enable_breadth:
        breadth_by_ts, breadth_cols = compute_breadth_features(rows, ema_windows=breadth_ema_windows)
    new_cols.extend(breadth_cols)

    corr_by_ts: dict[datetime, dict[str, float]] = {}
    corr_cols: list[str] = []
    if realized_corr_enabled:
        corr_by_ts, corr_cols = compute_realized_correlation_features(
            rows,
            windows_bars=corr_windows_bars,
            liquid_subset_size=corr_liquid_subset_size,
        )
    new_cols.extend(corr_cols)

    gap_by_date: dict[Any, dict[str, float]] = {}
    gap_cols: list[str] = []
    if enable_gap_regime and benchmarks_t:
        gap_by_date, gap_cols = compute_gap_regime_features(
            raw_root,
            benchmarks=benchmarks_t,
            source_timezone=source_timezone,
            market_timezone=market_timezone,
            z_window_days=gap_z_window_days,
        )
    new_cols.extend(gap_cols)

    if not new_cols:
        return []

    for row in rows:
        ts = row.get("timestamp")
        bench_feats = bench_by_ts.get(ts, {}) if bench_by_ts else {}
        for col in bench_cols:
            row[col] = float(bench_feats.get(col, 0.0))
        breadth_feats = breadth_by_ts.get(ts, {}) if breadth_by_ts else {}
        for col in breadth_cols:
            row[col] = float(breadth_feats.get(col, 0.0))
        corr_feats = corr_by_ts.get(ts, {}) if corr_by_ts else {}
        for col in corr_cols:
            row[col] = float(corr_feats.get(col, 0.0))
        if gap_cols:
            session_date = ts.date() if isinstance(ts, datetime) else None
            gap_feats = gap_by_date.get(session_date, {}) if session_date else {}
            for col in gap_cols:
                row[col] = float(gap_feats.get(col, 0.0))

    return new_cols


# ----- math helpers ---------------------------------------------------------


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _compute_close_breakout_flags(
    rows: list[dict[str, Any]],
    *,
    window_days: int,
) -> tuple[dict[tuple[Any, str], int], dict[tuple[Any, str], int]]:
    by_ticker_date: dict[str, dict[Any, list[tuple[Any, float]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        ts = row.get("timestamp")
        ticker = str(row.get("ticker", ""))
        if ts is None or not ticker:
            continue
        session_date = ts.date() if isinstance(ts, datetime) else None
        if session_date is None:
            continue
        try:
            close = float(row.get("close", 0.0))
        except (TypeError, ValueError):
            close = 0.0
        by_ticker_date[ticker][session_date].append((ts, close))

    high_flags: dict[tuple[Any, str], int] = {}
    low_flags: dict[tuple[Any, str], int] = {}
    for ticker, by_date in by_ticker_date.items():
        past_highs: list[float] = []
        past_lows: list[float] = []
        for session_date in sorted(by_date.keys()):
            intraday = sorted(by_date[session_date], key=lambda item: item[0])
            trailing_high = max(past_highs[-window_days:]) if past_highs else None
            trailing_low = min(past_lows[-window_days:]) if past_lows else None
            day_high = max(close for _, close in intraday)
            day_low = min(close for _, close in intraday)
            for ts, close in intraday:
                high_flags[(ts, ticker)] = int(trailing_high is not None and close >= trailing_high)
                low_flags[(ts, ticker)] = int(trailing_low is not None and close <= trailing_low)
            past_highs.append(day_high)
            past_lows.append(day_low)
    return high_flags, low_flags


def _standardize(values: list[float]) -> list[float]:
    std = _std(values)
    if std <= 1e-12:
        return [0.0 for _ in values]
    mean = sum(values) / len(values)
    return [(value - mean) / std for value in values]


def _corr(left: list[float], right: list[float]) -> float:
    left_z = _standardize(left)
    right_z = _standardize(right)
    if not left_z or len(left_z) != len(right_z):
        return 0.0
    return sum(l * r for l, r in zip(left_z, right_z)) / len(left_z)


def _pairwise_corr_summary(series: list[list[float]]) -> tuple[float, float]:
    values: list[float] = []
    for i in range(len(series)):
        for j in range(i + 1, len(series)):
            values.append(_corr(series[i], series[j]))
    if not values:
        return 0.0, 0.0
    return sum(values) / len(values), _std(values)


def _market_mode_share(series: list[list[float]]) -> float:
    if not series:
        return 0.0
    standardized = [_standardize(values) for values in series]
    if not standardized or not standardized[0]:
        return 0.0
    market_series = [sum(values[t] for values in standardized) / len(standardized) for t in range(len(standardized[0]))]
    total_variance = sum(_std(values) ** 2 for values in standardized)
    market_variance = _std(market_series) ** 2
    return market_variance / total_variance if total_variance > 1e-12 else 0.0


def _cross_sectional_residualize(series: list[list[float]]) -> list[list[float]]:
    if not series:
        return []
    length = len(series[0])
    market_series = [sum(values[t] for values in series) / len(series) for t in range(length)]
    return [[values[t] - market_series[t] for t in range(length)] for values in series]


def _common_factor_share(series: list[list[float]]) -> float:
    if not series:
        return 0.0
    residualized = _cross_sectional_residualize(series)
    total_var = sum(_std(values) ** 2 for values in series)
    resid_var = sum(_std(values) ** 2 for values in residualized)
    if total_var <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, 1.0 - (resid_var / total_var)))

"""Model waterfall: a magnitude gate plus a three-model direction ensemble; trade only when every stage is confident.

Cutoffs are chosen on the validation period and applied once to the test period.
Trades are market-hedged (stock minus equal-weight basket), entered one bar after the signal,
held `--hold` bars, closed the same session, and charged costs on both legs.

Usage: python scripts/model_waterfall.py --raw <dir> --bars-per-day 7 --hold 3
"""

from __future__ import annotations

import argparse
import glob
import itertools
import math

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

QUANTILES = (0.0, 0.5, 0.7, 0.8, 0.9)
GRAPH_LOOKBACKS = (1, 3, 6)
REGIME_COLS = ("mkt_vol", "mkt_abs_6", "dispersion")


def graph_neighbor_returns(excess: pd.DataFrame, corr_bars: int, top_k: int = 5) -> dict[str, pd.DataFrame]:
    """Correlation-weighted neighbor excess returns; the graph for day d uses only bars before day d."""
    days = pd.Series(excess.index.date, index=excess.index)
    values = excess.to_numpy()
    out = {lookback: np.full_like(values, np.nan) for lookback in GRAPH_LOOKBACKS}
    rolled = {lookback: excess.rolling(lookback).sum().to_numpy() for lookback in GRAPH_LOOKBACKS}
    for day in days.unique():
        rows = np.flatnonzero((days == day).to_numpy())
        history = values[max(0, rows[0] - corr_bars) : rows[0]]
        history = history[~np.isnan(history).any(axis=1)]
        if len(history) < corr_bars // 2:
            continue
        corr = np.nan_to_num(np.corrcoef(history.T))
        np.fill_diagonal(corr, 0.0)
        corr[corr < 0] = 0.0
        cutoff = -np.sort(-corr, axis=1)[:, top_k - 1 : top_k]
        corr[corr < cutoff] = 0.0
        weights = corr / np.maximum(corr.sum(axis=1, keepdims=True), 1e-12)
        for lookback in GRAPH_LOOKBACKS:
            out[lookback][rows] = rolled[lookback][rows] @ weights.T
    return {
        f"graph_nbr_{lookback}": pd.DataFrame(out[lookback], index=excess.index, columns=excess.columns)
        for lookback in GRAPH_LOOKBACKS
    }


def sec_event_features(index: pd.DatetimeIndex, tickers: list[str], filings_path: str) -> dict[str, pd.DataFrame]:
    """Hours since the last 8-K and recent-earnings (item 2.02) flags, using only filings accepted before the bar."""
    filings = pd.read_csv(filings_path, parse_dates=["accepted_utc"])
    hours_since = pd.DataFrame(np.nan, index=index, columns=tickers)
    earnings_3d = pd.DataFrame(0.0, index=index, columns=tickers)
    bars = pd.DataFrame({"timestamp": index})
    for ticker in tickers:
        own = filings[filings["ticker"] == ticker].sort_values("accepted_utc")
        if own.empty:
            continue
        joined = pd.merge_asof(bars, own, left_on="timestamp", right_on="accepted_utc", allow_exact_matches=False)
        hours_since[ticker] = ((joined["timestamp"] - joined["accepted_utc"]).dt.total_seconds() / 3600).to_numpy()
        earn = own[own["items"].fillna("").str.contains("2.02", regex=False)]
        if not earn.empty:
            joined_e = pd.merge_asof(bars, earn, left_on="timestamp", right_on="accepted_utc", allow_exact_matches=False)
            age = (joined_e["timestamp"] - joined_e["accepted_utc"]).dt.total_seconds() / 3600
            earnings_3d[ticker] = (age <= 72).astype(float).to_numpy()
    return {"sec_log_hours_since_8k": np.log1p(hours_since.fillna(24 * 365)), "sec_earnings_3d": earnings_3d}


def load_panel(raw_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, volumes = {}, {}
    for path in sorted(glob.glob(f"{raw_dir}/*.txt")):
        frame = pd.read_csv(path)
        stamps = pd.to_datetime(
            frame["<DATE>"].astype(str) + frame["<TIME>"].astype(str).str.zfill(6), format="%Y%m%d%H%M%S"
        )
        ticker = str(frame["<TICKER>"].iloc[0]).removesuffix(".US")
        closes[ticker] = pd.Series(frame["<CLOSE>"].to_numpy(), index=stamps)
        volumes[ticker] = pd.Series(frame["<VOL>"].to_numpy(dtype=float), index=stamps)
    close = pd.DataFrame(closes).sort_index().ffill(limit=2)
    volume = pd.DataFrame(volumes).reindex(close.index)
    return close, volume


def build_rows(
    close: pd.DataFrame, volume: pd.DataFrame, hold: int, *, bars_per_day: int, filings_path: str | None
) -> pd.DataFrame:
    log_close = np.log(close)
    ret1 = log_close.diff()
    market = ret1.mean(axis=1)
    excess = ret1.sub(market, axis=0)
    features: dict[str, pd.DataFrame] = {}
    for lookback in (1, 3, 6, 12, 24):
        raw = excess.rolling(lookback).sum()
        features[f"xret_{lookback}"] = raw
        features[f"xrank_{lookback}"] = raw.rank(axis=1, pct=True)
    features["vol_12"] = ret1.rolling(12).std()
    features["vol_ratio"] = ret1.rolling(6).std() / ret1.rolling(48).std()
    features["rel_volume"] = np.log1p(volume) - np.log1p(volume).rolling(48).mean()
    features.update(graph_neighbor_returns(excess, corr_bars=bars_per_day * 20))
    for lookback in GRAPH_LOOKBACKS:
        features[f"graph_gap_{lookback}"] = features[f"graph_nbr_{lookback}"] - features[f"xret_{lookback}"]
    if filings_path:
        features.update(sec_event_features(close.index, list(close.columns), filings_path))
    regime_inputs = {
        "mkt_vol": market.rolling(12).std(),
        "mkt_abs_6": market.rolling(6).sum().abs(),
        "dispersion": ret1.std(axis=1),
    }
    for name, series in (("mkt_1", market), ("mkt_6", market.rolling(6).sum()), *regime_inputs.items()):
        features[name] = pd.DataFrame({ticker: series for ticker in close.columns})

    day = pd.Series(close.index.date, index=close.index)
    bar_of_day = day.groupby(day).cumcount()
    features["bar_of_day"] = pd.DataFrame({ticker: bar_of_day for ticker in close.columns})

    # Enter at close of t+1 (1-bar lag), exit at close of t+1+hold, same session only.
    entry = log_close.shift(-1)
    exit_ = log_close.shift(-1 - hold)
    stock_fwd = exit_ - entry
    target = stock_fwd.sub(stock_fwd.mean(axis=1), axis=0)
    same_session = day.shift(-1 - hold) == day
    target.loc[~same_session.to_numpy()] = np.nan

    stacked = pd.concat({name: frame.stack(future_stack=True) for name, frame in features.items()}, axis=1)
    stacked["target"] = target.stack(future_stack=True)
    stacked.index.names = ["timestamp", "ticker"]
    return stacked.replace([np.inf, -np.inf], np.nan).dropna().reset_index()


def chronological_split(rows: pd.DataFrame, hold: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stamps = np.array(sorted(rows["timestamp"].unique()))
    n = len(stamps)
    gap = hold + 1
    train_end, val_end = int(n * 0.6), int(n * 0.8)
    train = rows[rows["timestamp"] < stamps[train_end - gap]]
    val = rows[(rows["timestamp"] >= stamps[train_end]) & (rows["timestamp"] < stamps[val_end - gap])]
    test = rows[rows["timestamp"] >= stamps[val_end]]
    return train, val, test


def fit_models(train: pd.DataFrame, feature_cols: list[str], seed: int = 0) -> dict:
    x, y = train[feature_cols].to_numpy(), train["target"].to_numpy()
    graph_cols = [feature_cols.index(c) for c in feature_cols if c.startswith("graph_")]
    regime = make_pipeline(StandardScaler(), GaussianMixture(n_components=3, random_state=seed))
    regime.fit(train.drop_duplicates("timestamp")[list(REGIME_COLS)].to_numpy())
    direction = (y > 0).astype(int)
    magnitude = (np.abs(y) > np.median(np.abs(y))).astype(int)
    gate = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed)
    gate.fit(x, magnitude)
    voters = {
        "logit": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500)),
        "hgb": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200, min_samples_leaf=200, max_features=0.5, n_jobs=-1, random_state=seed
        ),
    }
    for model in voters.values():
        model.fit(x, direction)
    graph_voter = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    graph_voter.fit(x[:, graph_cols], direction)
    return {"gate": gate, "voters": voters, "graph": (graph_voter, graph_cols), "regime": regime}


def score(models: dict, frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    x = frame[feature_cols].to_numpy()
    out = frame[["timestamp", "ticker", "target"]].copy()
    out["gate"] = models["gate"].predict_proba(x)[:, 1]
    graph_voter, graph_cols = models["graph"]
    probs = np.column_stack(
        [model.predict_proba(x)[:, 1] for model in models["voters"].values()]
        + [graph_voter.predict_proba(x[:, graph_cols])[:, 1]]
    )
    out["regime"] = models["regime"].predict(frame[list(REGIME_COLS)].to_numpy())
    out["p_mean"] = probs.mean(axis=1)
    out["agree"] = (probs > 0.5).all(axis=1) | (probs < 0.5).all(axis=1)
    out["p_hgb"] = probs[:, 1]
    return out


def evaluate(scored: pd.DataFrame, take: np.ndarray, side: np.ndarray, cost: float, periods_per_year: float) -> dict:
    trades = scored.loc[take].copy()
    trades["net"] = side[take] * trades["target"] - cost
    per_period = trades.groupby("timestamp")["net"].mean().reindex(scored["timestamp"].unique(), fill_value=0.0)
    std = per_period.std()
    trade_std = trades["net"].std()
    return {
        "trades": int(len(trades)),
        "hit_rate": float((trades["net"] > 0).mean()) if len(trades) else float("nan"),
        "net_bps_per_trade": float(trades["net"].mean() * 1e4) if len(trades) else float("nan"),
        "t_stat": float(trades["net"].mean() / trade_std * math.sqrt(len(trades))) if len(trades) > 1 and trade_std > 0 else 0.0,
        "sharpe": float(per_period.mean() / std * math.sqrt(periods_per_year)) if std > 0 else 0.0,
        "cum_net": float(per_period.sum()),
    }


def waterfall_mask(
    scored: pd.DataFrame, gate_cut: float, conf_cut: float, allowed_regimes: list[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    confidence = (scored["p_mean"] - 0.5).abs()
    take = (scored["gate"] >= gate_cut) & scored["agree"] & (confidence >= conf_cut)
    if allowed_regimes is not None:
        take &= scored["regime"].isin(allowed_regimes)
    side = np.where(scored["p_mean"] > 0.5, 1.0, -1.0)
    return take.to_numpy(), side


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--bars-per-day", type=float, required=True)
    parser.add_argument("--hold", type=int, default=3)
    parser.add_argument("--filings", default=None, help="CSV from scripts/download_sec_8k.py")
    parser.add_argument("--cost-bps", type=float, default=1.5, help="per leg per side; hedged round trip pays 4x")
    args = parser.parse_args()

    close, volume = load_panel(args.raw)
    rows = build_rows(close, volume, args.hold, bars_per_day=int(args.bars_per_day), filings_path=args.filings)
    feature_cols = [c for c in rows.columns if c not in ("timestamp", "ticker", "target")]
    train, val, test = chronological_split(rows, args.hold)
    print(f"rows train/val/test: {len(train)}/{len(val)}/{len(test)}  hold={args.hold}")
    models = fit_models(train, feature_cols)
    val_s, test_s = score(models, val, feature_cols), score(models, test, feature_cols)
    cost = 4 * args.cost_bps / 1e4
    stamps = pd.Series(val_s["timestamp"].unique())
    ppy = len(stamps) / stamps.dt.date.nunique() * 252  # tradable bars per day after same-session target filter

    grid = []
    for gq, cq in itertools.product(QUANTILES, QUANTILES):
        gate_cut = float(np.quantile(val_s["gate"], gq))
        conf_cut = float(np.quantile((val_s["p_mean"] - 0.5).abs(), cq))
        take, side = waterfall_mask(val_s, gate_cut, conf_cut)
        stats = evaluate(val_s, take, side, cost, ppy)
        if stats["trades"] >= 100:
            grid.append((stats["sharpe"], gq, cq, gate_cut, conf_cut, stats))
    if not grid:
        print("no cutoff pair produced >= 100 validation trades")
        return
    _, gq, cq, gate_cut, conf_cut, val_stats = max(grid, key=lambda item: item[0])
    take_val, side_val = waterfall_mask(val_s, gate_cut, conf_cut)
    val_net = pd.Series(side_val * val_s["target"].to_numpy() - cost)[take_val]
    by_regime = val_net.groupby(val_s["regime"].to_numpy()[take_val]).agg(["mean", "count"])
    allowed = [int(r) for r, row in by_regime.iterrows() if row["count"] >= 30 and row["mean"] > 0]
    print("validation net by regime:", by_regime.round(5).to_dict("index"), "allowed:", allowed)
    take, side = waterfall_mask(test_s, gate_cut, conf_cut, allowed)
    test_stats = evaluate(test_s, take, side, cost, ppy)
    print(f"cutoff pairs tried: {len(grid)}; chosen gate_q={gq} conf_q={cq}")
    print("waterfall val :", val_stats)
    print("waterfall test (regime-gated):", test_stats)

    # Single-model comparison at the same trade count on validation.
    hgb_conf_val = (val_s["p_hgb"] - 0.5).abs()
    single_cut = float(np.quantile(hgb_conf_val, 1 - val_stats["trades"] / len(val_s)))
    single_take = ((test_s["p_hgb"] - 0.5).abs() >= single_cut).to_numpy()
    single_side = np.where(test_s["p_hgb"] > 0.5, 1.0, -1.0)
    print("single hgb test:", evaluate(test_s, single_take, single_side, cost, ppy))
    gross_take, gross_side = waterfall_mask(test_s, gate_cut, conf_cut, allowed)
    print("waterfall test, zero cost:", evaluate(test_s, gross_take, gross_side, 0.0, ppy))


if __name__ == "__main__":
    main()

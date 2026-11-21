"""Select the most liquid tickers with near-complete bar coverage and symlink them into one flat directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="root searched recursively for *.txt bar files")
    parser.add_argument("--out", required=True)
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--min-coverage", type=float, default=0.95, help="fraction of the max bar count")
    args = parser.parse_args()

    stats = []
    for path in Path(args.raw).rglob("*.txt"):
        try:
            frame = pd.read_csv(path, usecols=["<CLOSE>", "<VOL>"])
        except (ValueError, pd.errors.EmptyDataError):
            continue
        dollar = frame["<CLOSE>"] * frame["<VOL>"]
        stats.append((path, len(frame), float(dollar.median()), float(frame["<CLOSE>"].median())))
    table = pd.DataFrame(stats, columns=["path", "bars", "median_dollar_vol", "median_price"])
    table["name"] = table["path"].map(lambda path: path.name)
    eligible = table[(table["bars"] >= args.min_coverage * table["bars"].max()) & (table["median_price"] >= 5)]
    eligible = eligible.sort_values("median_dollar_vol", ascending=False).drop_duplicates("name")
    chosen = eligible.nlargest(args.top, "median_dollar_vol")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.txt"):
        if stale.is_symlink():
            stale.unlink()
    for path in chosen["path"]:
        (out / path.name).symlink_to(path.resolve())
    print(f"{len(table)} files, {len(eligible)} eligible, linked {len(chosen)} into {out}")


if __name__ == "__main__":
    main()

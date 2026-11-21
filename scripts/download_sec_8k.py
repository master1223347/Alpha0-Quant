"""Download SEC EDGAR 8-K filing timestamps and item codes for a ticker list (free, no key)."""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from download_yfinance_bars import DEFAULT_TICKERS

USER_AGENT = "Alpha0-Quant research contact@example.com"


def _get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--out", default="data/raw/events/sec_8k_filings.csv")
    args = parser.parse_args()

    mapping = _get_json("https://www.sec.gov/files/company_tickers.json")
    cik_by_ticker = {row["ticker"].upper(): int(row["cik_str"]) for row in mapping.values()}
    records = []
    for ticker in args.tickers:
        cik = cik_by_ticker.get(ticker.upper())
        if cik is None:
            print(f"{ticker}: no CIK")
            continue
        recent = _get_json(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")["filings"]["recent"]
        for form, accepted, items in zip(recent["form"], recent["acceptanceDateTime"], recent["items"], strict=True):
            if form in ("8-K", "8-K/A"):
                records.append({"ticker": ticker.upper(), "accepted_utc": accepted, "items": items})
        time.sleep(0.15)  # SEC fair-access limit is 10 req/s
    frame = pd.DataFrame(records)
    # EDGAR acceptanceDateTime is US Eastern despite the trailing Z.
    stamps = pd.to_datetime(frame["accepted_utc"].str.rstrip("Z").str.replace(".000", "", regex=False))
    frame["accepted_utc"] = stamps.dt.tz_localize("America/New_York").dt.tz_convert("UTC").dt.tz_localize(None)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["ticker", "accepted_utc"]).to_csv(args.out, index=False)
    print(f"{len(frame)} filings, {frame['ticker'].nunique()} tickers, {frame['accepted_utc'].min()} .. {frame['accepted_utc'].max()}")


if __name__ == "__main__":
    main()

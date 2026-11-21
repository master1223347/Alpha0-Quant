"""Download recent intraday bars from Yahoo Finance into the raw loader format (UTC <DATE>/<TIME>)."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_TICKERS = (
    "AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO JPM V MA UNH XOM JNJ PG HD COST ABBV MRK PEP KO WMT BAC CVX "
    "ADBE CRM AMD NFLX INTC CSCO ORCL QCOM TXN LLY PFE DIS MCD NKE BA CAT GE IBM GS MS C WFC T VZ"
).split()


def main() -> None:
    import yfinance as yf

    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=list(DEFAULT_TICKERS))
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--out", default="data/raw/us/nasdaq_stocks")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker in args.tickers:
        frame = yf.download(ticker, period=args.period, interval=args.interval, progress=False, auto_adjust=False)
        if frame.empty:
            print(f"{ticker}: no data")
            continue
        frame.columns = [column[0] if isinstance(column, tuple) else column for column in frame.columns]
        frame = frame.tz_convert("UTC")
        lines = ["<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>"]
        for timestamp, row in frame.iterrows():
            if row["Volume"] <= 0:
                continue
            lines.append(
                f"{ticker}.US,{args.interval},{timestamp:%Y%m%d},{timestamp:%H%M%S},{row['Open']:.6f},{row['High']:.6f},"
                f"{row['Low']:.6f},{row['Close']:.6f},{int(row['Volume'])},0"
            )
        (out_dir / f"{ticker.lower()}.us.txt").write_text("\n".join(lines) + "\n")
        print(f"{ticker}: {len(lines) - 1} bars")


if __name__ == "__main__":
    main()

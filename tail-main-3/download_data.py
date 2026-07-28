"""Download aligned multivariate price data and write a log-return CSV for fit_returns.py.

Usage:
  # ~20y of daily US ETFs (recommended first run; needs: pip install yfinance pandas)
  python download_data.py --source yahoo --tickers SPY,QQQ,TLT --max-len 5000 --out returns.csv

  # crypto from Binance public API (needs: pip install requests pandas)
  python download_data.py --source binance --symbols BTCUSDT,ETHUSDT,PAXGUSDT \
      --interval 1d --max-len 3000 --out returns.csv

Output: CSV with one column per asset, rows = time steps, values = log returns
r_t = log(P_t / P_{t-1}), aligned on common timestamps (inner join), most recent
`max_len` observations kept. Also writes <out>_prices.csv for reference.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd


# --------------------------------------------------------------------- fetchers
def fetch_yahoo(tickers: list[str], period: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    if isinstance(df, pd.Series):
        df = df.to_frame(name=tickers[0])
    return df[tickers]                      # enforce requested column order


def fetch_binance(symbols: list[str], interval: str, n_bars: int) -> pd.DataFrame:
    import requests
    series = {}
    for sym in symbols:
        rows: list = []
        end_ms = None
        while len(rows) < n_bars:
            params = {"symbol": sym, "interval": interval,
                      "limit": min(1000, n_bars - len(rows))}
            if end_ms is not None:
                params["endTime"] = end_ms
            r = requests.get("https://api.binance.com/api/v3/klines",
                             params=params, timeout=30)
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break                        # reached the start of history
            rows = batch + rows              # prepend older bars
            end_ms = batch[0][0] - 1
            time.sleep(0.25)                 # stay well under rate limits
        idx = pd.to_datetime([b[0] for b in rows], unit="ms")
        s = pd.Series([float(b[4]) for b in rows], index=idx, name=sym)  # close px
        series[sym] = s[~s.index.duplicated(keep="first")]
    return pd.concat(series.values(), axis=1, join="inner")


# --------------------------------------------------------------- processing/IO
def prices_to_returns_csv(prices: pd.DataFrame, max_len: int, out: str) -> np.ndarray:
    prices = prices.dropna(how="any").sort_index()
    if (prices <= 0).any().any():
        raise ValueError("Non-positive prices found; check the raw data.")
    prices = prices.tail(max_len + 1)        # max_len returns need max_len+1 prices
    ret = np.log(prices / prices.shift(1)).dropna(how="any")
    zero_frac = (ret == 0.0).mean()

    prices.to_csv(out.replace(".csv", "_prices.csv"))
    ret.to_csv(out, index=False, float_format="%.8e")

    print(f"assets : {list(ret.columns)}")
    print(f"span   : {ret.index[0]} -> {ret.index[-1]}   T={len(ret)} returns")
    with pd.option_context("display.float_format", "{:.4f}".format):
        stats = pd.DataFrame({"std": ret.std(), "skew": ret.skew(),
                              "ex_kurtosis": ret.kurt(), "min": ret.min(),
                              "max": ret.max(), "zero_frac": zero_frac})
        print(stats)
    print(f"\nwritten: {out} (and {out.replace('.csv', '_prices.csv')})")
    print(f"next   : python fit_returns.py --data {out} --n 24 --steps 8000 --gen 50000")
    return ret.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["yahoo", "binance"], required=True)
    ap.add_argument("--tickers", type=str, default="SPY,QQQ,TLT",
                    help="Yahoo tickers, comma-separated")
    ap.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,PAXGUSDT",
                    help="Binance symbols, comma-separated")
    ap.add_argument("--period", type=str, default="max", help="Yahoo lookback period")
    ap.add_argument("--interval", type=str, default="1d",
                    help="Binance bar interval (1d, 4h, 1h, ...)")
    ap.add_argument("--max-len", type=int, default=5000,
                    help="maximum number of returns kept (most recent)")
    ap.add_argument("--out", type=str, default="returns.csv")
    args = ap.parse_args()

    if args.source == "yahoo":
        prices = fetch_yahoo([t.strip() for t in args.tickers.split(",")], args.period)
    else:
        prices = fetch_binance([s.strip() for s in args.symbols.split(",")],
                               args.interval, args.max_len + 1)
    prices_to_returns_csv(prices, args.max_len, args.out)


if __name__ == "__main__":
    main()

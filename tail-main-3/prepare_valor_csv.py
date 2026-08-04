"""Turn a long-format price extract into the wide log-return CSV that fit_returns.py eats.

Input : one row per (VALOR, PRICE_TYPE, PRICE_DATE), semicolon-separated, columns
        VALOR;FI_ID;PRICE_TYPE;PRICE_DATE;PRICE;CURRENCY
Output: <out>            wide log returns, one column per kept VALOR, header = V<valor>,
                         no date column, "%.8e"   -> feed to fit_returns.py WITHOUT --prices
        <out>_prices.csv wide aligned price levels with a Date column (reference / --prices)
        <out>_assets.csv one row per kept VALOR with the selection + moment diagnostics
        <out>_rejected.csv every dropped VALOR with the reason

Selection logic (this is the part that separates the daily series from the quarterly ones):

  1. reference calendar  C = { dates observed by >= --calendar-quorum of all VALORs }
     i.e. the trading calendar is *recovered from the data*, no exchange calendar needed.
  2. a VALOR v is "daily" iff, restricted to [--start, --end],
        median(diff(dates_v)) <= --max-median-gap   (daily ~1, quarterly ~91)
        max(diff(dates_v))    <= --max-gap          (no multi-week hole)
        |dates_v & C| / |C|   >= --min-coverage
        first(dates_v) <= start + --edge-tol  and  last(dates_v) >= end - --edge-tol
  3. kept series are reindexed on C, holes are forward-filled up to --ffill-limit days,
     remaining NaN rows are dropped (inner join across assets).
  4. post-return filters: strictly positive prices, and zero-return fraction
     <= --max-zero-frac (stale/illiquid quotes destroy the GPD tail fit).

Usage:
    python prepare_valor_csv.py --raw prices_raw.csv --out returns_valor.csv \
        --start 2019-12-23 --end 2026-01-05
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

RAW_COLS = ["VALOR", "FI_ID", "PRICE_TYPE", "PRICE_DATE", "PRICE", "CURRENCY"]


# --------------------------------------------------------------------- parsing
def sniff_sep(path: str, encoding: str) -> str:
    with open(path, encoding=encoding, errors="replace") as fh:
        head = fh.readline()
    counts = {sep: head.count(sep) for sep in [";", "\t", "|", ","]}
    return max(counts, key=counts.get)


def to_float(s: pd.Series) -> pd.Series:
    """Swiss/European numerics: 1'234.56 / 1 234,56 / 1.234,56 -> float."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = (s.astype(str).str.strip()
         .str.replace("'", "", regex=False)
         .str.replace("\u00a0", "", regex=False)
         .str.replace(" ", "", regex=False))
    # decimal comma iff there is a comma and no dot after it
    comma_dec = t.str.contains(",") & ~t.str.contains(r",\d*\.")
    t = t.mask(comma_dec, t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    t = t.str.replace(",", "", regex=False)
    return pd.to_numeric(t, errors="coerce")


def to_date(s: pd.Series) -> pd.Series:
    txt = s.astype(str).str.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y%m%d", "%d.%m.%y"):
        out = pd.to_datetime(txt, format=fmt, errors="coerce")
        if out.notna().mean() > 0.95:
            return out.dt.normalize()
    return pd.to_datetime(txt, dayfirst=True, errors="coerce").dt.normalize()


def read_raw(path: str, encoding: str) -> pd.DataFrame:
    sep = sniff_sep(path, encoding)
    df = pd.read_csv(path, sep=sep, dtype=str, encoding=encoding,
                     engine="python", skipinitialspace=True)
    df.columns = [re.sub(r"[^A-Z_]", "", c.strip().upper().replace(" ", "_"))
                  for c in df.columns]
    missing = [c for c in RAW_COLS if c not in df.columns]
    if missing:
        raise SystemExit(f"missing columns {missing}; found {list(df.columns)} (sep={sep!r})")
    df = df[RAW_COLS].copy()
    df["VALOR"] = df["VALOR"].astype(str).str.strip()
    for c in ("FI_ID", "PRICE_TYPE", "CURRENCY"):
        df[c] = df[c].astype(str).str.strip().str.upper()
    df["PRICE_DATE"] = to_date(df["PRICE_DATE"])
    df["PRICE"] = to_float(df["PRICE"])
    n0 = len(df)
    df = df.dropna(subset=["VALOR", "PRICE_DATE", "PRICE"])
    print(f"raw: {n0} rows (sep={sep!r}) -> {len(df)} parseable, "
          f"{df.VALOR.nunique()} VALOR, PRICE_TYPE={sorted(df.PRICE_TYPE.unique())[:8]}")
    return df


# ------------------------------------------------------- one series per VALOR
def pick_one_quote(df: pd.DataFrame, price_type: str | None,
                   currency: str | None) -> pd.DataFrame:
    """Collapse (VALOR, PRICE_TYPE, CURRENCY) to one series per VALOR: the
    (type, ccy) combination with the most observations, ties broken by name."""
    if price_type:
        wanted = [p.strip().upper() for p in price_type.split(",")]
        df = df[df.PRICE_TYPE.isin(wanted)]
    if currency:
        df = df[df.CURRENCY == currency.strip().upper()]
    if df.empty:
        raise SystemExit("no rows left after PRICE_TYPE/CURRENCY filtering")

    cnt = (df.groupby(["VALOR", "PRICE_TYPE", "CURRENCY"], as_index=False)
             .agg(n=("PRICE", "size")))
    cnt = cnt.sort_values(["VALOR", "n", "PRICE_TYPE", "CURRENCY"],
                          ascending=[True, False, True, True])
    best = cnt.drop_duplicates("VALOR")[["VALOR", "PRICE_TYPE", "CURRENCY"]]
    df = df.merge(best, on=["VALOR", "PRICE_TYPE", "CURRENCY"], how="inner")
    # residual duplicates on (VALOR, date): keep the last row of the file
    return df.drop_duplicates(["VALOR", "PRICE_DATE"], keep="last")


# ------------------------------------------------------------------ selection
def classify(df: pd.DataFrame, cal: pd.DatetimeIndex, start, end, a) -> pd.DataFrame:
    rows = []
    for v, g in df.groupby("VALOR", sort=True):
        d = np.sort(g.PRICE_DATE.unique())
        gaps = np.diff(d).astype("timedelta64[D]").astype(float) if d.size > 1 else np.array([np.inf])
        cov = np.isin(cal.values, d).mean()
        rows.append(dict(
            VALOR=v, FI_ID=g.FI_ID.iloc[0], PRICE_TYPE=g.PRICE_TYPE.iloc[0],
            CURRENCY=g.CURRENCY.iloc[0], n_obs=d.size, coverage=cov,
            median_gap=float(np.median(gaps)), max_gap=float(gaps.max()),
            first=pd.Timestamp(d[0]), last=pd.Timestamp(d[-1])))
    m = pd.DataFrame(rows)
    tol = pd.Timedelta(days=a.edge_tol)
    tests = {
        "median_gap": m.median_gap <= a.max_median_gap,
        "max_gap": m.max_gap <= a.max_gap,
        "coverage": m.coverage >= a.min_coverage,
        "starts_early": m["first"] <= start + tol,
        "ends_late": m["last"] >= end - tol,
    }
    m["daily"] = np.logical_and.reduce(list(tests.values()))
    m["reason"] = ["ok" if ok else ",".join(k for k, t in tests.items() if not t.iloc[i])
                   for i, ok in enumerate(m.daily)]
    return m.sort_values(["daily", "coverage"], ascending=[False, False])


# ----------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--out", default="returns_valor.csv")
    p.add_argument("--start", default="2019-12-23")
    p.add_argument("--end", default="2026-01-05")
    p.add_argument("--encoding", default="utf-8-sig")
    p.add_argument("--price-type", default=None, help="e.g. CLOSE or NAV,CLOSE (default: auto)")
    p.add_argument("--currency", default=None, help="keep only this currency")
    p.add_argument("--calendar-quorum", type=float, default=0.5)
    p.add_argument("--min-coverage", type=float, default=0.98)
    p.add_argument("--max-median-gap", type=float, default=5.0, help="days")
    p.add_argument("--max-gap", type=float, default=10.0, help="days")
    p.add_argument("--edge-tol", type=int, default=7, help="days of slack at both ends")
    p.add_argument("--ffill-limit", type=int, default=2, help="0 = pure inner join")
    p.add_argument("--max-zero-frac", type=float, default=0.30)
    p.add_argument("--max-assets", type=int, default=0, help="0 = keep all daily VALOR")
    p.add_argument("--valors", default=None, help="comma-separated whitelist")
    a = p.parse_args()

    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    df = read_raw(a.raw, a.encoding)
    df = df[(df.PRICE_DATE >= start) & (df.PRICE_DATE <= end)]
    if a.valors:
        df = df[df.VALOR.isin([v.strip() for v in a.valors.split(",")])]
    df = pick_one_quote(df, a.price_type, a.currency)

    # ---- reference trading calendar, recovered from the data
    per_date = df.groupby("PRICE_DATE").VALOR.nunique()
    cal = pd.DatetimeIndex(sorted(per_date[per_date >= a.calendar_quorum *
                                           df.VALOR.nunique()].index))
    print(f"calendar: {len(cal)} dates {cal[0].date()} -> {cal[-1].date()} "
          f"({len(cal) / max((end - start).days / 365.25, 1e-9):.0f}/yr)")

    # ---- daily vs non-daily
    meta = classify(df, cal, start, end, a)
    keep = meta[meta.daily].VALOR.tolist()
    if a.max_assets and len(keep) > a.max_assets:
        keep = meta[meta.daily].nlargest(a.max_assets, "n_obs").VALOR.tolist()
    print(f"daily VALOR: {meta.daily.sum()} kept / {len(meta)} total | "
          f"dropped reasons: {meta.loc[~meta.daily, 'reason'].value_counts().to_dict()}")
    if not keep:
        raise SystemExit("no VALOR passed the daily test; relax --min-coverage/--max-gap")

    # ---- wide price panel on the reference calendar
    px = (df[df.VALOR.isin(keep)]
          .pivot(index="PRICE_DATE", columns="VALOR", values="PRICE")
          .reindex(cal).sort_index())
    if a.ffill_limit > 0:
        px = px.ffill(limit=a.ffill_limit)
    before = len(px)
    px = px.dropna(how="any")
    print(f"panel: {before} calendar dates -> {len(px)} complete rows, {px.shape[1]} assets")
    if (px <= 0).any().any():
        bad = px.columns[(px <= 0).any()].tolist()
        print(f"dropping {len(bad)} VALOR with non-positive prices: {bad[:10]}")
        px = px.drop(columns=bad)

    # ---- log returns + liquidity filter
    ret = np.log(px / px.shift(1)).dropna(how="any")
    zf = (ret == 0.0).mean()
    stale = zf[zf > a.max_zero_frac].index.tolist()
    if stale:
        print(f"dropping {len(stale)} stale VALOR (zero-return frac > {a.max_zero_frac}): "
              f"{stale[:10]}")
        px, ret = px.drop(columns=stale), ret.drop(columns=stale)
        zf = zf.drop(stale)
    if ret.shape[1] == 0:
        raise SystemExit("everything filtered out; loosen --max-zero-frac")

    # ---- write. header "V<valor>" is deliberately non-numeric: a purely numeric
    # header would be silently parsed as a *data row* by fit_returns.load_returns.
    ret.columns = [f"V{c}" for c in ret.columns]
    px.columns = ret.columns
    ret.to_csv(a.out, index=False, float_format="%.8e")
    px.to_csv(a.out.replace(".csv", "_prices.csv"), index_label="Date")

    stats = pd.DataFrame({"std": ret.std(), "skew": ret.skew(),
                          "ex_kurtosis": ret.kurt(), "min": ret.min(),
                          "max": ret.max(), "zero_frac": zf.values})
    meta.set_index(meta.VALOR.map(lambda v: f"V{v}")).join(stats, how="inner") \
        .to_csv(a.out.replace(".csv", "_assets.csv"))
    meta[~meta.daily].to_csv(a.out.replace(".csv", "_rejected.csv"), index=False)

    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.max_rows", 40):
        print(stats)
    print(f"\nT={len(ret)} returns, f={ret.shape[1]} assets, "
          f"{ret.index[0].date()} -> {ret.index[-1].date()}")
    print(f"written: {a.out} (+ _prices.csv, _assets.csv, _rejected.csv)")
    print(f"next   : python fit_returns.py --data {a.out} --n 24 --test-frac 0.2 "
          f"--horizon 10 --seed 0 --steps 20000 --gen 50000 --outdir run_out")


if __name__ == "__main__":
    main()

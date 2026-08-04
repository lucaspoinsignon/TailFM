"""Find the columns that break the EVT marginals, BEFORE you burn a training run.

    python diagnose_data.py --data returns_valor.csv --n 24 --test-frac 0.2
    python diagnose_data.py --data returns_valor.csv --write-keep keep.txt

Reports, per asset on the TRAIN slice of the raw return matrix:
  unique        number of distinct return values
  maxtie        largest fraction of observations sharing a single value
  zeros         fraction of returns exactly equal to 0
  below/above   STRICT exceedances at the nominal q_tail thresholds -- if either is 0,
                `stats.genpareto.fit` receives an empty array and raises
                "zero-size array to reduction operation minimum which has no identity"
  q_lo/q_hi     the threshold levels the robust marginal would actually use
"""

from __future__ import annotations

import argparse

import numpy as np

from fit_returns import feature_names_from_csv, load_returns
from tailfm.evt_robust import screen_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--q-tail", type=float, default=0.05)
    ap.add_argument("--k-min", type=int, default=25)
    ap.add_argument("--q-max", type=float, default=0.25)
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--write-keep", type=str, default=None)
    args = ap.parse_args()

    r = load_returns(args.data, args.prices)
    T, f = r.shape
    names = feature_names_from_csv(args.data, f)
    split = int((1.0 - args.test_frac) * T)
    train = r[:split]
    print(f"T={T} (train {split}), f={f}, q_tail={args.q_tail}, k_min={args.k_min}")

    rows = screen_features(train, names, args.q_tail, args.k_min, args.q_max)
    bad = [x for x in rows if not x["ok"]]
    tied = [x for x in rows if x["ok"] and
            (x["n_below_nominal"] < args.k_min or x["n_above_nominal"] < args.k_min)]
    zeros = sorted((x for x in rows), key=lambda d: -d["zero_frac"])

    print(f"\n{len(bad)} unfittable / {len(tied)} needing a raised threshold / "
          f"{f - len(bad) - len(tied)} clean")

    hdr = (f"{'name':>14s} {'ok':>5s} {'unique':>7s} {'maxtie':>7s} {'zeros':>7s} "
           f"{'below':>6s} {'above':>6s} {'q_lo':>7s} {'q_hi':>7s}")
    def show(title, subset):
        if not subset:
            return
        print(f"\n--- {title} ---\n{hdr}")
        for x in subset[:args.show]:
            ql = f"{x['q_lo']:.3f}" if x["q_lo"] is not None else "  --"
            qh = f"{x['q_hi']:.3f}" if x["q_hi"] is not None else "  --"
            print(f"{x['name']:>14s} {str(x['ok']):>5s} {x['n_unique']:>7d} "
                  f"{100*x['max_tie_frac']:>6.1f}% {100*x['zero_frac']:>6.1f}% "
                  f"{x['n_below_nominal']:>6d} {x['n_above_nominal']:>6d} {ql:>7s} {qh:>7s}")

    show("UNFITTABLE (drop these)", bad)
    show("threshold will be raised (usable, but check them)", tied)
    show("highest zero-return fraction", [x for x in zeros if x["zero_frac"] > 0][:args.show])

    keep = [x["name"] for x in rows if x["ok"]]
    print(f"\nkeep {len(keep)}/{f} columns")
    if args.write_keep:
        open(args.write_keep, "w").write("\n".join(keep) + "\n")
        print(f"written to {args.write_keep}")


if __name__ == "__main__":
    main()

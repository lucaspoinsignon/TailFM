"""End-to-end demonstration on synthetic data with known ground-truth tail structure.

Runs the full pipeline
    EVT PIT -> CFM (Student-t base, shared mixing) -> inverse PIT -> risk report
and, with --ablation, trains a vanilla baseline (Gaussian base, standardized raw data,
no EVT wrapper) to make the tail failure mode of standard flow matching visible on the
same diagnostics.

Usage:
    python example.py --quick               # small model / few steps (CPU smoke test)
    python example.py --ablation            # also train the vanilla baseline
    python example.py                       # full-size run
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tailfm import (MarginalEnsemble, VelocityField, train_cfm, sample,
                    estimate_risk, kupiec_test, portfolio_losses,
                    make_windows, synthetic_market, print_report,
                    tail_dependence_report, var_cvar_empirical)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    if args.quick:
        T, n, stride, steps, d_model, depth, M, ode_steps = 20_000, 24, 4, 800, 64, 3, 2048, 60
    else:
        T, n, stride, steps, d_model, depth, M, ode_steps = 100_000, 32, 2, 8000, 128, 4, 8192, 100
    h, alphas = 10, (0.95, 0.99, 0.995)
    names = ["A(dep)", "B(dep)", "C(ind)", "D(gauss)"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------ data
    print(f"[1/6] Simulating synthetic market (T={T}) ...")
    series = synthetic_market(T, seed=args.seed)
    test_series = synthetic_market(T // 2, seed=args.seed + 1)      # held out
    real = make_windows(series, n, stride)
    real_test = make_windows(test_series, n, stride=n)              # non-overlapping
    print(f"      train windows: {real.shape},  test windows: {real_test.shape}")

    # ------------------------------------------------- EVT marginals + PIT
    print("[2/6] Fitting semi-parametric EVT marginals and applying the t_nu PIT ...")
    marg = MarginalEnsemble(q_tail=0.05, nu="auto").fit(real)
    print(f"      reference nu = {marg.nu_:.2f}   "
          f"xi_lower per feature = "
          f"{[round(m.xi_lo_, 2) for m in marg.marginals_]}")
    z = torch.tensor(marg.transform(real), dtype=torch.float32)

    # ------------------------------------------------------------- training
    print(f"[3/6] Training CFM (Student-t base, shared mixing) on {device} ...")
    model = VelocityField(f=4, n_max=n, d_model=d_model, depth=depth)
    ema, _ = train_cfm(model, z, nu=marg.nu_, steps=steps, device=device,
                       seed=args.seed)

    # ------------------------------------------------------------- sampling
    print(f"[4/6] Sampling {M} windows ({ode_steps} Heun steps) ...")
    z_gen = sample(ema.shadow, M, n, 4, nu=marg.nu_, n_steps=ode_steps,
                   device=device, seed=args.seed)
    gen = marg.inverse_transform(z_gen.numpy())

    # ------------------------------------------------------------ ablation
    gen_vanilla = None
    if args.ablation:
        print("[4b ] Ablation: vanilla FM (Gaussian base, standardized raw data) ...")
        mu, sd = real.reshape(-1, 4).mean(0), real.reshape(-1, 4).std(0)
        zv = torch.tensor((real - mu) / sd, dtype=torch.float32)
        model_v = VelocityField(f=4, n_max=n, d_model=d_model, depth=depth)
        # nu -> inf is Gaussian; nu=1e6 with per-window mixing ~ N(0, I) exactly enough
        ema_v, _ = train_cfm(model_v, zv, nu=1e6, steps=steps, device=device,
                             seed=args.seed)
        zv_gen = sample(ema_v.shadow, M, n, 4, nu=1e6, n_steps=ode_steps,
                        device=device, seed=args.seed)
        gen_vanilla = zv_gen.numpy() * sd + mu

    # ----------------------------------------------------------- diagnostics
    print("\n[5/6] Tail diagnostics: TAIL-AWARE model vs real")
    print_report(real, gen, names)
    if gen_vanilla is not None:
        print("\n[5b ] Tail diagnostics: VANILLA model vs real")
        print_report(real, gen_vanilla, names)

    # ------------------------------------------------------------ risk report
    print(f"\n[6/6] Portfolio risk (equal weights, horizon h={h}) ...")
    report = estimate_risk(gen, alphas=alphas, horizon=h, n_boot=200, seed=args.seed)

    # ground truth from a very long fresh simulation of the same DGP
    big = synthetic_market(2_000_000, seed=args.seed + 7)
    L_true = portfolio_losses(make_windows(big, h, stride=h), horizon=h)
    L_test = portfolio_losses(make_windows(test_series, h, stride=h), horizon=h)

    print(f"{'alpha':>7} | {'VaR gen (GPD)':>14} {'95% CI':>20} | {'VaR true':>9} |"
          f" {'CVaR gen (GPD)':>14} {'95% CI':>20} | {'CVaR true':>9}")
    for a in alphas:
        r = report[a]
        vt, ct = var_cvar_empirical(L_true, a)
        print(f"{a:7.3f} | {r['var_gpd']:14.4f} "
              f"[{r['var_ci'][0]:8.4f},{r['var_ci'][1]:8.4f}] | {vt:9.4f} |"
              f" {r['cvar_gpd']:14.4f} "
              f"[{r['cvar_ci'][0]:8.4f},{r['cvar_ci'][1]:8.4f}] | {ct:9.4f}")

    print("\nKupiec unconditional-coverage backtest on held-out losses:")
    for a in alphas:
        k = kupiec_test(L_test, report[a]["var_gpd"], a)
        print(f"  a={a:5.3f}: exceed {k['exceedances']:3d} / expected "
              f"{k['expected']:6.1f}  (N={k['n']})   LR_uc={k['LR_uc']:6.2f}  "
              f"p={k['p_value']:.3f}")

    # ---------------------------------------------------------------- figures
    make_figures(real, gen, gen_vanilla, names, h, args.outdir)
    print(f"\nFigures written to {args.outdir}/tail_diagnostics.png")


def make_figures(real, gen, gen_vanilla, names, h, outdir):
    import os
    os.makedirs(outdir, exist_ok=True)
    q_grid = np.linspace(0.005, 0.10, 20)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) tail-dependence curves: dependent pair vs independent pair
    ax = axes[0, 0]
    for pair, style in [((0, 1), "-"), ((0, 2), "--")]:
        td = tail_dependence_report(real, gen, q_grid)[pair]
        ax.plot(q_grid, td[0], "k" + style, label=f"real {names[pair[0]]},{names[pair[1]]}")
        ax.plot(q_grid, td[1], "C0" + style, label=f"gen  {names[pair[0]]},{names[pair[1]]}")
        if gen_vanilla is not None:
            tdv = tail_dependence_report(real, gen_vanilla, q_grid)[pair]
            ax.plot(q_grid, tdv[1], "C3" + style, alpha=0.8,
                    label=f"vanilla {names[pair[0]]},{names[pair[1]]}")
    ax.set_xlabel("q"); ax.set_ylabel(r"$\hat\lambda_L(q)$")
    ax.set_title("Lower tail dependence (co-crash) curves")
    ax.legend(fontsize=7); ax.set_ylim(0, 1)

    # (b) lower-tail QQ plot, feature 0
    ax = axes[0, 1]
    ql = np.linspace(0.0005, 0.05, 200)
    r0 = real.reshape(-1, 4)[:, 0]
    ax.plot(np.quantile(r0, ql), np.quantile(gen.reshape(-1, 4)[:, 0], ql),
            "C0.", ms=3, label="tail-aware")
    if gen_vanilla is not None:
        ax.plot(np.quantile(r0, ql), np.quantile(gen_vanilla.reshape(-1, 4)[:, 0], ql),
                "C3.", ms=3, label="vanilla")
    lims = [np.quantile(r0, 0.0005), np.quantile(r0, 0.05)]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("real quantile"); ax.set_ylabel("generated quantile")
    ax.set_title(f"Lower-tail QQ, {names[0]} (q in [0.05%, 5%])"); ax.legend(fontsize=8)

    # (c) portfolio loss survival function (log scale)
    ax = axes[1, 0]
    for lab, w, c in [("real", real, "k"), ("tail-aware", gen, "C0")] + (
            [("vanilla", gen_vanilla, "C3")] if gen_vanilla is not None else []):
        L = np.sort(portfolio_losses(w, horizon=h))
        sf = 1.0 - np.arange(1, L.size + 1) / (L.size + 1)
        ax.semilogy(L, sf, c, label=lab)
    ax.set_xlabel(f"portfolio loss ({h}-step)"); ax.set_ylabel("P(L > l)")
    ax.set_title("Loss survival function"); ax.legend(fontsize=8)

    # (d) marginal CVaR(99%) per feature
    ax = axes[1, 1]
    width, xs = 0.25, np.arange(4)
    cv = lambda w, j: var_cvar_empirical(-w.reshape(-1, 4)[:, j], 0.99)[1]
    ax.bar(xs - width, [cv(real, j) for j in range(4)], width, color="k", label="real")
    ax.bar(xs, [cv(gen, j) for j in range(4)], width, color="C0", label="tail-aware")
    if gen_vanilla is not None:
        ax.bar(xs + width, [cv(gen_vanilla, j) for j in range(4)], width,
               color="C3", label="vanilla")
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_title("Marginal 1-step CVaR at 99%"); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f"{outdir}/tail_diagnostics.png", dpi=140)


if __name__ == "__main__":
    main()

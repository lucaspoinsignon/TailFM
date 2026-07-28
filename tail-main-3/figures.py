"""Diagnostic figures for real vs. generated windows -- one PNG per diagnostic.

Shared by `fit_returns.py` (a single generator) and `run_baselines.py` (several),
so both runners emit the same files with the same conventions.  Each function
takes an ordered mapping ``gens = {label: (M, n, f) array}`` and compares every
entry with the same real windows (always black, ``lw=2``):

    qq_lower_tail.png            lower-tail QQ, one panel per feature + pooled
    tail_dependence.png          lambda_L(q), one panel per feature pair
    portfolio_loss_survival.png  P(L > l) of the h-step portfolio loss
    empirical_distributions.png  per feature: log-density and 1-step loss survival

Colours follow the insertion order of `gens` (C0, C1, ...), so a label keeps its
colour across all four figures.
"""

from __future__ import annotations

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tailfm import portfolio_losses, tail_dependence_report

REAL_KW = dict(color="k", lw=2)                 # the real sample, everywhere


def model_colors(gens: dict) -> dict:
    return {name: f"C{i}" for i, name in enumerate(gens)}


def _grid(k: int, max_cols: int = 3) -> tuple[int, int]:
    """(nrow, ncol) for k panels, at most `max_cols` per row."""
    ncol = min(max_cols, max(k, 1))
    return math.ceil(k / ncol), ncol


def _survival(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted sample and its empirical survival function 1 - i/(N+1)."""
    xs = np.sort(np.asarray(x, dtype=float))
    return xs, 1.0 - np.arange(1, xs.size + 1) / (xs.size + 1.0)


def _blank(axes, used: int) -> None:
    for ax in axes.ravel()[used:]:
        ax.axis("off")


# --------------------------------------------------------------------- QQ
def qq_figure(real, gens, names, path, q_lo=0.001, q_hi=0.05, n_q=150):
    """Lower-tail QQ plots: one panel per feature, plus one with features pooled.

    Points below the 45-degree line mean the generated lower quantile is more
    negative than the real one, i.e. the generated left tail is too heavy.
    """
    f = real.shape[-1]
    colors = model_colors(gens)
    R = real.reshape(-1, f)
    G = {name: gen.reshape(-1, f) for name, gen in gens.items()}
    ql = np.linspace(q_lo, q_hi, n_q)

    panels = [(names[j], R[:, j], {k: g[:, j] for k, g in G.items()})
              for j in range(f)]
    if f > 1:
        panels.append(("pooled features", R, {k: g for k, g in G.items()}))

    nrow, ncol = _grid(len(panels))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 4.3 * nrow),
                             squeeze=False)
    for ax, (title, r, gs) in zip(axes.ravel(), panels):
        rq = np.quantile(r, ql)
        lo = float(rq.min())
        for name, g in gs.items():
            gq = np.quantile(g, ql)
            ax.plot(rq, gq, ".", ms=3, color=colors[name], label=name)
            lo = min(lo, float(gq.min()))
        ax.plot([lo, 0.0], [lo, 0.0], "k--", lw=1)
        ax.set_title(title)
        ax.set_xlabel("real quantile")
        ax.set_ylabel("generated quantile")
        ax.legend(fontsize=7)
    _blank(axes, len(panels))
    fig.suptitle(f"Lower-tail QQ, q in [{q_lo:.1%}, {q_hi:.1%}] "
                 "(below the diagonal = generated tail too heavy)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# --------------------------------------------------------- tail dependence
def tail_dependence_figure(real, gens, names, path, q_grid=None, tail="lower"):
    """lambda_L(q) = P(U_i < q, U_j < q) / q for every feature pair, one panel each."""
    f = real.shape[-1]
    if f < 2:
        return None
    if q_grid is None:
        q_grid = np.linspace(0.005, 0.10, 20)
    colors = model_colors(gens)

    # One report per model; each already contains every pair -- and the real
    # curve, which is recomputed identically inside every call, so take it from
    # the first report instead of paying for a separate (real, real) call.
    reports = {name: tail_dependence_report(real, gen, q_grid, tail)
               for name, gen in gens.items()}
    any_report = next(iter(reports.values()))
    pairs = [k for k in any_report if k != "q_grid"]

    nrow, ncol = _grid(len(pairs))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.9 * ncol, 4.1 * nrow),
                             squeeze=False)
    for ax, pair in zip(axes.ravel(), pairs):
        i, j = pair
        ax.plot(q_grid, any_report[pair][0], label="real", **REAL_KW)
        for name, rep in reports.items():
            ax.plot(q_grid, rep[pair][1], "--", color=colors[name], label=name)
        ax.set_ylim(0, 1)
        ax.set_title(f"({names[i]}, {names[j]})")
        ax.set_xlabel("q")
        ax.set_ylabel(r"$\hat\lambda_L(q)$")
        ax.legend(fontsize=7)
    _blank(axes, len(pairs))
    fig.suptitle(rf"{tail.capitalize()} tail dependence $\hat\lambda(q)$ "
                 "per feature pair (plateau > 0 = co-crash, decay to 0 = tail independence)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------ portfolio survival
def portfolio_survival_figure(real, gens, path, weights=None, horizon=10,
                              real_label="real (train)"):
    """Survival function of the h-step portfolio loss, log y-axis."""
    colors = model_colors(gens)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    L, sf = _survival(portfolio_losses(real, weights=weights, horizon=horizon))
    ax.semilogy(L, sf, label=real_label, **REAL_KW)
    for name, gen in gens.items():
        L, sf = _survival(portfolio_losses(gen, weights=weights, horizon=horizon))
        ax.semilogy(L, sf, color=colors[name], label=name)
    ax.set_title(f"Portfolio loss survival (h={horizon})")
    ax.set_xlabel("loss l")
    ax.set_ylabel("P(L > l)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# -------------------------------------------- per-feature marginal picture
def empirical_distribution_figure(real, gens, names, path):
    """Per feature: pooled 1-step density (log y) and loss survival P(-X > x)."""
    f = real.shape[-1]
    colors = model_colors(gens)
    R = real.reshape(-1, f)
    G = {name: gen.reshape(-1, f) for name, gen in gens.items()}

    fig, axes = plt.subplots(2, f, figsize=(5.4 * f, 8.2), squeeze=False)
    for j in range(f):
        # row 1: pooled 1-step empirical density on a log scale (both tails)
        ax = axes[0][j]
        lo = min(R[:, j].min(), *[g[:, j].min() for g in G.values()])
        hi = max(R[:, j].max(), *[g[:, j].max() for g in G.values()])
        bins = np.linspace(lo, hi, 120)
        ax.hist(R[:, j], bins=bins, density=True, histtype="step",
                label="real", **REAL_KW)
        for name, g in G.items():
            ax.hist(g[:, j], bins=bins, density=True, histtype="step",
                    color=colors[name], label=name)
        ax.set_yscale("log")
        ax.set_title(f"{names[j]}: empirical density (log scale)")
        ax.set_xlabel("1-step return")
        if j == 0:
            ax.set_ylabel("density")
        ax.legend(fontsize=7)
        # row 2: 1-step loss survival P(-X > x), the lower tail head-on
        ax = axes[1][j]
        Lr, sr = _survival(-R[:, j])
        ax.semilogy(Lr, sr, label="real", **REAL_KW)
        xmax = Lr[-1]
        for name, g in G.items():
            Lg, sg = _survival(-g[:, j])
            ax.semilogy(Lg, sg, color=colors[name], label=name)
            xmax = max(xmax, Lg[-1])
        ax.set_xlim(0, 1.02 * xmax)
        ax.set_ylim(1e-5, 1)
        ax.set_title(f"{names[j]}: loss survival P(-X > x)")
        ax.set_xlabel("loss x")
        if j == 0:
            ax.set_ylabel("P(-X > x)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ driver
def save_all_figures(real, gens, names, outdir, weights=None, horizon=10,
                     prefix: str = "") -> list[str]:
    """Write the four diagnostic PNGs into `outdir`; return the paths written."""
    os.makedirs(outdir, exist_ok=True)
    p = lambda stem: os.path.join(outdir, f"{prefix}{stem}.png")
    paths = [
        qq_figure(real, gens, names, p("qq_lower_tail")),
        tail_dependence_figure(real, gens, names, p("tail_dependence")),
        portfolio_survival_figure(real, gens, p("portfolio_loss_survival"),
                                  weights=weights, horizon=horizon),
        empirical_distribution_figure(real, gens, names,
                                      p("empirical_distributions")),
    ]
    return [q for q in paths if q is not None]

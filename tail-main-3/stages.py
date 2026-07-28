"""Checkpointed runner mirroring example.py --quick --ablation, one stage per call."""
import pickle, sys
import numpy as np
import torch
from tailfm import (MarginalEnsemble, VelocityField, train_cfm, sample,
                    estimate_risk, kupiec_test, portfolio_losses,
                    make_windows, synthetic_market, print_report,
                    var_cvar_empirical)

CFG = dict(T=20_000, n=24, stride=4, steps=600, d=64, depth=3,
           M=1536, ode=40, h=10, alphas=(0.95, 0.99, 0.995), seed=0)
NAMES = ["A(dep)", "B(dep)", "C(ind)", "D(gauss)"]


def net():
    return VelocityField(f=4, n_max=CFG["n"], d_model=CFG["d"], depth=CFG["depth"])


def stage_a():
    series = synthetic_market(CFG["T"], seed=CFG["seed"])
    real = make_windows(series, CFG["n"], CFG["stride"])
    np.save("ck_real.npy", real)
    marg = MarginalEnsemble(q_tail=0.05, nu="auto").fit(real)
    pickle.dump(marg, open("ck_marg.pkl", "wb"))
    print("windows:", real.shape, " nu =", round(marg.nu_, 2),
          " xi_lo =", [round(m.xi_lo_, 2) for m in marg.marginals_])
    z = torch.tensor(marg.transform(real), dtype=torch.float32)
    ema, _ = train_cfm(net(), z, nu=marg.nu_, steps=CFG["steps"], log_every=150,
                       seed=CFG["seed"])
    torch.save(ema.shadow.state_dict(), "ck_tail.pt")


def stage_b():
    marg = pickle.load(open("ck_marg.pkl", "rb"))
    m = net(); m.load_state_dict(torch.load("ck_tail.pt"))
    zg = sample(m, CFG["M"], CFG["n"], 4, nu=marg.nu_, n_steps=CFG["ode"],
                seed=CFG["seed"])
    gen = marg.inverse_transform(zg.numpy())
    np.save("ck_gen.npy", gen)
    print("gen:", gen.shape, "finite:", np.isfinite(gen).all())


def stage_c():
    real = np.load("ck_real.npy")
    mu, sd = real.reshape(-1, 4).mean(0), real.reshape(-1, 4).std(0)
    np.save("ck_musd.npy", np.stack([mu, sd]))
    zv = torch.tensor((real - mu) / sd, dtype=torch.float32)
    ema, _ = train_cfm(net(), zv, nu=1e6, steps=CFG["steps"], log_every=150,
                       seed=CFG["seed"])
    torch.save(ema.shadow.state_dict(), "ck_van.pt")


def stage_d():
    mu, sd = np.load("ck_musd.npy")
    m = net(); m.load_state_dict(torch.load("ck_van.pt"))
    zg = sample(m, CFG["M"], CFG["n"], 4, nu=1e6, n_steps=CFG["ode"],
                seed=CFG["seed"])
    gen_v = zg.numpy() * sd + mu
    np.save("ck_genv.npy", gen_v)
    print("gen vanilla:", gen_v.shape, "finite:", np.isfinite(gen_v).all())


def stage_e():
    real, gen, gen_v = (np.load(f) for f in ("ck_real.npy", "ck_gen.npy", "ck_genv.npy"))
    h, alphas = CFG["h"], CFG["alphas"]

    print("=== TAIL-AWARE vs real ===")
    print_report(real, gen, NAMES)
    print("\n=== VANILLA vs real ===")
    print_report(real, gen_v, NAMES)

    print("\n=== Portfolio risk (equal weights, h=10) ===")
    rep = estimate_risk(gen, alphas=alphas, horizon=h, n_boot=150, seed=0)
    rep_v = estimate_risk(gen_v, alphas=alphas, horizon=h, n_boot=150, seed=0)
    big = synthetic_market(2_000_000, seed=CFG["seed"] + 7)
    L_true = portfolio_losses(make_windows(big, h, stride=h), horizon=h)
    test = synthetic_market(CFG["T"] // 2, seed=CFG["seed"] + 1)
    L_test = portfolio_losses(make_windows(test, h, stride=h), horizon=h)

    for a in alphas:
        vt, ct = var_cvar_empirical(L_true, a)
        r, rv = rep[a], rep_v[a]
        print(f"a={a:5.3f}  TRUE VaR {vt:7.4f} CVaR {ct:7.4f} | "
              f"TAIL-AWARE VaR {r['var_gpd']:7.4f} [{r['var_ci'][0]:.4f},{r['var_ci'][1]:.4f}] "
              f"CVaR {r['cvar_gpd']:7.4f} [{r['cvar_ci'][0]:.4f},{r['cvar_ci'][1]:.4f}] | "
              f"VANILLA VaR {rv['var_gpd']:7.4f} CVaR {rv['cvar_gpd']:7.4f}")

    print("\nKupiec backtest (held-out, non-overlapping 10-step blocks):")
    for a in alphas:
        k = kupiec_test(L_test, rep[a]["var_gpd"], a)
        kv = kupiec_test(L_test, rep_v[a]["var_gpd"], a)
        print(f"  a={a:5.3f} tail-aware: exc {k['exceedances']:3d}/exp {k['expected']:5.1f} "
              f"p={k['p_value']:.3f} | vanilla: exc {kv['exceedances']:3d} p={kv['p_value']:.3f}")

    from example import make_figures
    make_figures(real, gen, gen_v, NAMES, h, "outputs")
    print("figures saved.")


if __name__ == "__main__":
    {"a": stage_a, "b": stage_b, "c": stage_c, "d": stage_d, "e": stage_e}[sys.argv[1]]()

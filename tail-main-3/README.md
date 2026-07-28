# tailfm — Tail-aware Flow Matching for Multivariate Time Series

> **Quickstart (merged package: tailfm + the three baselines).**
> Every script mirrors its console output into a `.log` file and writes four
> diagnostic PNGs; `--data/--n/--test-frac/--horizon/--seed` (and `--prices`)
> must be identical across all three commands or the comparison is not valid.
>
> ```bash
> COMMON="--data data/timeseries.csv --prices --n 24 --test-frac 0.2 --horizon 10 --seed 0"
>
> # 1) tailfm  -> run_out/{generated_windows.npy, report.log, *.png}
> python fit_returns.py $COMMON --steps 20000 --gen 50000 \
>        --d-model 128 --depth 4 --device cuda --outdir run_out
>
> # 2) the three baselines + tailfm in the comparison table
> #    -> baseline_out/{gen_<model>.npy, report.log, *.png}
> python run_baselines.py $COMMON --gen 50000 --device cuda --outdir baseline_out \
>        --timegan-iters 50000 \
>        --tailgan-alphas 0.05,0.01,0.005 --tailgan-epochs 10000 \
>        --timevae-recon-wt 100 --timevae-latent 32 \
>        --tailfm-gen run_out/generated_windows.npy
>
> # 3) multi-portfolio backtest on the saved samples (no retraining)
> python backtest_portfolios.py $COMMON --gen-dir baseline_out \
>        --tailfm-gen run_out/generated_windows.npy
> ```
>
> Smoke test first: `python run_baselines.py $COMMON --quick --outdir smoke_base`.
> Helper scripts: `diagnostic_csv.py` (find the values that break `np.log`),
> `diagnostic_timevae.py` (posterior collapse vs incompressible data).


Generative modeling of multivariate return windows `x ∈ R^{n×f}` with conditional flow
matching, engineered so that **VaR / CVaR estimated from generated scenarios is
calibrated** — including joint tail events ("crash of A implies crash of B").

## Why vanilla flow matching fails at tails

1. **Lipschitz argument.** The flow map of an ODE with `L_t`-Lipschitz velocity has
   `Lip(φ₁) ≤ exp(∫₀¹ L_t dt)` (Grönwall), and Lipschitz images of a Gaussian are
   sub-Gaussian. A Gaussian-base flow with any bounded-Lipschitz network therefore has
   exponentially light tails, regardless of fit quality in the body (cf. Jaini et al.
   2020, *Tails of Lipschitz Triangular Flows*).
2. **Statistical argument.** The CFM L² loss weights regions by their probability mass;
   the optimum `v*(x,t) = E[x₁ − x₀ | x_t = x]` is estimated from almost no data in the
   tail and regresses to the bulk.
3. **Dependence argument.** The Gaussian has tail-dependence coefficient
   `λ = lim_{q→1} P(X₂ > F₂⁻¹(q) | X₁ > F₁⁻¹(q)) = 0` for any ρ < 1: the network must
   manufacture all co-crash structure from a source that has none.

## What this package does instead

**(1) EVT marginal wrapper (`evt.py`).** Per feature: empirical CDF in the body, GPD
tails beyond the ±q_tail thresholds (POT / Pickands–Balkema–de Haan). PIT to a
Student-t_ν reference scale, `z = T_ν⁻¹(F̂(x))`, with ν auto-set from Hill indices.
Strictly increasing marginal maps leave the copula invariant, so the flow learns exactly
the temporal + cross-sectional dependence; generated marginals are GPD **by
construction**, with closed-form tail risk
`ES_α = VaR_α/(1−ξ) + (β−ξu)/(1−ξ)`.

**(2) Heavy-tailed base with shared mixing (`base.py`).**
`x₀ = z·√(ν/W)`, `z ~ N(0,I)`, `W ~ χ²_ν` shared per window ⇒ elliptical t source with
(i) regularly varying marginals matching the PIT scale — in z-space, source and target
**marginals coincide**, so the transport is purely a copula transport, a
bounded-Lipschitz task; (ii) strictly positive base tail dependence
`λ = 2 t_{ν+1}(−√((ν+1)(1−ρ)/(1+ρ)))`, which the flow *modulates* per pair (up for
co-crashing pairs, down toward 0 for tail-independent ones) instead of creating from
nothing. CFM theory permits arbitrary sources (Tong et al. 2023), so nothing else
changes.

**(3) Risk estimators (`risk.py`).** From M generated windows: portfolio h-step losses,
empirical and **GPD-refined** VaR/CVaR (POT re-fit on generated losses ⇒ variance-reduced
extrapolation beyond the 1−1/M empirical resolution), Rockafellar–Uryasev-consistent
CVaR, bootstrap CIs, and a Kupiec unconditional-coverage backtest on non-overlapping
held-out blocks.

**Model (`model.py`, `cfm.py`).** DiT-style temporal transformer velocity field
(adaLN-Zero conditioning on flow time; channel mixing in projections/MLPs, time mixing
in attention), OT-path CFM loss, EMA, Heun ODE sampler.

## Verified results (synthetic market with known ground truth, quick CPU config)

DGP: features A,B share a t₄ shock (tail-dependent, ρ=0.7); C is t₄ tail-independent;
D is Gaussian; common stochastic-vol factor. 600 training steps, d=64, depth=3,
1536 generated windows — i.e. a deliberately *small* run; numbers improve with the full
config.

| Diagnostic | Real | Tail-aware FM | Vanilla FM |
|---|---|---|---|
| Hill index A (lower) | 3.47 | **3.96** | 8.75 (≈2.5× too light) |
| λ_L(0.02) pair (A,B) | 0.494 | **0.258** | 0.111 |
| Marginal CVaR₉₉.₅ A | 0.0908 | **0.0826** | 0.0586 (−35%) |
| Marginal CVaR₉₉.₅ C | 0.0748 | **0.0717** | 0.0443 (−41%) |
| Portfolio VaR₉₅ (truth 0.0508) | — | **0.0516** [0.047, 0.056] | 0.0670 |
| Portfolio CVaR₉₉.₅ (truth 0.1349) | — | **0.1602** [0.123, 0.206] (CI covers) | 0.1149 |
| Kupiec p-values (95/99/99.5) | — | **0.07 / 0.31 / 0.64** (all pass) | 0.000 / 0.51 / 0.66 |

The vanilla failure is the predicted one: *light tails, fat shoulders* — it
over-generates moderate losses (hence VaR₉₅ too conservative, Kupiec rejection) and
under-generates extremes (CVaR underestimated by 35–41% at 99.5%). See
`outputs/tail_diagnostics.png`: the vanilla QQ plot saturates in the deep tail and its
loss survival function collapses, while the tail-aware model tracks the real one.

Honest caveats from this run, to watch in yours:
* At 600 steps the tail-aware model has not fully learned to *suppress* base tail
  dependence on independent pairs (λ ≈ 0.12 vs real 0.06) nor to fully *amplify* it on
  the dependent pair (0.26 vs 0.49). Both move in the right direction with training;
  `mix_dim="time"` in `sample_base` is the knob if your data has cross-sectional but not
  temporal tail clustering.
* The pooled within-window squared-ACF is a weak diagnostic at n=24 after per-window
  normalization; use longer windows or an unnormalized statistic for volatility
  clustering.

## Usage

```python
import torch
from tailfm import (MarginalEnsemble, VelocityField, train_cfm, sample,
                    estimate_risk, kupiec_test, make_windows)

real = make_windows(returns, n=64, stride=1)            # (N, n, f) log-returns

marg = MarginalEnsemble(q_tail=0.05, nu="auto").fit(real)
z    = torch.tensor(marg.transform(real), dtype=torch.float32)

model  = VelocityField(f=real.shape[-1], n_max=64, d_model=128, depth=4)
ema, _ = train_cfm(model, z, nu=marg.nu_, steps=20_000, device="cuda")

z_gen = sample(ema.shadow, num=50_000, n=64, f=real.shape[-1],
               nu=marg.nu_, n_steps=100, device="cuda")
gen   = marg.inverse_transform(z_gen.numpy())

report = estimate_risk(gen, alphas=(0.95, 0.99, 0.995), weights=w, horizon=10)
# validate: kupiec_test(held_out_losses, report[0.99]["var_gpd"], 0.99)
```

End-to-end demo with ablation and figures: `python example.py [--quick] [--ablation]`.
`stages.py` is the same demo split into checkpointed stages (useful under wall-clock
limits).

## Scaling to real data

* Real config: `d_model=128–256`, `depth=4–8`, `steps ≥ 20k`, `M ≥ 50k` generated
  windows for 99.5% statistics; GPU strongly recommended.
* Fit `MarginalEnsemble` on the **training period only**; regime shifts move thresholds
  and ξ — refit on a rolling basis if used in production.
* For 1-step *single-feature* risk, `SemiParametricMarginal.var_es` alone is sufficient
  (no generation needed); the generative model earns its keep for **portfolio** and
  **multi-step** CVaR, where joint tails and temporal aggregation matter.

## Extensions (in decreasing priority)

1. **Fissler–Ziegel fine-tuning.** Distill/reflow to a few-step generator, then add a
   tail penalty `Σ_α (VaR̂_gen − VaR̂_real)² + (EŜ_gen − EŜ_real)²` (pinball VaR,
   Rockafellar–Uryasev CVaR, both differentiable) plus smoothed joint-exceedance
   frequencies — the joint elicitability of (VaR, ES) makes this a proper scoring
   objective (Fissler & Ziegel 2016; cf. Tail-GAN, Cont et al.).
2. **Conditioning** on the recent past (encode a context window into the adaLN signal)
   turns this into a *conditional* scenario generator for dynamic VaR backtesting
   (Christoffersen independence test then becomes applicable).
3. **Per-group mixing**: one W per learned feature cluster if only subsets co-crash.

## Baselines

`baselines/` contains three published generative baselines, wired to consume the same
`(N, n, f)` windows and the same evaluation stack (`print_report`, `estimate_risk`,
`kupiec_test`), so their numbers are directly comparable with tailfm:

| Baseline | Reference | Relation to the reference code |
|---|---|---|
| `baselines/timevae.py` | Desai et al. 2021, *TimeVAE* | Faithful PyTorch port of the TF2/Keras reference (`timeVAE-main/src/vae`): same conv encoder / level+trend+residual decoder, same summed reconstruction+KL loss with `reconstruction_wt=3`, same Adam(1e-3)/batch-16/early-stopping schedule, same per-feature min-max scaling. Custom-seasonality layer omitted (disabled in the reference default config). |
| `baselines/timegan.py` | Yoon et al., NeurIPS 2019, *TimeGAN* | Faithful PyTorch port of the TF1 reference (`TimeGAN-master/timegan.py`) — the original uses `tf.contrib` and cannot run on Python ≥ 3.8. Same five GRU networks, identical loss formulas and weights (10·√E, 100·√S, 100·V, γ=1, D-update gate at 0.15), same three-phase schedule, U[0,1] noise, min-max scaling. |
| `baselines/tailgan.py` | Cont, Cucuringu, Xu, Zhang 2022, *Tail-GAN* | Adaptation of the PyTorch reference (`Tail-GAN-main`): same MLP generator with [-1,1] clamp, NeuralSort + projected (VaR,ES) discriminator, `S_quant` joint elicitability score, buy-and-hold/static-portfolio/mean-reversion/trend-following PnLs with [31,69]-percentile thresholds. The precomputed portfolio matrix and threshold files (not shipped in the repo) are regenerated in memory from the training windows; one GAN is trained instead of the 10-model screened ensemble. |

Run all three on your CSV (accepts `returns.csv` or `returns_prices.csv --prices`):

```bash
python run_baselines.py --data returns.csv --n 24 --gen 50000
python run_baselines.py --data returns.csv --quick          # CPU smoke test only
python run_baselines.py --data returns.csv --n 24 --reuse \
    --tailfm-gen run_out/generated_windows.npy   # re-evaluate saved gen_<model>.npy
                                                 # files without retraining
```

To include tailfm in the final comparison table, first run `fit_returns.py` with the
same `--data/--n/--test-frac/--horizon/--seed`, then pass its samples:

```bash
python fit_returns.py   --data returns.csv --n 24 --outdir run_out
python run_baselines.py --data returns.csv --n 24 --tailfm-gen run_out/generated_windows.npy
```

Outputs per model: `gen_<model>.npy`, the full tail-diagnostic report, portfolio
VaR/CVaR with bootstrap CIs, a Kupiec backtest on the held-out period, and a
cross-model comparison table.

Every line printed by `fit_returns.py`, `run_baselines.py` and
`backtest_portfolios.py` is mirrored into a text file (`{outdir}/report.log`,
`{gen_dir}/backtest_report.log`; override with `--log`), preceded by a banner with
the timestamp and the exact command line. The figures are written one PNG per
diagnostic, with every model overlaid inside each (`figures.py`):

| File | Content |
|---|---|
| `qq_lower_tail.png` | lower-tail QQ, one panel per feature + one with features pooled |
| `tail_dependence.png` | `lambda_L(q)` for **every** feature pair, one panel per pair |
| `portfolio_loss_survival.png` | `P(L > l)` of the h-step portfolio loss |
| `empirical_distributions.png` | per feature: pooled log-density and 1-step loss survival |

Practical notes:
* Training budgets: reference defaults are `--timevae-epochs 1000` (early-stopped),
  TimeGAN `iterations=50000` *per phase* (runner default `--timegan-iters 10000`; use
  the full 50k on GPU for paper-grade numbers), Tail-GAN `--tailgan-epochs 3000` with
  batch 1000 and lr_G=1e-6 / lr_D=1e-7 as in the paper.
* The Tail-GAN discriminator's first linear layer has `in_features = batch_size`
  (it maps the sorted PnL sample to (VaR, ES)), so batches are dropped to a fixed
  size; NeuralSort memory scales as O(batch²) per strategy.
* Expected pathologies to look for in the diagnostics — TimeVAE/TimeGAN squash tails
  (Gaussian latent + sigmoid/[0,1] output ⇒ Hill indices far too large, CVaR
  underestimated); Tail-GAN targets (VaR, ES) at its training α (default 0.05) but its
  hard clamp caps extremes at the training max, and none of the three carries an EVT
  marginal guarantee — which is precisely the comparison tailfm is designed to win.
* **TimeVAE on log-return windows collapses to near-deterministic output** (marginal
  VaR flat at ~4e-4, λ_L = 0). This is a property of the model, not the port: under the
  MSE reconstruction loss the deterministic decoder converges to E[x|z], so sample
  variance equals only the variance *explained by the latent* — and return windows,
  unlike the smooth price/sine/energy sequences in the TimeVAE paper, are nearly
  incompressible, so the KL term drives posterior collapse and prior samples degenerate
  to ~the mean window. (The port reproduces healthy dispersion on the reference's own
  `stockv` price dataset: generated per-feature std 8.0 vs real 8.5 at default
  hyperparameters.) Mitigation via the reference's own knobs: `--timevae-recon-wt 100`
  and/or a larger `--timevae-latent` reduce (but do not remove) the under-dispersion —
  a fair line for the thesis is that reconstruction-based VAEs structurally
  under-disperse on returns, before tails are even considered.
* `tailfm/risk.py::var_cvar_gpd` now falls back to the empirical VaR/CVaR when the POT
  fit is not identified (< 10 strict exceedances above the threshold, as happens for
  massively tied losses from a collapsed generator, or a failing GPD MLE), instead of
  crashing inside `scipy.stats.genpareto.fit`.
* Baselines are evaluated **as generated** (no rank-recalibration), since the EVT
  marginal map is part of the tailfm model, not a shared post-processing step.

## Files

```
tailfm/evt.py       EVT marginals: GPD tails, t_ν PIT, closed-form VaR/ES, Hill
tailfm/base.py      elliptical Student-t source (shared χ²_ν mixing)
tailfm/model.py     DiT-style temporal transformer velocity field
tailfm/cfm.py       CFM training (OT path, EMA) + Heun sampler
tailfm/risk.py      portfolio losses, empirical & GPD-refined VaR/CVaR, Kupiec
tailfm/evaluate.py  Hill tables, tail-dependence curves, risk tables, ACF
tailfm/data.py      windowing + synthetic market with known tail structure
example.py          end-to-end demo (+ vanilla ablation, figures)
stages.py           checkpointed variant of the demo
fit_returns.py      fit tailfm on your own CSV/npy return series
figures.py          diagnostic figures shared by the runners (one PNG per diagnostic)
run_logging.py      tee: mirrors everything printed into {outdir}/report.log
dataio.py           compat shim: re-exports fit_returns.load_returns (old layout)
diagnostic_csv.py      locate values that break np.log() in a price CSV
diagnostic_timevae.py  TimeVAE collapse: prior hole vs incompressible data
baselines/common.py    scalers matching each reference's convention
baselines/timevae.py   TimeVAE baseline (PyTorch port of the TF2 reference)
baselines/timegan.py   TimeGAN baseline (PyTorch port of the TF1 reference)
baselines/tailgan.py   Tail-GAN baseline (adapted PyTorch reference)
run_baselines.py       train + evaluate baselines on your CSV, comparison table
```

Dependencies: `torch`, `numpy`, `scipy`, `matplotlib`.

# tailfm — Tail-aware Flow Matching for Multivariate Time Series

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
>
> ```bash
>TAG=$(date +%Y%m%d)-seed0
>mkdir -p results/$TAG/tailfm results/$TAG/baselines
>cp run_out/report.log run_out/*.png                 results/$TAG/tailfm/
>cp baseline_out/report.log baseline_out/*.png       results/$TAG/baselines/
> cp baseline_out/backtest_report.log                 results/$TAG/baselines/ 2>/dev/null
> du -sh results/$TAG        # sanity: should be ~1 MB, not 30

```

```bash
git add -A                 # your code changes
git add -f results/        # the artifacts, overriding .gitignore
git status                 # verify: no .npy, no .pt, no .pkl
git commit -m "Add tailfm integration + first run results"
git push                   # or: git push -u origin HEAD
```


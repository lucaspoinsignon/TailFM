# tailfm — Tail-aware Flow Matching for Multivariate Time Series
```bash
export PATH=/home/ubuntu/tails/bin:$PATH; hash -r; python -c "import numpy,pandas,scipy,matplotlib,torch" 2>/dev/null || { /domino/datasets/local/Quail/envs/tails/bin/python -m venv /home/ubuntu/tails; hash -r; python -m pip install --no-cache-dir numpy pandas scipy matplotlib torch; }; python -V && python -c "import torch;print('ready')"

python prepare_valor_csv.py --raw prices_raw.csv --out returns_all.csv --start 2019-12-23 --end 2026-01-05 --max-zero-frac 0.99 --ffill-limit 3

python -c "
import pandas as pd
r = pd.read_csv('returns_all.csv'); q=0.05
lo=(r<r.quantile(q)).sum(); hi=(r>r.quantile(1-q)).sum()
bad=list(r.columns[(lo<30)|(hi<30)])
print(f'dropping {len(bad)} EVT-infeasible ->', end=' ')
r.drop(columns=bad).to_csv('returns_all.csv', index=False, float_format='%.8e')
print(pd.read_csv('returns_all.csv').shape)
"

python fit_returns.py --data returns_all.csv --n 24 --test-frac 0.2 --horizon 10 --seed 0 --steps 20000 --gen 50000 --outdir run_all


```

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
> ```


```bash
git add -A                 # your code changes
git add -f results/        # the artifacts, overriding .gitignore
git status                 # verify: no .npy, no .pt, no .pkl
git commit -m "Add tailfm integration + first run results"
git push                   # or: git push -u origin HEAD
```
to get the data first 
```bash
pip install pandas   # not in environment.yml, only download_data.py used it

python prepare_valor_csv.py --raw prices_raw.csv --out returns_valor.csv \
       --start 2019-12-23 --end 2026-01-05

python fit_returns.py --data returns_valor.csv --n 24 --test-frac 0.2 \
       --horizon 10 --seed 0 --steps 20000 --gen 50000 --outdir run_out
```
see what time series is bad
```bash
python -c "
import pandas as pd
r = pd.read_csv('returns_valor.csv'); q=0.05
d = pd.DataFrame({'n_lo':(r<r.quantile(q)).sum(),'n_hi':(r>r.quantile(1-q)).sum(),
                  'zero_frac':(r==0).mean(),'min':r.min(),'std':r.std()})
print(d[(d.n_lo<30)|(d.n_hi<30)].sort_values('n_lo').to_string(float_format='%.4g'))
"
```
```bash
python -c "
import pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
p = pd.read_csv('returns_valor_prices.csv', index_col=0, parse_dates=True)['V4157089']
r = pd.read_csv('returns_valor.csv')['V4157089'].values
q = 0.05
print(p.describe(), sep='\n')
print(f'n_lo={(r<pd.Series(r).quantile(q)).sum()}  zero_frac={(r==0).mean():.3f}  min={r.min():.3e}  n_uniq={pd.Series(r).nunique()}')
fig, ax = plt.subplots(3, 1, figsize=(11, 9))
ax[0].plot(p.index, p.values, lw=.8); ax[0].set_title('V4157089 price')
ax[1].plot(p.index[1:], r, lw=.5);     ax[1].set_title('log returns')
ax[2].hist(r, bins=200);               ax[2].set_yscale('log'); ax[2].set_title('return histogram (log count)')
fig.tight_layout(); fig.savefig('V4157089.png', dpi=110)
print('wrote V4157089.png')
"
python -c "
import pandas as pd
r = pd.read_csv('returns_valor.csv'); q=0.05
lo=(r<r.quantile(q)).sum(); hi=(r>r.quantile(1-q)).sum()
bad=list(r.columns[(lo<30)|(hi<30)])
print(f'dropping {len(bad)}:', bad)
r.drop(columns=bad).to_csv('returns_valor.csv', index=False, float_format='%.8e')
print(pd.read_csv('returns_valor.csv').shape)
"
```
```bash
export ENV=/domino/datasets/local/Quail/envs/tails
export PATH=$ENV/bin:$PATH
hash -r
python -V
python -c "import sys; assert sys.version_info[:2]>=(3,10), sys.version; print('OK', sys.executable)"
python -m pip install --upgrade pip
python -m pip install torch numpy scipy matplotlib pandas
python -c "import pandas,numpy,scipy,matplotlib,torch;print('imports OK')"
ls -la prices_raw.csv
python prepare_valor_csv.py --raw prices_raw.csv --out returns_valor.csv --start 2019-12-23 --end 2026-01-05
```


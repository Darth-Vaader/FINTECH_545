# Assignment 3 — Options, Option Risk, and Portfolio Construction

FinTech 545 — Quantitative Risk Management

## Contents

| file | what it is |
|---|---|
| `Assignment3_Writeup.pdf` | **The write-up (submission document).** All five problems: each sub-question restated, its code output, and the Predict / Fit / Reconcile answers. Exported from the notebook below. |
| `Assignment3.ipynb` | The same write-up as an executed notebook: one labeled code cell (`# Problem k(x)`) per sub-question, calling the modules below, with all output and figures embedded. |
| `common.py` | Shared code, **all written from scratch**: generalized Black–Scholes–Merton value and closed-form Greeks, implied volatility (Brent root finding), the CRR binomial tree for American options, the non-recombining discrete-dividend tree, moments, AICc, Normal / Student-t fits, VaR/ES, OLS, plot style. |
| `problem1.py` … `problem5.py` | One module per problem. Each is runnable on its own and prints every number quoted in the write-up. |
| `export_pdf.py` | Rebuilds `Assignment3_Writeup.pdf` from the notebook (nbconvert → HTML → headless Chrome; no LaTeX needed). |
| `figures/` | PNGs written by the modules (also embedded in the notebook and PDF). |
| `problem1.csv`, `problem3.csv`, `problem3_portfolios.csv`, `problem4.csv`, `problem5.csv`, `F-F_Research_Data_Factors_daily.CSV`, `F-F_Momentum_Factor_daily.CSV` | The supplied data, copied unchanged from the class repository. `Assignment 3.pdf` is the problem statement. |

## Requirements

Python 3.10+ with

```
numpy  pandas  scipy  matplotlib
```

Verified on Python 3.13 with numpy 2.3, pandas 2.3, scipy 1.16, matplotlib 3.10.

```bash
pip install numpy pandas scipy matplotlib
# to re-run the notebook and rebuild the PDF, also:
pip install jupyter nbconvert nbclient nbformat
```

## Reproducing every number

From this folder:

```bash
python problem1.py     # ~2 s
python problem2.py     # ~60 s (the N = 1000 dividend tree in the convergence table is ~30 s per option)
python problem3.py     # ~2 s  (2 x 100,000 ten-day paths)
python problem4.py     # ~12 s (2,000 bootstrap re-optimisations)
python problem5.py     # ~1 s
```

Each prints its full results to stdout and writes its figures to `figures/`. The modules
are independent and can be run in any order. All paths are resolved relative to the
module file, so they can be run from any working directory.

| problem | command | produces |
|---|---|---|
| 1 — IV from a dirty chain | `python problem1.py` | parity gap and band at every strike, butterfly check, vega/spread preview, bid/mid/ask IVs with the reason any is missing, vega and dIV per cent, the K=175 gap in vol points, the call-wing table, the clean smile, `figures/problem1_smile.png` |
| 2 — American options, discrete dividends | `python problem2.py` | Hull's condition per dividend, the eight prices plus escrowed GBSM, convergence in N (incl. 100/105 and 1000), delta/gamma at bumps 1…0.001, dV/dD per dividend, early exercise premia, the critical stock price before D2, the gamma diagnosis, `figures/problem2_convergence.png`, `figures/problem2_greeks.png` |
| 3 — VaR of an option book | `python problem3.py` | position/book Greeks, no-move P&L, moments and Normal/t fits with AICc, delta-normal VaR/ES, 100,000-path full-revaluation VaR/ES under both fits, Taylor approximations on the same paths, the mean-P&L decomposition, t/Normal ratios, `figures/problem3_returns.png`, `figures/problem3_pnl.png` |
| 4 — a Sharpe ratio that sells insurance | `python problem4.py` | moments and Sharpe ratios, leave-worst-month-out, the three tangency portfolios, historical vs Normal VaR/ES, CAPM all months and EQ < −5% months, standard errors, ±1 se weight shocks, bootstrap of the weights, `figures/problem4_predict.png`, `figures/problem4_frontier.png`, `figures/problem4_boot.png` |
| 5 — alpha or factor exposure | `python problem5.py` | summary and CAPM, residual–factor correlations, CAPM/FF3/Carhart with se, t, R², AICc, expected returns from the recent 10-year premia, the worst momentum day on \$10m, the exact alpha decomposition, `figures/problem5.png` |

### Re-running the notebook

The notebook ships executed. To re-run it (about 2 minutes):

```bash
jupyter nbconvert --to notebook --execute --inplace Assignment3.ipynb
```

or open it in Jupyter and use *Kernel → Restart & Run All*.

### Rebuilding the PDF

```bash
python export_pdf.py       # writes Assignment3_Writeup.pdf; needs Chrome or Chromium
```

Without Chrome: *File → Save and Export Notebook As → PDF* in JupyterLab (needs TeX), or
export to HTML and print from a browser.

## Validation of the pricing code

The pricers in `common.py` reproduce the worked examples in the course notes:

| check | course value | `common.py` |
|---|---|---|
| 2-step American put, S=X=100, T=0.5, r=8%, σ=30% (Week 07) | 6.4218 | 6.4218 |
| same put, converged tree | 6.8662 | 6.8658 (N=2000) |
| same put, European GBSM | 6.4670 | 6.4671 |
| BSM call S=X=100, T=1, r=5%, σ=10% (Week 06) | 6.8050 | 6.8050 |
| European tree N=1000 for that call | 6.8039 | 6.8039 |
| implied vol of a \$6 call, 125/255 years (Week 06) | ≈ 17% | 17.01% |

The vectorised dividend tree `common.bt_div` returns values identical (to machine
precision) to the direct recursive port of `bt_american.jl`, `common.bt_american_div`,
for all four option types; the vectorised one is used because it makes N = 500 take one
second instead of minutes.

## Conventions chosen, and why

1. **Returns and signs.** Arithmetic returns $P_t/P_{t-1} - 1$; VaR and ES are positive
   numbers for a loss (assignment instructions, Week 04 §4.2).
2. **Empirical VaR/ES** follow the class `library/RiskStats.jl` exactly: the α-quantile is the
   average of the ⌊Nα⌋-th and ⌈Nα⌉-th order statistics; ES is the mean of all values at or
   below it. (For 240 months at 1% this averages the 2nd and 3rd worst months.)
3. **GBSM.** $b = r - q$ for Problem 1 (continuous yield), $b = r$ elsewhere. Theta from the
   closed form is $-\partial V/\partial T$ per year; per trading day it is divided by 255.
4. **Problem 1 time** is 35 calendar days / 365, as instructed. "Implied vol from the bid"
   uses the same root finder; a quote with no root is reported with its reason (price below
   the σ → 0 limit, i.e. the discounted forward intrinsic value).
5. **Problem 2 tree.** CRR, $u = e^{\sigma\sqrt{\Delta t}}$, $d = 1/u$; with dividends the stock
   grows at $r$ between payments and the dividend is subtracted on the dividend step;
   exercise on a dividend step pays the cum-dividend payoff (Week 07 / `bt_american.jl`).
   The European-with-dividends value is the same tree with exercise switched off; the
   comparison GBSM is the escrowed model on $S - \sum D_i e^{-rt_i}$. Headline N = 500.
   Early exercise premia are measured with both legs on the same tree so that the tree's
   own error cancels. The (h) figure uses N = 100 because hundreds of starting prices on the
   N = 500 dividend tree need gigabytes of memory; the piecewise-linear argument does not
   depend on N.
6. **Problem 3.** Delta-normal uses the sample sd (n − 1) and zero mean, scaled by √10,
   on the dollar delta, as in the Week 07 homework. Simulation: 100,000 paths per fitted
   distribution, location set to zero, ten daily returns compounded, full GBSM revaluation
   with T − 10/255. Normal and t paths use separate seeds (545, 546).
7. **Problem 4.** $r_{f,m} = 3\%/12$; annualised Sharpe = √12 × monthly. Sample means and
   covariance (n − 1). Unconstrained tangency in closed form; constrained ones by SLSQP with
   several starting points (the role Ipopt plays in `optimizers.jl`). Bootstrap: 2,000
   resamples of the 240 months with replacement, seed 545.
8. **Problem 5.** Fama–French and momentum divided by 100; inner join on dates; OLS with
   classical standard errors; AICc counts the coefficients plus the error variance;
   annualisation × 255 (course convention). "Most recent 10 years" = the 10 years ending at
   the last date in the factor files (2014-10-01 … 2024-09-30).
9. **Seeds.** Every Monte Carlo step is seeded from `common.SEED = 545`, so all output is
   identical run to run.

# Assignment 2 — Covariance, VaR, and Copulas

FinTech 545 — Quantitative Risk Management

## Contents

| file | what it is |
|---|---|
| `Assignment2_Writeup.pdf` | **The write-up (submission document).** All five problems: predictions, fits, reconciliations, figures. Exported from the notebook below. |
| `Assignment2.ipynb` | The same write-up as an executed notebook: markdown for the reasoning, one code cell per stage calling the modules below, all output and figures embedded. |
| `common.py` | Shared helpers: moments, AICc/BIC, VaR and ES, EW variance, Rebonato–Jackel and Higham repairs, PSD-tolerant Cholesky, multivariate-normal simulation, plot style. |
| `problem1.py` … `problem5.py` | One module per problem. Each is runnable on its own and prints every number quoted in the write-up. |
| `export_pdf.py` | Rebuilds `Assignment2_Writeup.pdf` from the notebook (nbconvert → HTML → headless Chrome; no LaTeX needed). |
| `figures/` | PNGs written by the modules (also embedded in the notebook and PDF). |
| `problem1.csv` … `problem5.csv` | The supplied data, copied unchanged from the class repository (the problem statement itself lives there). |

The computation lives in the `.py` modules rather than in notebook cells, so each
problem can be run and checked independently. The notebook imports them, so no
number in the prose is typed by hand — every figure quoted in a markdown cell is
printed by the code cell below it.

## Requirements

Python 3.9+ and:

```
numpy  pandas  scipy  matplotlib
```

Verified on Python 3.13.9 with numpy 2.3.5, pandas 2.3.3, scipy 1.16.3,
matplotlib 3.10.6.

```bash
pip install numpy pandas scipy matplotlib
# to re-run the notebook itself, also:
pip install jupyter nbconvert nbclient nbformat
```

## Reproducing every number

### Everything at once

```bash
python problem1.py
python problem2.py
python problem3.py
python problem4.py
python problem5.py
```

Each prints its full results to stdout and writes its figures to `figures/`.
Nothing depends on anything else, so they can be run in any order. Problem 4 takes
about 10 seconds (two 100,000-day copula simulations and a 200-point profile
likelihood); the others run in a second or two.

### Per problem

| problem | command | produces |
|---|---|---|
| 1 — mismatched histories | `python problem1.py` | joint-observation table, eigenvalue bound, complete-case and pairwise correlation matrices with eigenvalues and Cholesky attempts, tracking-portfolio variance, Rebonato–Jackel and Higham repairs, entry-by-entry repair table, `figures/problem1.png` |
| 2 — volatility after a regime change | `python problem2.py` | variance break, $n_{\text{eff}}$ / half life / regime weights, moments, the five VaRs and fitted $t$, the VaR crossing point, within-regime sds and mixture kurtosis, EW standard errors, `figures/problem2_predict.png`, `figures/problem2_fit.png` |
| 3 — diversification and VaR | `python problem3.py` | moments, default counts, count-based prediction, historical VaR/ES at 5% and 1%, Normal VaR, subadditivity check, `figures/problem3_predict.png`, `figures/problem3.png` |
| 4 — Gaussian or $t$ copula | `python problem4.py` | moments with leave-one-out, joint tail counts against independence / Gaussian / $t$, margin fits by AICc, $R$ from Kendall's $\tau$, both copula fits with AICc and BIC, 100,000-day simulations with VaR/ES, implied tail counts, per-day likelihood contributions, tail dependence, `figures/problem4_predict.png`, `figures/problem4_fit.png` |
| 5 — model-based simulation | `python problem5.py` | regressions, residual covariance, 100,000-day simulations with full and diagonal residual covariance, delta-normal VaR, the OLS covariance identity, `figures/problem5.png` |

### Re-running the notebook

The notebook ships already executed — every cell has its output and all eight
figures are embedded, so it can be read without running anything. To re-run it:

```bash
jupyter notebook Assignment2.ipynb     # then Kernel > Restart & Run All
```

or headless:

```bash
jupyter nbconvert --to notebook --execute --inplace Assignment2.ipynb
```

### Rebuilding the PDF

```bash
python export_pdf.py                   # writes Assignment2_Writeup.pdf; needs Chrome or Chromium
```

This converts the executed notebook to HTML with nbconvert and prints it with
headless Chrome, so no TeX installation is needed. Two details are handled on the
way and documented in the script: nbconvert's markdown renderer drops the
backslash from escaped dollar signs (`\$19,594`) and MathJax then pairs them into
inline math, so the script doubles the escape in a temporary copy; and the modules'
88-column output is given an 11px monospace so it never wraps on a Letter page.

Without Chrome, open the notebook in JupyterLab and use **File > Save and Export
Notebook As > PDF** (needs TeX), or export to HTML and print from a browser.

## Where each definition comes from

Every definition is taken from the course notes and cited inline in the code and
the write-up:

| definition | source | as used |
|---|---|---|
| PSD: $w'\Sigma w \ge 0\ \forall w$; a variance cannot be negative | Week 02 §2.2.1 | Problem 1(a), (g) |
| Fact 7, portfolio variance as a double sum over the covariance | Week 02 §2.1.1 | Problem 1(e), Problem 5(a) |
| AIC, AICc, BIC and $k = p + d$ | Week 02 §2.5.3 | margins $k = 2$ / $3$; copulas $k = 0$ / $1$ |
| Complete-case vs pairwise estimation; pairwise need not be PSD | Week 03 §3.3 | Problem 1(a), (d) |
| Correlations from any estimator, variances from the full history: $\Sigma = DCD$ | Week 03 §3.4 item 7 | Problem 1(e) |
| EW weights $w_{t-i} = (1-\lambda)\lambda^{i-1}$, normalised | Week 03 §3.5.1 | Problem 2 |
| $n_{\text{eff}} = (1+\lambda)/(1-\lambda)$, half life $\ln 0.5/\ln\lambda$, se $\approx \sigma/\sqrt{2n_{\text{eff}}}$ | Week 03 §3.5.3 | Problem 2(b), (g) |
| Responsiveness vs stability in the choice of $\lambda$ | Week 03 §3.5.4 | Problem 2(g) |
| Rebonato–Jackel eigenvalue clip and rescale | Week 03 §3.6.1 | `common.near_psd` |
| Higham alternating projections with Dykstra correction, $W = I$ | Week 03 §3.6.2 | `common.higham_nearest_corr` |
| Cholesky for PSD matrices (zero column on a zero pivot) | Week 03 §3.2.2 | `common.chol_psd` |
| Multivariate normal draw $X = LZ$ | Week 03 §3.1.2 | `common.simulate_normal`, Problem 5(c) |
| Arithmetic returns; portfolio P&L as holdings × returns | Week 04 §4.1.2 | Problems 2, 3, 4, 5 |
| $\mathrm{VaR}_\alpha = -F^{-1}(\alpha)$, positive for a loss; absolute vs relative | Week 04 §4.2, §4.2.1 | `common.var_es`, `common.var_normal` |
| Subadditivity and the two-bond counterexample | Week 04 §4.3 | Problem 3 |
| Delta-normal VaR, $-\Phi^{-1}(\alpha)\sqrt{\nabla R'\Sigma\nabla R}$ | Week 04 §4.4 | Problem 5(d) |
| Historical VaR: sort, take the $N\alpha$-th value | Week 04 §4.5–4.6 | `common.var_es` |
| ES as the mean beyond VaR; Normal ES; coherence | Week 05 §5.1 | `common.var_es`, `common.es_normal`, Problem 3 |
| Model-based simulation; $\Sigma_\epsilon$ is "the part that is easy to forget"; OLS is already inside the joint normal | Week 05 §5.2 | Problem 5 |
| Copula fitting recipe; Spearman shortcut; tail dependence of the Gaussian is 0 | Week 05 §5.4 | Problem 4 |
| Multivariate $t$ density; mixture representation; Kendall's $\tau$ with $\rho = \sin(\pi\tau/2)$; profiling $\nu$ on $\theta = 1/\nu$ | Week 05 §5.5 | `problem4.mvt_logpdf`, `profile_nu`, `simulate` |
| $t$ copula density as a difference of logs; $k_{\text{Gauss}} = 0$, $k_t = 1$; $\Delta\mathrm{BIC} = 2(\ell_t - \ell_G) - \ln m$; tail dependence $\lambda = 2t_{\nu+1}(\cdot)$ | Week 05 §5.6 | Problem 4(d), (g), (j) |

### Conventions chosen here, and why

1. **VaR and ES sign.** Both are positive for a loss, following the leading minus
   in Week 04 §4.2 and Week 05 §5.1. A negative VaR is a profit at the
   $\alpha$-quantile and is reported as such (Problem 3).
2. **Empirical quantile.** `common.var_es` sorts the $N$ outcomes and takes the
   $N\alpha$-th as the quantile (Week 04 §4.5), with ES the mean of those $N\alpha$
   worst outcomes. $N\alpha$ is an integer for every case here (500 × 0.05,
   10,000 × 0.05 and 0.01, 1,000 × 0.05 and 0.01, 100,000 × 0.05 and 0.01).
   Problem 2 also prints numpy's interpolating quantile; the two differ by \$60 on a
   \$20,574 VaR.
3. **Exact normal quantile.** $\Phi^{-1}(0.05) = -1.6449$ rather than the slides'
   rounded $1.65$, so closed-form and simulated numbers can be compared to four
   digits (Problem 5 checks them against each other).
4. **Mean removal in Problem 2.** The sample mean is removed once with equal
   weights, as instructed, and the EW variance is then $\sum w_i x_i^2$. The
   alternative — removing the EW-weighted mean — is also printed; it changes the EW
   sd in the fourth significant figure.
5. **Variance divisor.** $n - 1$ everywhere (Week 01 §4.2), *including* the
   residual covariance in Problem 5, so that the in-sample identity
   $\Sigma_{\text{sample}} = \beta\beta'\sigma_M^2 + \Sigma_\epsilon$ holds exactly;
   the OLS $n - 2$ residual sd is printed alongside. Normal margins in Problem 4 use
   the MLE ($n$) for the likelihood.
6. **Pairwise correlation.** Each cell uses the rows where both series are present,
   with means and sds computed on those same rows. The full-history sd of each
   series is used for the covariance (Week 03 §3.4 item 7).
7. **Higham tolerances.** $W = I$, stop when the change in $\|Y_k - Y_0\|_F$ is below
   $10^{-9}$ *and* the smallest eigenvalue exceeds $-10^{-9}$ (Week 03 §3.6.3:
   "check the eigenvalue, not the norm"). Both repairs floor the bad eigenvalue at
   zero, so the repaired matrices are PSD but singular and need the PSD-tolerant
   Cholesky; the code shows numpy's strict Cholesky still failing on them.
8. **Student's $t$ fits.** Location–scale $t$, $x = \mu + sT_\nu$, with $s$ a *scale*
   (sd $= s\sqrt{\nu/(\nu-2)}$), $\nu \in [2.0001, 100]$. The likelihood is flat in
   $\nu$, so `common.fit_t` runs L-BFGS-B, a Nelder–Mead polish and scipy's `t.fit`
   and keeps the best; a single gradient run on Problem 2's data stopped at
   $\nu = 6.8$, $0.23$ log-likelihood units short of the true optimum at $8.27$.
9. **Copula details.** Ranks scaled as $\text{rank}/(n+1)$; $U$ through the
   AICc-chosen margin; one $R$ from Kendall's $\tau$ for both copulas; $\nu$ profiled
   on a 200-point grid in $\theta = 1/\nu$ then refined by a bounded scalar search; the
   $t$ copula simulated through the chi-square mixture with the *copula's* $\hat\nu$ in
   the uniform step; both simulations share the same standard normal block. The
   bivariate $t$ copula tail probabilities used as benchmarks in Problem 4(b) are
   computed by one-dimensional quadrature over the mixing variable, because scipy's
   `multivariate_t.cdf` is quasi-Monte Carlo and moves in the third digit between
   calls.
10. **"Zero expected returns" in Problem 5** means the simulation drops $\alpha$
    and gives the market a zero mean; $\alpha$ is still reported from the fit.
11. **"Loses more than 20%" in Problem 3** is taken literally as the default
    indicator. `A` has no scenario between $-21.9\%$ and $+6.9\%$, so any cut in
    that gap gives the same count; `B` has one scenario at $-19.6\%$ (a
    high-recovery default) that the 20% cut leaves out, so its count is 393 rather
    than 394. The code prints this; no VaR or ES depends on the cut.

## Notes on results

- Every Monte Carlo step uses `numpy.random.default_rng(545)` (`common.SEED`), so
  all output — including the simulated VaRs — is identical run to run. Problem 4's
  output was checked to be byte-identical across two runs.
- Monte Carlo standard errors are reported next to every simulated VaR/ES
  (Problem 4: 10 blocks of 10,000; Problem 5: the asymptotic quantile formula and
  a closed-form check), so the reader can see which digits are meaningful.
- Figures are written at 150 dpi to `figures/`; the directory is created
  automatically.
- All modules resolve their CSV paths relative to their own location, so they can
  be run from any working directory.
- All output is formatted to stay within 88 characters per line so nothing wraps or
  clips in the PDF. The modules run with clean stderr.

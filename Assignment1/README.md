# Assignment 1 — Univariate and Multivariate Statistics

FinTech 545 — Quantitative Risk Management

## Contents

| file | what it is |
|---|---|
| `Assignment1.ipynb` | **The write-up.** All five problems: predictions, fits, reconciliations, figures. Exported to PDF for submission. |
| `common.py` | Shared helpers: moment estimators, AICc, plot style and colour palette. |
| `problem1.py` … `problem5.py` | One module per problem. Each is runnable on its own and prints every number quoted in the write-up. |
| `figures/` | PNGs written by the modules (also embedded in the notebook). |
| `problem1.csv` … `problem5.csv` | The supplied data. |

The computation lives in the `.py` modules rather than in notebook cells, so each
problem can be run and checked independently. The notebook imports them, so no
number in the prose is typed by hand — every figure quoted there is printed by the
code below it.

## Requirements

Python 3.9+ and:

```
numpy  pandas  scipy  statsmodels  matplotlib
```

Verified on Python 3.13.11 with numpy 2.4.2, pandas 3.0.2, scipy 1.17.1,
statsmodels 0.14.6, matplotlib 3.10.9.

```bash
pip install numpy pandas scipy statsmodels matplotlib
# to re-run the notebook itself, also:
pip install jupyter nbclient nbformat
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
Nothing depends on anything else, so they can be run in any order or individually.

### Per problem

| problem | command | produces |
|---|---|---|
| 1 — shape of a sample | `python problem1.py` | moments, moment screen, 1% exceedance counts, `figures/problem1.png` |
| 2 — non-Normal regression | `python problem2.py` | OLS / MLE-Normal / MLE-$t$ fits, AICc, quantile comparison, `figures/problem2_predict.png`, `figures/problem2_fit.png` |
| 3 — Pearson vs Spearman | `python problem3.py` | both correlation matrices, ranked gap table, cubic mechanism, `figures/problem3_pairs.png`, `figures/problem3_mechanism.png` |
| 4 — conditional distributions | `python problem4.py` | covariance blocks, conditional mean/variance, bucketed coverage, heteroskedasticity tests, `figures/problem4.png` |
| 5 — AR/MA order | `python problem5.py` | ADF, ACF/PACF table, AICc for six models, AR(2)-vs-AR(3) trade-off, `figures/problem5_identify.png`, `figures/problem5_fit.png` |

### Re-running the notebook

The notebook ships already executed — every cell has its output and all eight
figures are embedded, so it can be read or exported without running anything. To
re-run it:

```bash
jupyter notebook Assignment1.ipynb     # then Kernel > Restart & Run All
```

or headless:

```bash
pip install nbclient nbformat          # if not already present
jupyter nbconvert --to notebook --execute --inplace Assignment1.ipynb
```

### Exporting to PDF

Simplest route, and the one that needs no extra tooling: open the notebook and use
**File > Save and Export Notebook As > PDF** (JupyterLab), or **File > Print
Preview** then print to PDF (classic Notebook / browser).

The command-line routes each need a dependency that is not part of the analysis
stack above, so install whichever you prefer:

```bash
# via headless Chromium (no LaTeX needed)
pip install "nbconvert[webpdf]" && playwright install chromium
jupyter nbconvert --to webpdf Assignment1.ipynb

# or via LaTeX
jupyter nbconvert --to pdf Assignment1.ipynb        # requires a TeX install

# or export HTML and print it to PDF from a browser (works with no extra deps)
jupyter nbconvert --to html Assignment1.ipynb
```

All output is formatted to stay within 88 characters per line so that nothing
clips in a PDF: nbconvert renders stdout in a fixed-width block that does not
wrap.

## Where each definition comes from

Every definition used is taken from the course slides and cited inline in the code
and the notebook, rather than chosen here. The mapping:

| definition | source | as used |
|---|---|---|
| Standardized skewness, excess kurtosis | Week 01 §3.2 | `mu3/sigma^3`, `mu4/sigma^4 - 3`; kurtosis always **excess** (Normal = 0) |
| Variance estimator | Week 01 §4.2 | divides by `n-1` |
| Sample skew / kurtosis estimators | Week 01 §4.3 | bias-corrected forms are the headline values |
| Package-convention warning | Week 01 §4.4 | all three variants printed, not assumed |
| Moment existence | Week 01 §3.3 | why Problem 2's moment-matched `nu` is only an upper bound |
| Candidate distributions | Week 01 §5 | Normal, Lognormal, Student's t, NIG — the four families screened |
| Skew/kurtosis filter | Week 01 §6 Table 6 | Problem 1's moment screen, plus the `kurtosis >= skew^2 + 1` feasibility bound |
| Pearson, Spearman | Week 02 §1.3–1.4 | including the `y = x^3` case that Problem 3's data reproduces |
| Conditional distributions | Week 02 §2.4 | the partitioned result, and the block ordering (see below) |
| Seven OLS assumptions | Week 02 §3.1 | Problem 2 violates #7 (normality); Problem 4 violates #5 (constant variance) |
| MLE, log-likelihood | Week 02 §4 | Problems 1, 2, 5 |
| R², adjusted R² | Week 02 §5.1–5.2 | Problem 5(e)'s contrast against AICc |
| AIC, AICc, and `k = p + d` | Week 02 §5.3 | `AICc = -2*loglik + 2k + (2k^2+2k)/(n-k-1)` |
| ACF/PACF identification | Week 02 §6.1 Table 3 | Problem 5's AR(2) call |

Two points worth calling out because they are easy to get backwards:

**Block ordering in Problem 4.** Week 02 §2.4 partitions `X = [x1; x2]`, conditions
on the **second** block, and returns the distribution of the **first**. Problem 4
asks for `x2` given `x1`, so the CSV's `x1` must occupy the slides' *second* block —
the partition is `[x2; x1]`, giving `Sigma_11 = Var(x2)` (target) and
`Sigma_22 = Var(x1)` (conditioning). Reversing this still yields a number, and on a
2×2 problem a plausible-looking one, so `problem4.py` states the mapping explicitly
in its docstring.

**Parameter counts.** Week 02 §5.3 defines `k = p + d`, where `p` counts regression
parameters including the intercept and `d` counts distribution parameters fitted
during MLE. So: Normal-error regression `k = 3`; the same with t errors `k = 4`;
AR(2) with an intercept `k = 4`. §5.3 also notes that absolute AIC/AICc values mean
nothing and only differences between models on the same data are interpretable,
which is why every table reports ΔAICc.

### Implementation choices not fixed by the slides

1. **Problem 5's PACF estimator** is Yule-Walker (`method="ywm"`), which stays
   inside `[-1, 1]` at long lags. The "cuts off" call is made against the
   white-noise Bartlett band `±1.96/sqrt(n)`, since Week 02 §6.1's "cuts off after
   lag q" means "lags beyond q are indistinguishable from zero"; the widening band
   that conditions on the lower-order autocorrelations is plotted alongside for
   reading the ACF decay.
2. **Problem 2's t parameterisation:** `(y - a - b*x)/s ~ t(nu)`, so `s` is a
   *scale*, not the error sd. The implied sd is `s*sqrt(nu/(nu-2))`, which is what
   gets compared against the Normal's sigma. `nu` is bounded below at 2.01 so the
   error variance stays finite and that comparison stays meaningful.
3. **Problem 4's band** is a **prediction band for a new observation**
   (`± 1.95996 * conditional sd`), not a confidence band for the fitted line,
   because the question asks what fraction of *observations* fall inside. The
   multiplier is 1.95996, not 2.

MLE standard errors come from inverting a numerically differentiated Hessian of the
negative log-likelihood (observed information), computed explicitly in `problem2.py`
rather than taken from a library, so the source of every reported standard error is
visible.

## Notes on results

- No random number generation anywhere, so all output is deterministic and
  reproducible run to run.
- Figures are written at 150 dpi to `figures/`. The directory is created
  automatically.
- All modules resolve their CSV paths relative to their own location, so they can
  be run from any working directory.
- All five modules run with clean stderr. `common.finish` checks the active
  matplotlib backend: on an inline backend it calls `plt.show()` so figures render
  under the notebook cell, and on a headless Agg run it closes the figure instead,
  the PNG on disk being the artefact.

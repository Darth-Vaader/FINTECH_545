# Functional Tests 1.1 – 7.6

FinTech 545 — Quantitative Risk Management

Python implementation of the functional tests in the class repository
(`testfiles/Tests.xlsx`), covering covariance estimation, PSD repair, Cholesky,
Monte Carlo simulation, return calculation, and distribution fitting.

## Contents

| path | what it is |
|---|---|
| `run_tests.py` | Runs every test, writes our output to `output/`, compares against the expected files, prints PASS/FAIL. |
| `risklib/` | The library. One module per topic, see below. |
| `data/` | Inputs (`test*.csv`) and expected outputs (`testout*.csv`) copied from the class repo. |
| `output/` | Our results, one file per test, named identically to the expected file. Regenerated on every run. |

### `risklib` modules

| module | functions | tests |
|---|---|---|
| `covariance.py` | `missing_cov`, `ew_covar`, `ew_correlation`, `ew_cov_var_corr`, `cov_to_corr` | 1.1–1.4, 2.1–2.3 |
| `psd.py` | `near_psd`, `higham_nearest_psd`, `chol_psd` | 3.1–3.4, 4.1 |
| `simulation.py` | `simulate_normal`, `simulate_pca` | 5.1–5.5 |
| `returns.py` | `return_calculate` | 6.1, 6.2 |
| `fitting.py` | `fit_normal`, `fit_general_t`, `fit_regression_t`, `aicc`, `fit_nig_moments`, `fit_nig_mle` | 7.1–7.6 |

## Requirements

Python 3.9+ with

```
numpy  pandas  scipy
```

Verified on Python 3.13 / numpy 2.4.2 / pandas 3.0.2 / scipy 1.17.1, and on
Python 3.12 / numpy 1.26.4 / pandas 2.2.1 / scipy 1.12.0.

```bash
pip install numpy pandas scipy
```

## Running

From this directory:

```bash
python run_tests.py             # all 25 tests, ~3 seconds
python run_tests.py 2.1 3.3     # a subset
```

Each line reports the test, PASS/FAIL, and the largest absolute and relative
difference from the expected file. Exit code is 0 only if everything passes.

## How results are compared

| tests | tolerance | why |
|---|---|---|
| 1.x, 2.x, 3.x, 4.1, 6.x, 7.1, 7.5 | 1e-8 relative | closed-form calculations; agreement is at machine precision (~1e-15) |
| 7.2, 7.3, 7.4, 7.6 | 1e-4 relative | numerical optimisers; the instructor used Ipopt, we use scipy. Achieved agreement is ~1e-8. |
| 5.x | 5 standard errors | Monte Carlo; see below |

**Monte Carlo tests (5.x).** The expected files are themselves the sample
covariance of 100,000 random draws, so they cannot be matched digit for digit.
The standard error of a sample-covariance entry from \(N\) normal draws is
\(\sqrt{(\sigma_{ii}\sigma_{jj} + \sigma_{ij}^2)/N}\). Each entry of our result
must lie within 5 standard errors of the expected file (times \(\sqrt{2}\), since
both are independent simulations) and within 5 standard errors of the covariance
the simulation was meant to reproduce: the input for 5.1/5.2, the `near_psd` or
Higham repaired matrix for 5.3/5.4. Observed distances are 1–2 standard errors.
A relative tolerance would be the wrong yardstick here: the (x1, x2) entry in
`test5_3.csv` is set to exactly 0, the repair lifts it to ~0.008, and the
relative error of a near-zero Monte Carlo estimate is unbounded however good the
draw.

## Conventions and choices

Where the one-line test description leaves a convention open, the choice below
follows the instructor's `testfiles/test_setup.jl` and the `library/*.jl` it
calls, so that the expected output is reproducible rather than approximately
matched.

- **Sample variance / covariance** divide by \(n-1\) (Week 01 §4.2).
- **Missing data, pairwise (1.3, 1.4).** Each cell \((i,j)\) uses only the rows
  where both \(i\) and \(j\) are present; the means and standard deviations in
  that cell are computed on those same rows. The result need not be PSD, which
  is why tests 3.x take these matrices as input.
- **EW weights (2.x).** \(w_t = (1-\lambda)\lambda^{m-t}\), normalised to sum to
  one; the last row is the most recent observation and carries weight
  \((1-\lambda)\). The mean removed is the weighted mean. Test 2.3 takes the
  variances from \(\lambda = 0.97\) and the correlations from \(\lambda = 0.94\).
- **`near_psd` / Higham (3.x).** Both convert a covariance to a correlation,
  repair, and rescale back, so the input variances are preserved exactly.
  Higham uses \(W = I\), tolerance 1e-9, at most 100 iterations, and stops when
  the change in \(\|Y_k - Y_0\|_F^2\) is below tolerance and the smallest
  eigenvalue exceeds \(-10^{-9}\).
- **`chol_psd` (4.1).** A diagonal residual in \([-10^{-8}, 0]\) is treated as an
  exact zero eigenvalue and the rest of that column is zeroed. Anything more
  negative raises, which is how `simulate_normal` detects a non-PSD input.
- **`simulate_normal` (5.1–5.4).** Try numpy's Cholesky; on failure try
  `chol_psd`; on failure repair with `fix_method` and take `chol_psd` again.
  5.2's input has a perfect correlation between x1 and x2 and is PSD but not PD,
  so it goes through `chol_psd` without repair.
- **`simulate_pca` (5.5).** Eigenvalues sorted largest first; drop anything
  below 1e-8; keep the smallest number of factors whose cumulative variance share
  reaches 99%.
- **Returns (6.x).** Arithmetic \(P_t/P_{t-1} - 1\); log \(\ln(P_t/P_{t-1})\). The
  first date is dropped.
- **Generalised t (7.2, 7.3).** \(x = \mu + s\,T\), \(T \sim t_\nu\), so `sigma`
  in the output is the scale, not the standard deviation. Bounds \(s > 10^{-6}\),
  \(\nu > 2.0001\). In the regression (7.3) the error location is fixed at 0 and
  the intercept `Alpha` absorbs it, which is why `mu` reports as 0.0. Starting
  values come from OLS and the moment relation \(\text{excess kurtosis} =
  6/(\nu - 4)\).
- **AICc (7.4).** \(-2\ln L + 2k + 2k(k+1)/(n-k-1)\) with \(k = 3\) for the
  generalised t (location, scale, \(\nu\)); regression betas would add to \(k\).
- **NIG parameterisation (7.5, 7.6).** Output is \((\mu, \alpha, \beta, \delta)\)
  as in the notes. `scipy.stats.norminvgauss` uses \((a, b, \text{loc},
  \text{scale})\) with \(a = \alpha\delta\), \(b = \beta\delta\), so
  `fit_nig_mle` converts: \(\delta = \text{scale}\), \(\alpha = a/\delta\),
  \(\beta = b/\delta\), \(\mu = \text{loc}\).
- **NIG method of moments (7.5).** Skewness and excess kurtosis use the
  population (biased) estimators, variance uses \(n-1\), matching the Julia
  `StatsBase` conventions the expected file was generated with. The inversion is
  \(\rho^2 = t/(3-4t)\) with \(t = \text{skew}^2/\text{kurt}\), then
  \(\delta\gamma = 3(1+4\rho^2)/\text{kurt}\), \(\alpha\) from the variance, and
  \(\mu\) from the mean.

"""
common.py -- shared helpers for FinTech 545 Assignment 2.

Everything that more than one problem needs lives here so the per-problem
modules stay short and readable:

  * data loading, resolved relative to this file
  * moment estimators (Week 01 s3-4), AICc and BIC (Week 02 s5.3)
  * VaR and ES from a sample and under a Normal (Week 04 s4.2, Week 05 s5.1)
  * exponentially weighted variance (Week 03 s3.5)
  * the two non-PSD repairs, Rebonato-Jackel and Higham, and a PSD-tolerant
    Cholesky (Week 03 s3.6, s3.2.2)
  * multivariate normal simulation through the Cholesky root (Week 03 s3.1.2)
  * a small matplotlib style and colour palette

Conventions chosen here are documented inline, because a grader cannot tell a
deliberate convention from an accident. The ones that matter most:

  VaR / ES SIGN.  Both are reported as POSITIVE numbers for a loss. Week 04 s4.2
      defines VaR_a(X) = -F^{-1}(a) on the P&L distribution and Week 05 s5.1 puts
      the same leading minus on ES. A negative VaR therefore means the a-quantile
      of P&L is a PROFIT, which actually happens in Problem 3.

  EMPIRICAL QUANTILE.  Week 04 s4.5 finds the a-quantile of N simulated or
      historical values by sorting and taking the N*a-th value. That is the
      convention used in `var_es`: with N*a an integer (it is for every N and a
      in this assignment: 500*0.05, 10000*0.05, 10000*0.01, 1000*0.05, ...) VaR
      is minus the N*a-th smallest P&L and ES is minus the mean of those N*a
      smallest values -- literally "the average of the a% worst scenarios".
      numpy's default interpolating quantile differs from this by a fraction of
      one order statistic; Problem 2 prints both to show nothing depends on it.

  NORMAL QUANTILE.  The exact Phi^{-1}(0.05) = -1.6449 is used throughout, not
      the rounded 1.65 in the slides, so that closed-form and simulated numbers
      can be compared to four digits.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

# --------------------------------------------------------------------------
# Data location. Every module resolves CSVs relative to THIS file, so the code
# runs the same from the notebook, from the shell, or from another directory.
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# One seed for every Monte Carlo step in the assignment, so each module prints
# the same numbers on every run and the write-up can quote them.
SEED = 545


def load(name: str) -> pd.DataFrame:
    """Load one of the assignment CSVs by short name, e.g. load('problem1')."""
    return pd.read_csv(os.path.join(HERE, f"{name}.csv"))


# --------------------------------------------------------------------------
# Moments  (Week 01 s3.2-4.4, same conventions as Assignment 1)
# --------------------------------------------------------------------------
def moments(x) -> dict:
    """First four moments of a 1-D sample.

    * mean, variance with n-1 (Week 01 s4.2)
    * skewness and EXCESS kurtosis (Normal = 0), bias-corrected sample forms
      (Week 01 s4.3). These match scipy's skew(bias=False) and
      kurtosis(bias=False). The population (1/n) forms are returned too, per the
      s4.4 warning that packages differ -- on every series in this assignment the
      two agree to the second decimal and no conclusion depends on the choice.
    * se_skew = sqrt(6/n) and se_exkurt = sqrt(24/n): the standard errors of the
      two shape statistics UNDER THE NULL of normality, which is what turns
      "is 2.17 big?" into a decidable question.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    mean = x.mean()
    d = x - mean
    m2, m3, m4 = np.mean(d ** 2), np.mean(d ** 3), np.mean(d ** 4)
    var = x.var(ddof=1)
    sd = np.sqrt(var)
    skew_pop = m3 / m2 ** 1.5
    exkurt_pop = m4 / m2 ** 2 - 3.0
    skew_bc = n / ((n - 1) * (n - 2)) * np.sum(d ** 3) / sd ** 3
    exkurt_bc = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * exkurt_pop + 6)
    return {"n": n, "mean": mean, "var": var, "sd": sd,
            "skew": skew_bc, "exkurt": exkurt_bc,
            "skew_pop": skew_pop, "exkurt_pop": exkurt_pop,
            "se_skew": np.sqrt(6.0 / n), "se_exkurt": np.sqrt(24.0 / n)}


def print_moments(label: str, m: dict, pct: bool = False) -> None:
    """One formatted row of the four moments. pct=True prints mean/sd in %."""
    scale = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    print(f"  {label:<10}{m['mean'] * scale:>12.4f}{unit}{m['sd'] * scale:>11.4f}{unit}"
          f"{m['skew']:>9.3f}{m['exkurt']:>9.3f}")


# --------------------------------------------------------------------------
# Model selection  (Week 02 s5.3)
# --------------------------------------------------------------------------
def aicc(loglik: float, k: int, n: int) -> float:
    """AICc = -2 ln L + 2k + (2k^2 + 2k)/(n - k - 1).

    k counts every fitted parameter: for a Normal margin k = 2 (mu, sigma), for a
    Student's t margin k = 3 (mu, scale, nu). For the copula comparison in
    Problem 4 the margins are fitted first and frozen, so (Week 05 s5.6.5)
    k_Gauss = 0 and k_t = 1 -- only the parameters the copula itself fits.
    Absolute values mean nothing; only differences on the same data do.
    """
    if n - k - 1 <= 0:
        return np.inf
    return -2.0 * loglik + 2.0 * k + (2.0 * k * k + 2.0 * k) / (n - k - 1)


def bic(loglik: float, k: int, n: int) -> float:
    """BIC = k ln n - 2 ln L. Charges ln(n) per parameter against AIC's 2, so it
    is the harder test for the extra copula parameter in Problem 4."""
    return k * np.log(n) - 2.0 * loglik


# --------------------------------------------------------------------------
# Distribution fits (Week 01 s5, Week 02 s4)
# --------------------------------------------------------------------------
NU_MAX = 100.0   # Week 05 s5.5.3.2: past nu ~ 100 a t and a Normal are indistinguishable


def fit_normal(x) -> dict:
    """Normal by MLE. The MLE of sigma divides by n; the n-1 version is what the
    moments report. Both are returned; the log-likelihood is at the MLE."""
    x = np.asarray(x, dtype=float)
    mu, sig = x.mean(), x.std(ddof=0)
    return {"mu": mu, "sigma": sig, "sigma_unbiased": x.std(ddof=1),
            "loglik": float(np.sum(stats.norm.logpdf(x, mu, sig))), "k": 2,
            "dist": stats.norm(mu, sig)}


def fit_t(x) -> dict:
    """MLE of x = mu + s * T_nu  (location-scale Student's t).

    `scale` s is NOT the standard deviation: sd = s * sqrt(nu / (nu - 2)). nu is
    bounded below at 2.0001 so the variance exists, and above at NU_MAX = 100.
    The likelihood is very flat in nu (Week 05 s5.6.5.1 makes the same point for
    the copula), so a single gradient run can stop a few tenths of a log-lik
    unit short. Three attempts are made -- L-BFGS-B with tight tolerances, a
    Nelder-Mead polish, and scipy's own t.fit -- and the best is kept.
    """
    x = np.asarray(x, dtype=float)

    def nll(p):
        return -np.sum(stats.t.logpdf(x, df=p[2], loc=p[0], scale=p[1]))

    k0 = max(stats.kurtosis(x), 0.05)
    nu0 = min(max(6.0 / k0 + 4.0, 3.0), 50.0)
    x0 = [x.mean(), x.std(ddof=1) * np.sqrt((nu0 - 2) / nu0), nu0]
    bounds = [(None, None), (1e-8, None), (2.0001, NU_MAX)]
    cands = []
    r1 = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                           options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000})
    cands.append(r1.x)
    r2 = optimize.minimize(nll, r1.x, method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    cands.append(r2.x)
    df, loc, sc = stats.t.fit(x)
    cands.append(np.array([loc, sc, df]))
    lo = np.array([-np.inf, 1e-8, 2.0001])
    hi = np.array([np.inf, np.inf, NU_MAX])
    best = min((np.clip(c, lo, hi) for c in cands), key=nll)
    mu, s, nu = best
    return {"mu": mu, "scale": s, "nu": nu, "loglik": float(-nll(best)), "k": 3,
            "sd_implied": s * np.sqrt(nu / (nu - 2)) if nu > 2 else np.inf,
            "dist": stats.t(df=nu, loc=mu, scale=s)}


# --------------------------------------------------------------------------
# VaR and ES
# --------------------------------------------------------------------------
Z05 = stats.norm.ppf(0.05)   # -1.6449, the exact standard normal 5% quantile


def var_es(pnl, alpha: float = 0.05) -> tuple:
    """Historical / simulated VaR and ES at level alpha, positive = loss.

    Week 04 s4.5 convention: sort the N P&L values and take the N*alpha-th as the
    alpha-quantile. VaR = -(that value). Week 05 s5.1: ES is the mean of all
    values at or below the VaR quantile, negated -- here exactly the mean of the
    N*alpha worst outcomes. If N*alpha is not an integer it is rounded up, so the
    tail never contains fewer than alpha of the sample.
    """
    x = np.sort(np.asarray(pnl, dtype=float))
    n = x.size
    k = int(np.ceil(n * alpha - 1e-9))          # N*alpha-th order statistic
    k = min(max(k, 1), n)
    q = x[k - 1]
    return -q, -x[:k].mean()


def var_normal(sigma: float, alpha: float = 0.05, mu: float = 0.0,
               pv: float = 1.0) -> float:
    """Normal VaR in currency: -(mu + sigma * Phi^{-1}(alpha)) * PV (Week 04 s4.4).

    With mu = 0 this is the familiar 1.645 * sigma * PV at alpha = 5%. Keeping mu
    gives the "absolute" convention of Week 04 s4.2.1 (loss measured from zero),
    which is the one used when Problem 3 asks for a Normal VaR from the sample
    mean and standard deviation.
    """
    return -(mu + sigma * stats.norm.ppf(alpha)) * pv


def es_normal(sigma: float, alpha: float = 0.05, mu: float = 0.0,
              pv: float = 1.0) -> float:
    """Normal ES: (-mu + sigma * phi(Phi^{-1}(alpha)) / alpha) * PV (Week 05 s5.1.1).
    At 5% the multiplier on sigma is 2.063, against 1.645 for VaR."""
    z = stats.norm.ppf(alpha)
    return (-mu + sigma * stats.norm.pdf(z) / alpha) * pv


# --------------------------------------------------------------------------
# Exponentially weighted variance  (Week 03 s3.5)
# --------------------------------------------------------------------------
def ew_weights(n: int, lam: float) -> np.ndarray:
    """w_i = (1 - lam) * lam^(n - i), i = 1..n, normalised to sum to one.
    The LAST observation is the most recent and carries the largest weight."""
    i = np.arange(1, n + 1)
    w = (1.0 - lam) * lam ** (n - i)
    return w / w.sum()


def ew_variance(x, lam: float, demeaned: bool = True) -> float:
    """sigma^2 = sum_i w_i (x_i - xbar)^2 with the s3.5.1 weights.

    demeaned=True takes the series as already centred (the mean has been removed
    once, with equal weights, as Problem 2 instructs) and sums w_i x_i^2. With
    demeaned=False the weighted mean sum_i w_i x_i is removed instead, which is
    the literal s3.5.1 estimator. Problem 2 prints both; they differ in the
    fourth significant figure.
    """
    x = np.asarray(x, dtype=float)
    w = ew_weights(x.size, lam)
    if not demeaned:
        x = x - w @ x
    return float(np.sum(w * x ** 2))


def n_eff(lam: float) -> float:
    """Effective sample size (1 + lam)/(1 - lam), Week 03 s3.5.3 (infinite horizon)."""
    return (1.0 + lam) / (1.0 - lam)


def half_life(lam: float) -> float:
    """Half life ln(0.5)/ln(lam) in observations, Week 03 s3.5.3."""
    return np.log(0.5) / np.log(lam)


# --------------------------------------------------------------------------
# Non-PSD repairs  (Week 03 s3.6)  and a PSD-tolerant Cholesky (s3.2.2)
# --------------------------------------------------------------------------
def near_psd(corr: np.ndarray, epsilon: float = 0.0) -> np.ndarray:
    """Rebonato & Jackel, Week 03 s3.6.1, on a CORRELATION matrix.

      1. C S = S Lambda                     (eigen-decomposition)
      2. Lambda' = max(Lambda, epsilon)     (floor the eigenvalues)
      3. t_i = 1 / sum_j s_ij^2 lambda'_j   (row rescaling that restores the
                                             unit diagonal)
      4. B = sqrt(T) S sqrt(Lambda'),  C_hat = B B'
    Not the nearest PSD matrix in any norm; a projection that is "oftentimes
    good enough" (s3.6.1).
    """
    c = np.asarray(corr, dtype=float)
    vals, vecs = np.linalg.eigh(c)
    vals = np.maximum(vals, epsilon)
    t = 1.0 / ((vecs ** 2) @ vals)
    b = np.sqrt(t)[:, None] * vecs * np.sqrt(vals)[None, :]
    out = b @ b.T
    return 0.5 * (out + out.T)


def higham_nearest_corr(corr: np.ndarray, tol: float = 1e-9, max_iter: int = 200,
                        eig_tol: float = 1e-9) -> tuple:
    """Higham (2002), Week 03 s3.6.2, with W = I: alternating projections with
    Dykstra's correction onto (PSD matrices) and (unit-diagonal matrices).

        R_k = Y_{k-1} - dS_{k-1}
        X_k = P_S(R_k)        eigenvalues floored at 0
        dS_k = X_k - R_k      Dykstra correction carried to the next step
        Y_k = P_U(X_k)        diagonal reset to 1
    Stops when the change in ||Y_k - Y_0||_F is below `tol` AND the smallest
    eigenvalue is above -eig_tol (s3.6.3: "check the eigenvalue, not the norm").
    Returns (Y, iterations).
    """
    y = np.asarray(corr, dtype=float).copy()
    y0 = y.copy()
    ds = np.zeros_like(y)
    prev = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        r = y - ds
        vals, vecs = np.linalg.eigh(r)
        x = (vecs * np.maximum(vals, 0.0)) @ vecs.T
        ds = x - r
        y = x.copy()
        np.fill_diagonal(y, 1.0)
        norm = np.linalg.norm(y - y0, "fro")
        if abs(norm - prev) < tol and np.linalg.eigvalsh(y).min() > -eig_tol:
            break
        prev = norm
    return 0.5 * (y + y.T), it


def chol_psd(a: np.ndarray, epsilon: float = -1e-8) -> np.ndarray:
    """Lower-triangular L with L L' = a, tolerating zero eigenvalues (Week 03
    s3.2.2). A diagonal residual in [epsilon, 0] is treated as an exact zero and
    the rest of that column is zeroed. Anything more negative is a genuinely
    non-PSD matrix and raises ValueError naming the pivot -- Problem 1 uses that
    message to show WHERE the pairwise matrix breaks.
    """
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    root = np.zeros((n, n))
    for j in range(n):
        temp = a[j, j] - root[j, :j] @ root[j, :j]
        if epsilon <= temp <= 0.0:
            temp = 0.0
        if temp < 0.0:
            raise ValueError(f"not PSD: pivot {temp:.3e} at column {j}")
        root[j, j] = np.sqrt(temp)
        if root[j, j] == 0.0:
            continue
        for i in range(j + 1, n):
            root[i, j] = (a[i, j] - root[i, :j] @ root[j, :j]) / root[j, j]
    return root


def simulate_normal(cov: np.ndarray, nsim: int, rng: np.random.Generator) -> np.ndarray:
    """X = (L Z)' with L L' = cov, Z ~ N(0, I): the Week 03 s3.1.2 multivariate
    draw algorithm. Returns an (nsim x n) array, one row per scenario. Uses the
    PSD-tolerant root so a singular but valid covariance still simulates."""
    cov = np.asarray(cov, dtype=float)
    try:
        root = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        root = chol_psd(cov)
    z = rng.standard_normal((cov.shape[0], nsim))
    return (root @ z).T


def frob(a: np.ndarray, b: np.ndarray) -> float:
    """Frobenius distance ||a - b||_F (Week 03 s3.6.2)."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b), "fro"))


# --------------------------------------------------------------------------
# Printing helpers
# --------------------------------------------------------------------------
def header(title: str) -> None:
    """Print a section header when a module is run as a script."""
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)


def print_matrix(m, labels, title: str | None = None, fmt: str = "{:>9.4f}",
                 indent: str = "  ") -> None:
    """Print a square matrix with row/column labels, fixed width (<= 88 cols)."""
    m = np.asarray(m)
    if title:
        print(f"{indent}{title}")
    print(indent + " " * 6 + "".join(f"{l:>9}" for l in labels))
    for i, l in enumerate(labels):
        print(indent + f"{l:<6}" + "".join(fmt.format(v) for v in m[i]))


def money(x: float) -> str:
    """$ with thousands separators and no decimals; negative shown as -$."""
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


# --------------------------------------------------------------------------
# Plot style + palette (same as Assignment 1)
# --------------------------------------------------------------------------
C1 = "#2a78d6"  # blue    -- the data / the primary series
C2 = "#eb6834"  # orange  -- the model being criticised
C3 = "#1baf7a"  # aqua    -- the alternative
GRID = "#dcdbd6"
INK = "#0b0b0b"
INK2 = "#52514e"


def use_style() -> None:
    """Apply a quiet, print-friendly style: recessive grid, thin marks."""
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
        "legend.fontsize": 8.5, "lines.linewidth": 2.0, "figure.dpi": 110,
        "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9.5,
    })


def finish(fig, name: str) -> None:
    """Save to figures/<name>.png; show inline in a notebook, close on Agg."""
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"))
    if mpl.get_backend().lower().endswith("agg"):
        plt.close(fig)
    else:
        plt.show()

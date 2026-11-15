"""
common.py -- shared helpers for FinTech 545 Assignment 3.

Everything that more than one problem needs lives here so the per-problem
modules stay short:

  * data loading, resolved relative to this file
  * the generalized Black-Scholes-Merton formula and its closed-form Greeks
    (Week 06 s"The Generalized Form", Week 07 Table "The Main Greeks"), written
    from scratch as the assignment requires -- a line-by-line port of the
    class library/gbsm.jl
  * implied volatility by root finding on GBSM (Week 06 s"Implied Volatility")
  * the CRR binomial tree for American options, recombining and with discrete
    dividends (Week 07 s"Valuing with a Binomial Tree" and s"Implementing the
    Non-Recombining Tree") -- a port of library/bt_american.jl
  * moments, AICc, Normal and Student-t fits (Week 01, Week 02 s5.3)
  * VaR and ES from a sample and under a Normal (Week 04 s4.2, Week 05 s5.1)
  * OLS with standard errors (Week 02)
  * a quiet matplotlib style

Conventions chosen here, documented because a grader cannot tell a deliberate
convention from an accident:

  VaR / ES SIGN.  Both are POSITIVE numbers for a loss (assignment instructions;
      Week 04 s4.2 defines VaR_a = -F^{-1}(a) on the P&L distribution).

  EMPIRICAL QUANTILE.  `var_es` follows the class library/RiskStats.jl exactly:
      the a-quantile is the average of the floor(N a)-th and ceil(N a)-th order
      statistics, and ES is minus the mean of every value at or below it. For
      the 100,000-path simulations N a is an integer and this is just the N a-th
      value; for 240 months at 1% (Problem 4) N a = 2.4 and it averages the 2nd
      and 3rd worst months.

  NORMAL QUANTILE.  The exact Phi^{-1}(0.05) = -1.6449 is used, not 1.65.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

# --------------------------------------------------------------------------
# Data location. Every module resolves files relative to THIS file, so the code
# runs the same from the notebook, from the shell, or from another directory.
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# One seed for every Monte Carlo step, so the write-up can quote exact numbers.
SEED = 545


def load(name: str, **kw) -> pd.DataFrame:
    """Load one of the assignment CSVs by file name (with or without .csv)."""
    if not name.lower().endswith(".csv"):
        name = name + ".csv"
    return pd.read_csv(os.path.join(HERE, name), **kw)


# --------------------------------------------------------------------------
# Generalized Black-Scholes-Merton  (Week 06, Week 07)
#
#   b = r       Black-Scholes 1973 (no dividend)          -- Problems 2, 3
#   b = r - q   Merton 1973 (continuous dividend yield q)  -- Problem 1
# --------------------------------------------------------------------------
def _d1d2(S, X, T, r, b, sigma):
    d1 = (np.log(S / X) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def gbsm(call: bool, S, X, T, r, b, sigma) -> np.ndarray:
    """GBSM value. Vectorised over any of the inputs.

        Call = S e^{(b-r)T} Phi(d1) - X e^{-rT} Phi(d2)
        Put  = X e^{-rT} Phi(-d2) - S e^{(b-r)T} Phi(-d1)

    At T <= 0 the payoff is returned, so a position can be revalued at expiry.
    """
    S = np.asarray(S, dtype=float)
    T = np.asarray(T, dtype=float)
    if np.all(T <= 0):
        return np.maximum(0.0, (S - X) if call else (X - S))
    d1, d2 = _d1d2(S, X, T, r, b, sigma)
    if call:
        return S * np.exp((b - r) * T) * stats.norm.cdf(d1) - X * np.exp(-r * T) * stats.norm.cdf(d2)
    return X * np.exp(-r * T) * stats.norm.cdf(-d2) - S * np.exp((b - r) * T) * stats.norm.cdf(-d1)


def gbsm_greeks(call: bool, S, X, T, r, b, sigma) -> dict:
    """Closed-form Greeks from Week 07 (same formulas as library/gbsm.jl).

    theta is -dV/dT, i.e. the change in value per YEAR of calendar time passing
    (negative for a long option). Divide by 255 or 365 for a per-day number.
    vega is dV/dsigma per 1.00 of vol (divide by 100 for per vol point).
    """
    d1, d2 = _d1d2(S, X, T, r, b, sigma)
    carry = np.exp((b - r) * T)
    disc = np.exp(-r * T)
    pdf1 = stats.norm.pdf(d1)
    gamma = pdf1 * carry / (S * sigma * np.sqrt(T))
    vega = S * carry * pdf1 * np.sqrt(T)
    if call:
        delta = carry * stats.norm.cdf(d1)
        theta = (-S * carry * pdf1 * sigma / (2 * np.sqrt(T))
                 - (b - r) * S * carry * stats.norm.cdf(d1)
                 - r * X * disc * stats.norm.cdf(d2))
        rho = T * X * disc * stats.norm.cdf(d2)
    else:
        delta = carry * (stats.norm.cdf(d1) - 1.0)
        theta = (-S * carry * pdf1 * sigma / (2 * np.sqrt(T))
                 + (b - r) * S * carry * stats.norm.cdf(-d1)
                 + r * X * disc * stats.norm.cdf(-d2))
        rho = -T * X * disc * stats.norm.cdf(-d2)
    return {"value": gbsm(call, S, X, T, r, b, sigma), "delta": delta,
            "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def implied_vol(call: bool, price: float, S, X, T, r, b,
                lo: float = 1e-4, hi: float = 5.0) -> float:
    """Invert GBSM for sigma with Brent's method (Week 06: the value is
    monotone in sigma, so the root is unique when it exists).

    Returns NaN when no sigma in [lo, hi] reproduces the price. That happens in
    exactly two ways, and Problem 1(d) reports which:
      * price at or below the sigma -> 0 limit  max(0, S e^{(b-r)T} - X e^{-rT})
        for a call (the discounted intrinsic value): no volatility is low enough;
      * price above the sigma -> infinity limit (S e^{(b-r)T} for a call).
    """
    f = lambda s: gbsm(call, S, X, T, r, b, s) - price
    flo, fhi = f(lo), f(hi)
    if flo > 0 or fhi < 0:
        return np.nan
    return optimize.brentq(f, lo, hi, xtol=1e-12, maxiter=500)


def iv_lower_bound(call: bool, S, X, T, r, b) -> float:
    """sigma -> 0 limit of the GBSM value: the discounted forward intrinsic."""
    fwd = S * np.exp((b - r) * T)
    pvx = X * np.exp(-r * T)
    return max(0.0, fwd - pvx) if call else max(0.0, pvx - fwd)


# --------------------------------------------------------------------------
# Binomial trees  (Week 06 CRR tree, Week 07 American + discrete dividends)
# --------------------------------------------------------------------------
def bt_american(call: bool, S, X, T, r, b, sigma, N: int, american: bool = True) -> float:
    """Recombining CRR tree, u = e^{sigma sqrt(dt)}, d = 1/u,
    p = (e^{b dt} - d)/(u - d), discounted at r (Week 06/07).

    Backward induction with the Week 07 max(payoff, continuation) at every node
    when american=True; american=False drops the max and prices the European
    option on the same tree. Vectorised over the nodes of each time step, so
    N = 1000 runs in milliseconds.
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd_ = 1.0 - pu
    df = np.exp(-r * dt)
    z = 1.0 if call else -1.0
    # terminal nodes, i = number of up moves (0..N)
    i = np.arange(N + 1)
    v = np.maximum(0.0, z * (S * u ** i * d ** (N - i) - X))
    for j in range(N - 1, -1, -1):
        v = df * (pu * v[1:] + pd_ * v[:-1])
        if american:
            i = np.arange(j + 1)
            v = np.maximum(v, z * (S * u ** i * d ** (j - i) - X))
    return float(v[0])


def bt_american_div(call: bool, S, X, T, r, div_amts, div_steps, sigma, N: int,
                    american: bool = True) -> float:
    """Non-recombining tree for discrete cash dividends (Week 07, a port of
    library/bt_american.jl's dividend method).

    div_steps are integer step indices on the 0..N grid at which each dividend
    is paid. The tree recombines up to the first dividend step n1. At each node
    on that step:
        no exercise : recursively price the option on a fresh tree starting at
                      S_node - D_1, with the remaining dividends and N - n1 steps
        exercise    : the payoff at the CUM-dividend price S_node, i.e. the
                      Week 07 rule "exercising at j = n gets the holder the
                      stock plus the dividend"
    and then backward induction to the root. With american=False the exercise
    branch is skipped everywhere, which prices the European option with the
    same discrete dividends on the same grid (Problem 2(c) compares this to
    GBSM with the PV of dividends removed).

    The stock grows at the risk free rate (b = r) between dividends: the
    dividends are the only carry, exactly as in bt_american.jl.
    """
    div_amts = list(div_amts)
    div_steps = list(div_steps)
    if not div_amts or div_steps[0] > N:
        return bt_american(call, S, X, T, r, r, sigma, N, american)

    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(r * dt) - d) / (u - d)
    pd_ = 1.0 - pu
    df = np.exp(-r * dt)
    z = 1.0 if call else -1.0
    n1 = div_steps[0]

    # Nodes on the dividend step: each spawns its own sub-tree.
    i = np.arange(n1 + 1)
    s_n1 = S * u ** i * d ** (n1 - i)
    rest_amts = div_amts[1:]
    rest_steps = [k - n1 for k in div_steps[1:]]
    v = np.array([bt_american_div(call, s - div_amts[0], X, T - n1 * dt, r,
                                  rest_amts, rest_steps, sigma, N - n1, american)
                  for s in s_n1])
    if american:
        v = np.maximum(v, z * (s_n1 - X))
    for j in range(n1 - 1, -1, -1):
        v = df * (pu * v[1:] + pd_ * v[:-1])
        if american:
            i = np.arange(j + 1)
            v = np.maximum(v, z * (S * u ** i * d ** (j - i) - X))
    return float(v[0])


def _bt_div_vec(call, S, X, T, r, div_amts, div_steps, sigma, N, american):
    """Same algorithm as bt_american_div, vectorised over an ARRAY of starting
    prices S (shape (m,)). Every sub-tree spawned on a dividend step has the
    same length and the same remaining dividends, so all of them can be rolled
    back together as one (m, nodes) array instead of one Python call each.
    That is the only difference; the node values are identical."""
    S = np.asarray(S, dtype=float)
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(r * dt) - d) / (u - d)
    pd_ = 1.0 - pu
    df = np.exp(-r * dt)
    z = 1.0 if call else -1.0

    if not div_amts or div_steps[0] > N:
        n_last = N                                   # plain recombining tree
        i = np.arange(n_last + 1)
        v = np.maximum(0.0, z * (S[:, None] * u ** i * d ** (n_last - i) - X))
    else:
        n_last = div_steps[0]
        i = np.arange(n_last + 1)
        s_n = S[:, None] * u ** i * d ** (n_last - i)          # (m, n1+1)
        cont = _bt_div_vec(call, (s_n - div_amts[0]).ravel(), X, T - n_last * dt, r,
                           div_amts[1:], [k - n_last for k in div_steps[1:]],
                           sigma, N - n_last, american).reshape(s_n.shape)
        v = np.maximum(cont, z * (s_n - X)) if american else cont
    for j in range(n_last - 1, -1, -1):
        v = df * (pu * v[:, 1:] + pd_ * v[:, :-1])
        if american:
            i = np.arange(j + 1)
            v = np.maximum(v, z * (S[:, None] * u ** i * d ** (j - i) - X))
    return v[:, 0]


def bt_div(call: bool, S, X, T, r, div_amts, div_steps, sigma, N: int,
           american: bool = True):
    """Fast entry point for the discrete-dividend tree. S may be a scalar or an
    array (one tree per starting price, used for bump-and-revalue Greeks)."""
    s = np.atleast_1d(np.asarray(S, dtype=float))
    out = _bt_div_vec(call, s, X, T, r, list(div_amts), list(div_steps), sigma, N, american)
    return float(out[0]) if np.ndim(S) == 0 else out


# --------------------------------------------------------------------------
# Moments, model selection, distribution fits  (Week 01, Week 02 s5.3)
# --------------------------------------------------------------------------
def moments(x) -> dict:
    """Mean, sd (n-1), bias-corrected skewness and EXCESS kurtosis (Normal = 0),
    matching scipy's skew(bias=False) / kurtosis(bias=False)."""
    x = np.asarray(x, dtype=float)
    return {"n": x.size, "mean": x.mean(), "sd": x.std(ddof=1),
            "skew": stats.skew(x, bias=False),
            "exkurt": stats.kurtosis(x, bias=False)}


def aicc(loglik: float, k: int, n: int) -> float:
    """AICc = -2 ln L + 2k + (2k^2 + 2k)/(n - k - 1)   (Week 02 s5.3)."""
    return -2.0 * loglik + 2.0 * k + (2.0 * k * k + 2.0 * k) / (n - k - 1)


NU_MAX = 100.0   # past nu ~ 100 a t and a Normal are indistinguishable


def fit_normal(x) -> dict:
    """Normal by MLE (sigma divides by n); k = 2."""
    x = np.asarray(x, dtype=float)
    mu, sig = x.mean(), x.std(ddof=0)
    ll = float(np.sum(stats.norm.logpdf(x, mu, sig)))
    return {"mu": mu, "sigma": sig, "loglik": ll, "k": 2,
            "aicc": aicc(ll, 2, x.size)}


def fit_t(x) -> dict:
    """MLE of x = mu + s T_nu (location-scale t); k = 3.

    s is a SCALE, not the sd: sd = s sqrt(nu/(nu-2)). nu is bounded to
    [2.0001, 100]. The likelihood is flat in nu, so an L-BFGS-B run is polished
    by Nelder-Mead and compared with scipy's t.fit; the best is kept.
    """
    x = np.asarray(x, dtype=float)

    def nll(p):
        return -np.sum(stats.t.logpdf(x, df=p[2], loc=p[0], scale=p[1]))

    k0 = max(stats.kurtosis(x), 0.05)
    nu0 = min(max(6.0 / k0 + 4.0, 3.0), 50.0)
    x0 = [x.mean(), x.std(ddof=1) * np.sqrt((nu0 - 2) / nu0), nu0]
    bounds = [(None, None), (1e-10, None), (2.0001, NU_MAX)]
    r1 = optimize.minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                           options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000})
    r2 = optimize.minimize(nll, r1.x, method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    df_, loc, sc = stats.t.fit(x)
    lo = np.array([-np.inf, 1e-10, 2.0001])
    hi = np.array([np.inf, np.inf, NU_MAX])
    best = min((np.clip(c, lo, hi) for c in (r1.x, r2.x, np.array([loc, sc, df_]))), key=nll)
    mu, s, nu = best
    ll = float(-nll(best))
    return {"mu": mu, "scale": s, "nu": nu, "loglik": ll, "k": 3,
            "aicc": aicc(ll, 3, x.size), "sd_implied": s * np.sqrt(nu / (nu - 2))}


# --------------------------------------------------------------------------
# VaR and ES
# --------------------------------------------------------------------------
def var_es(pnl, alpha: float = 0.05) -> tuple:
    """Empirical VaR and ES, positive = loss, exactly as library/RiskStats.jl:
    v = average of the floor(N a)-th and ceil(N a)-th order statistics,
    VaR = -v, ES = -mean(x[x <= v])."""
    x = np.sort(np.asarray(pnl, dtype=float))
    n = x.size
    nup = int(np.ceil(n * alpha - 1e-12))
    ndn = max(int(np.floor(n * alpha + 1e-12)), 1)
    v = 0.5 * (x[nup - 1] + x[ndn - 1])
    return -v, -x[x <= v].mean()


def var_normal(sigma, alpha=0.05, mu=0.0):
    """-(mu + sigma Phi^{-1}(alpha))   (Week 04 s4.4)."""
    return -(mu + sigma * stats.norm.ppf(alpha))


def es_normal(sigma, alpha=0.05, mu=0.0):
    """-mu + sigma phi(Phi^{-1}(alpha))/alpha   (Week 05 s5.1.1)."""
    return -mu + sigma * stats.norm.pdf(stats.norm.ppf(alpha)) / alpha


# --------------------------------------------------------------------------
# OLS  (Week 02)
# --------------------------------------------------------------------------
def ols(y, X, names) -> dict:
    """y = X b + e with an intercept column already in X.

    Returns coefficients, classical standard errors (sigma^2 (X'X)^{-1} with
    sigma^2 = SSE/(n - p)), t statistics, R^2, the Normal log-likelihood at the
    MLE of the error variance, and AICc with k = p + 1 (the p coefficients plus
    the error variance, Week 02 s5.3 counts every fitted parameter).
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sse = float(resid @ resid)
    s2 = sse / (n - p)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    r2 = 1.0 - sse / float(((y - y.mean()) ** 2).sum())
    sig_mle = np.sqrt(sse / n)
    ll = float(np.sum(stats.norm.logpdf(resid, 0.0, sig_mle)))
    return {"names": list(names), "beta": beta, "se": se, "t": beta / se,
            "r2": r2, "resid": resid, "loglik": ll, "k": p + 1,
            "aicc": aicc(ll, p + 1, n), "n": n, "s": np.sqrt(s2)}


# --------------------------------------------------------------------------
# Printing helpers
# --------------------------------------------------------------------------
def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def show(df: pd.DataFrame, fmt: str | dict = "{:,.4f}", index: bool = True) -> None:
    """Print a DataFrame as plain fixed-width text (renders identically in the
    notebook, the terminal and the PDF, unlike HTML tables)."""
    out = df.copy()
    for c in out.columns:
        f = fmt.get(c, None) if isinstance(fmt, dict) else fmt
        if f is not None and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v, f=f: "" if pd.isna(v) else f.format(v))
    print(out.to_string(index=index))


def money(x: float, dp: int = 0) -> str:
    return f"-${-x:,.{dp}f}" if x < 0 else f"${x:,.{dp}f}"


# --------------------------------------------------------------------------
# Plot style + palette (same as Assignments 1 and 2)
# --------------------------------------------------------------------------
C1 = "#2a78d6"  # blue
C2 = "#eb6834"  # orange
C3 = "#1baf7a"  # aqua
C4 = "#8a5cd1"  # purple
GRID = "#dcdbd6"
INK = "#0b0b0b"
INK2 = "#52514e"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": INK2, "axes.linewidth": 0.8, "axes.labelcolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.frameon": False,
        "legend.fontsize": 8.5, "lines.linewidth": 1.8, "figure.dpi": 110,
        "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 9.5,
    })


def finish(fig, name: str) -> None:
    """Save to figures/<name>.png; show inline in a notebook, close on Agg."""
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"))
    if mpl.get_backend().lower().endswith("agg"):
        plt.close(fig)
    else:
        plt.show()

"""
fitting.py -- distribution and regression fits by MLE / moments (Weeks 01-02).

  FittedModel       container: fitted error distribution, betas, residuals, U's
  fit_normal        Normal by mean / sample sd                     (test 7.1)
  fit_general_t     location-scale Student's t by MLE              (test 7.2)
  fit_regression_t  linear regression with t errors by MLE         (test 7.3)
  aicc              corrected Akaike information criterion         (test 7.4)
  fit_nig_moments   Normal Inverse Gaussian by method of moments   (test 7.5)
  fit_nig_mle       Normal Inverse Gaussian by MLE (scipy)         (test 7.6)

Parameterisations
-----------------
* Generalised t:  x = mu + s * T,  T ~ t(nu). `s` is a SCALE, not the standard
  deviation; sd = s * sqrt(nu / (nu - 2)).
* NIG: (mu, alpha, beta, delta) as in the lecture notes. scipy's norminvgauss uses
  (a, b, loc, scale) with a = alpha*delta, b = beta*delta, loc = mu, scale = delta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import optimize, stats


# --------------------------------------------------------------------------
# container
# --------------------------------------------------------------------------
@dataclass
class FittedModel:
    """
    beta         regression coefficients (intercept first) or None
    error_model  frozen scipy distribution of the error term
    eval         function mapping U in (0,1) -> simulated value(s)
    errors       fitted residuals (x - mu for a plain distribution fit)
    u            CDF of each observation under the fitted model
    params       dict of the fitted parameters by name
    n_params     number of free parameters (used by aicc)
    """
    beta: Optional[np.ndarray]
    error_model: object
    eval: Callable
    errors: np.ndarray
    u: np.ndarray
    params: dict
    n_params: int


# --------------------------------------------------------------------------
# 7.1 Normal
# --------------------------------------------------------------------------
def fit_normal(x) -> FittedModel:
    """MLE of the Normal is the sample mean; sd uses n-1 (matches Julia std)."""
    x = np.asarray(x, dtype=float)
    mu, sigma = x.mean(), x.std(ddof=1)
    d = stats.norm(loc=mu, scale=sigma)
    return FittedModel(
        beta=None, error_model=d, eval=lambda u: d.ppf(u),
        errors=x - mu, u=d.cdf(x),
        params={"mu": mu, "sigma": sigma}, n_params=2,
    )


# --------------------------------------------------------------------------
# 7.2 generalised t
# --------------------------------------------------------------------------
def _t_ll(mu, s, nu, x) -> float:
    """Sum log-likelihood of x under mu + s*t(nu)."""
    return np.sum(stats.t.logpdf(x, df=nu, loc=mu, scale=s))


def _t_start(e: np.ndarray) -> tuple[float, float, float]:
    """Moment-based starting values: excess kurtosis of a t(nu) is 6/(nu-4)."""
    start_m = e.mean()
    k = stats.kurtosis(e, fisher=True, bias=True)
    start_nu = 6.0 / k + 4.0 if k > 0 else 10.0
    start_nu = min(max(start_nu, 2.5), 50.0)
    start_s = np.sqrt(e.var(ddof=1) * (start_nu - 2.0) / start_nu)
    return start_m, start_s, start_nu


def _minimize(neg_ll, x0, bounds):
    """L-BFGS-B with tight tolerances, then a Nelder-Mead polish on the result."""
    res = optimize.minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000})
    res2 = optimize.minimize(neg_ll, res.x, method="Nelder-Mead",
                             options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
    # keep whichever landed lower, but respect the bounds
    best = res2 if res2.fun < res.fun else res
    lo = np.array([b[0] if b[0] is not None else -np.inf for b in bounds])
    hi = np.array([b[1] if b[1] is not None else np.inf for b in bounds])
    return np.clip(best.x, lo, hi)


def fit_general_t(x) -> FittedModel:
    """MLE of (mu, s, nu) for x = mu + s * t(nu), with s > 0 and nu > 2."""
    x = np.asarray(x, dtype=float)
    start_m, start_s, start_nu = _t_start(x)

    def neg_ll(p):
        return -_t_ll(p[0], p[1], p[2], x)

    mu, s, nu = _minimize(neg_ll, [start_m, start_s, start_nu],
                          bounds=[(None, None), (1e-6, None), (2.0001, None)])
    d = stats.t(df=nu, loc=mu, scale=s)
    return FittedModel(
        beta=None, error_model=d, eval=lambda u: d.ppf(u),
        errors=x - mu, u=d.cdf(x),
        params={"mu": mu, "sigma": s, "nu": nu}, n_params=3,
    )


# --------------------------------------------------------------------------
# 7.3 regression with t errors
# --------------------------------------------------------------------------
def fit_regression_t(y, x) -> FittedModel:
    """
    y = X b + e,  e ~ s * t(nu)  (error location fixed at 0, so the intercept
    absorbs it). X gets a column of ones prepended. Returns beta = [alpha, b1..].
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    n = x.shape[0]
    X = np.hstack([np.ones((n, 1)), x])
    n_b = X.shape[1]

    # starting values from OLS
    b_start = np.linalg.lstsq(X, y, rcond=None)[0]
    _, start_s, start_nu = _t_start(y - X @ b_start)

    def neg_ll(p):
        s, nu, beta = p[0], p[1], p[2:]
        return -_t_ll(0.0, s, nu, y - X @ beta)

    p = _minimize(neg_ll, np.r_[start_s, start_nu, b_start],
                  bounds=[(1e-6, None), (2.0001, None)] + [(None, None)] * n_b)
    s, nu, beta = p[0], p[1], p[2:]
    d = stats.t(df=nu, loc=0.0, scale=s)

    def eval_model(x_new, u):
        x_new = np.asarray(x_new, dtype=float)
        if x_new.ndim == 1:
            x_new = x_new[:, None]
        Xn = np.hstack([np.ones((x_new.shape[0], 1)), x_new])
        return Xn @ beta + d.ppf(u)

    errors = y - X @ beta
    params = {"mu": 0.0, "sigma": s, "nu": nu, "Alpha": beta[0]}
    params.update({f"B{i}": beta[i] for i in range(1, n_b)})
    return FittedModel(
        beta=beta, error_model=d, eval=eval_model,
        errors=errors, u=d.cdf(errors),
        params=params, n_params=3 + n_b,
    )


# --------------------------------------------------------------------------
# 7.4 AICc
# --------------------------------------------------------------------------
def aicc(model: FittedModel, data) -> float:
    """
    AICc = -2 ln L + 2k + 2k(k+1)/(n-k-1)

    k counts every parameter of the error distribution (location, scale, and
    shape where present) plus the regression betas if any.
    """
    data = np.asarray(data, dtype=float)
    n = data.shape[0]
    k = model.n_params
    ll = np.sum(model.error_model.logpdf(data))
    return -2.0 * ll + 2.0 * k + 2.0 * k * (k + 1) / (n - k - 1)


# --------------------------------------------------------------------------
# 7.5 / 7.6 Normal Inverse Gaussian
# --------------------------------------------------------------------------
def _nig_model(mu, alpha, beta, delta, x) -> FittedModel:
    d = stats.norminvgauss(a=alpha * delta, b=beta * delta, loc=mu, scale=delta)
    return FittedModel(
        beta=None, error_model=d, eval=lambda u: d.ppf(u),
        errors=x - d.mean(), u=d.cdf(x),
        params={"mu": mu, "alpha": alpha, "beta": beta, "delta": delta}, n_params=4,
    )


def fit_nig_moments(x) -> FittedModel:
    """
    Closed-form inversion of the first four moments of the NIG.

    With gamma = sqrt(alpha^2 - beta^2) and rho = beta/alpha:
        mean     = mu + delta*beta/gamma
        var      = delta*alpha^2/gamma^3
        skew     = 3 rho / sqrt(delta*gamma)
        ex.kurt  = 3 (1 + 4 rho^2) / (delta*gamma)
    so skew^2 / kurt = 3 rho^2 / (1 + 4 rho^2), which pins rho, then delta*gamma
    from the kurtosis, then alpha from the variance, and mu last.
    Skewness and kurtosis are the population (biased) estimators; variance is n-1.
    """
    x = np.asarray(x, dtype=float)
    m = x.mean()
    v = x.var(ddof=1)
    s = stats.skew(x, bias=True)
    k = stats.kurtosis(x, fisher=True, bias=True)

    if k <= 0:
        raise ValueError(f"NIG method of moments needs positive excess kurtosis, got {k}")
    t = s ** 2 / k
    if t >= 3.0 / 5.0:
        raise ValueError("sample is outside the NIG region: excess kurtosis must exceed (5/3)*skew^2")

    rho2 = t / (3.0 - 4.0 * t)
    rho = np.sign(s) * np.sqrt(rho2)
    dg = 3.0 * (1.0 + 4.0 * rho2) / k              # delta * gamma
    alpha = np.sqrt(dg / (v * (1.0 - rho2) ** 2))
    beta = rho * alpha
    gamma = alpha * np.sqrt(1.0 - rho2)
    delta = dg / gamma
    mu = m - delta * beta / gamma
    return _nig_model(mu, alpha, beta, delta, x)


def fit_nig_mle(x) -> FittedModel:
    """MLE via scipy.stats.norminvgauss.fit, converted to (mu, alpha, beta, delta)."""
    x = np.asarray(x, dtype=float)
    a, b, loc, scale = stats.norminvgauss.fit(x)
    delta = scale
    return _nig_model(mu=loc, alpha=a / delta, beta=b / delta, delta=delta, x=x)

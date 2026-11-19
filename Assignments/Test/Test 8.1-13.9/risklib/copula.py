"""
copula.py -- Gaussian and t copulas: density, fit, simulate, tail dependence.

  fit_gaussian_copula / fit_t_copula
  simulate_gaussian_copula / simulate_t_copula
  tail_dependence_t / copula_aicc / copula_bic
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2, multivariate_normal, multivariate_t, norm, t

from .multivariate_t import kendall_correlation, profile_nu
from .simulation import simulate_pca


def gaussian_copula_ll(u, r) -> float:
    z = norm.ppf(u)
    # clip for numerical safety at the edges
    z = np.clip(z, -8.0, 8.0)
    joint = multivariate_normal.logpdf(z, mean=np.zeros(r.shape[0]), cov=r).sum()
    margins = norm.logpdf(z).sum()
    return float(joint - margins)


def t_copula_ll(u, r, nu) -> float:
    tt = t.ppf(u, df=nu)
    tt = np.clip(tt, -1e6, 1e6)
    joint = multivariate_t(loc=np.zeros(r.shape[0]), shape=r, df=nu,
                           allow_singular=True).logpdf(tt).sum()
    margins = t.logpdf(tt, df=nu).sum()
    return float(joint - margins)


def fit_gaussian_copula(u):
    r = kendall_correlation(u)
    return r, gaussian_copula_ll(u, r)


def fit_t_copula(u):
    r = kendall_correlation(u)
    nu, ll = profile_nu(lambda nu: t_copula_ll(u, r, nu))
    return r, nu, ll


def simulate_gaussian_copula(r, nsim, seed: int = 1234):
    z = simulate_pca(r, nsim, seed=seed)
    return norm.cdf(z)


def simulate_t_copula(r, nu, nsim, seed: int = 1234):
    z = simulate_pca(r, nsim, seed=seed)
    rng = np.random.default_rng(seed + 1)
    w = nu / rng.chisquare(nu, size=nsim)
    return t.cdf(np.sqrt(w)[:, None] * z, df=nu)


def tail_dependence_t(rho, nu) -> float:
    """Lower (and upper, by symmetry) tail dependence of a bivariate t copula."""
    return float(2.0 * t.cdf(-np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho)),
                             df=nu + 1.0))


def copula_aicc(ll, k, m) -> float:
    return -2.0 * ll + 2.0 * k + (2.0 * k ** 2 + 2.0 * k) / (m - k - 1)


def copula_bic(ll, k, m) -> float:
    return k * np.log(m) - 2.0 * ll

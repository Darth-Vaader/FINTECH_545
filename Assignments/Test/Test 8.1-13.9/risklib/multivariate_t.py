"""
multivariate_t.py -- Kendall correlation, multivariate-t MLE by profile (Week 05).

  fix_correlation / kendall_correlation
  mvt_scale / mvt_loglikelihood / profile_nu / fit_multivariate_t
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import multivariate_t

from .psd import higham_nearest_psd


def fix_correlation(r, tol: float = 1e-8, ridge: float = 1e-8):
    """Repair a correlation that lost PSD under the entrywise sin(pi*tau/2)."""
    r = np.asarray(r, dtype=float)
    if np.linalg.eigvalsh(r).min() < -tol:
        r = np.asarray(higham_nearest_psd(r))
    r = 0.5 * (r + r.T)
    if np.linalg.eigvalsh(r).min() < ridge:
        n = r.shape[0]
        r = (r + ridge * np.eye(n)) / (1.0 + ridge)
    return r


def kendall_correlation(x):
    """rho_ij = sin(pi * tau_ij / 2), repaired to positive definite."""
    x = np.asarray(x, dtype=float)
    tau = pd.DataFrame(x).corr(method="kendall").to_numpy()
    r = np.sin(np.pi * tau / 2.0)
    np.fill_diagonal(r, 1.0)
    return fix_correlation(r)


def mvt_scale(sd, r, nu):
    """Scale matrix S such that cov = nu/(nu-2) * S."""
    d = sd * np.sqrt((nu - 2.0) / nu)
    return np.outer(d, d) * r


def mvt_loglikelihood(x, mu, sd, r, nu):
    s = mvt_scale(sd, r, nu)
    dist = multivariate_t(loc=mu, shape=s, df=nu, allow_singular=True)
    return float(np.sum(dist.logpdf(x)))


def profile_nu(ll, lo: float = 0.01, hi: float = 0.49, n: int = 200):
    """
    Maximise ll(nu) on a grid in theta = 1/nu ∈ [lo, hi], then refine around
    the peak. Returns (nu_hat, ll_hat).
    """
    thetas = np.linspace(lo, hi, n)
    lls = np.array([ll(1.0 / th) for th in thetas])
    i = int(np.argmax(lls))
    fine = np.linspace(thetas[max(i - 1, 0)], thetas[min(i + 1, n - 1)], n)
    fine_lls = np.array([ll(1.0 / th) for th in fine])
    j = int(np.argmax(fine_lls))
    return 1.0 / fine[j], float(fine_lls[j])


def fit_multivariate_t(x):
    """
    mu at the sample mean, R from Kendall, scales pinned to nu by the sample
    sds, nu by one-dimensional profile. Returns (mu, S, nu, ll).
    """
    x = np.asarray(x, dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0, ddof=1)
    r = kendall_correlation(x)
    nu, ll = profile_nu(lambda nu: mvt_loglikelihood(x, mu, sd, r, nu))
    return mu, mvt_scale(sd, r, nu), nu, ll

"""
simulation.py -- multivariate normal simulation (Week 03).

  simulate_normal  draws via the Cholesky root of the covariance (tests 5.1 - 5.4)
  simulate_pca     draws via a truncated eigen-decomposition (test 5.5)

Both return an (nsim x n) array: one row per simulated observation, so that
np.cov(out, rowvar=False) is directly comparable to the input covariance.
"""

from __future__ import annotations

import numpy as np

from .psd import chol_psd, near_psd


def _cov_array(cov) -> np.ndarray:
    arr = np.asarray(cov, dtype=float)
    n, m = arr.shape
    if n != m:
        raise ValueError(f"covariance matrix is not square ({n}, {m})")
    return arr


def _mean_vector(mean, n) -> np.ndarray:
    if mean is None or len(mean) == 0:
        return np.zeros(n)
    mean = np.asarray(mean, dtype=float)
    if mean.shape[0] != n:
        raise ValueError(f"mean ({mean.shape[0]}) is not the size of cov ({n}, {n})")
    return mean


def simulate_normal(nsim: int, cov, mean=None, seed: int = 1234, fix_method=near_psd):
    """
    X = L Z + mu, with Z ~ N(0, I) and L L' = cov.

    Root selection, in order:
      1. numpy Cholesky            -- works when cov is positive definite
      2. chol_psd                  -- works when cov is PSD with zero eigenvalues
      3. chol_psd(fix_method(cov)) -- cov is not PSD; repair first
    """
    cov_arr = _cov_array(cov)
    n = cov_arr.shape[0]
    mu = _mean_vector(mean, n)

    try:
        root = np.linalg.cholesky(cov_arr)
    except np.linalg.LinAlgError:
        try:
            root = np.asarray(chol_psd(cov_arr))
        except ValueError:
            root = np.asarray(chol_psd(np.asarray(fix_method(cov_arr))))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, nsim))
    return (root @ z).T + mu


def simulate_pca(cov, nsim: int, pct_exp: float = 1.0, mean=None, seed: int = 1234):
    """
    Keep the largest eigen-pairs until the cumulative share of total variance
    reaches `pct_exp`, then X = V Lambda^1/2 Z + mu with Z ~ N(0, I_k).

    Eigenvalues below 1e-8 are dropped regardless, so a PSD (or mildly non-PSD)
    input never produces a NaN root.
    """
    cov_arr = _cov_array(cov)
    n = cov_arr.shape[0]
    mu = _mean_vector(mean, n)

    vals, vecs = np.linalg.eigh(cov_arr)
    order = np.argsort(vals)[::-1]          # largest first
    vals, vecs = vals[order], vecs[:, order]

    total_var = vals.sum()
    keep = np.where(vals >= 1e-8)[0]
    if pct_exp < 1.0:
        cum = np.cumsum(vals[keep]) / total_var
        nval = int(np.searchsorted(cum, pct_exp) + 1)   # first index reaching pct_exp
        keep = keep[:nval]
    vals, vecs = vals[keep], vecs[:, keep]

    b = vecs * np.sqrt(vals)[None, :]
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((len(vals), nsim))
    return (b @ z).T + mu

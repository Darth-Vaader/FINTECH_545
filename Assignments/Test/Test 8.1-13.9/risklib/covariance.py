"""
covariance.py -- covariance / correlation estimators (Week 03).

  missing_cov      covariance or correlation when some observations are missing
                   (tests 1.1 - 1.4)
  ew_weights       exponentially decaying weights that sum to one
  ew_covar         exponentially weighted covariance (test 2.1)
  ew_correlation   exponentially weighted correlation (test 2.2)
  ew_cov_var_corr  EW covariance assembled from one lambda for the variances and
                   another for the correlations (test 2.3)
  cov_to_corr      helper: covariance -> correlation

Conventions
-----------
* Missing values are NaN. Sample covariance divides by n-1 (Week 01 §4.2);
  numpy's ddof=1 does that.
* Rows are observations, columns are variables. Row 0 is the OLDEST observation
  and the last row is the most recent; the EW weights are largest on the last row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_array(x) -> tuple[np.ndarray, list | None]:
    """Return (float ndarray, column labels or None)."""
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float), list(x.columns)
    return np.asarray(x, dtype=float), None


def _wrap(out: np.ndarray, labels):
    """Put labels back on a square matrix if the input had them."""
    if labels is None:
        return out
    return pd.DataFrame(out, index=labels, columns=labels)


def cov_to_corr(cov):
    """Correlation matrix from a covariance matrix: D^-1/2 Sigma D^-1/2."""
    cov_arr, labels = _as_array(cov)
    inv_sd = 1.0 / np.sqrt(np.diag(cov_arr))
    return _wrap(inv_sd[:, None] * cov_arr * inv_sd[None, :], labels)


def _cov_or_cor(x: np.ndarray, fun: str) -> np.ndarray:
    if fun == "cov":
        return np.cov(x, rowvar=False, ddof=1)
    if fun == "cor":
        return np.corrcoef(x, rowvar=False)
    raise ValueError(f"fun must be 'cov' or 'cor', got {fun!r}")


# --------------------------------------------------------------------------
# Test 1 -- missing data
# --------------------------------------------------------------------------
def missing_cov(x, skip_miss: bool = True, fun: str = "cov"):
    """
    Covariance (fun='cov') or correlation (fun='cor') of data with missing values.

    skip_miss=True  -> drop every row that has ANY missing value, then compute the
                       matrix on the complete rows (listwise deletion).
    skip_miss=False -> pairwise: each cell (i, j) is computed from the rows where
                       BOTH column i and column j are present. Different cells
                       therefore use different rows and the result need not be PSD
                       (which is why tests 3.x feed these matrices to near_psd).
    """
    arr, labels = _as_array(x)
    n, m = arr.shape
    is_missing = np.isnan(arr)

    # nothing missing -> plain estimator
    if not is_missing.any():
        return _wrap(_cov_or_cor(arr, fun), labels)

    if skip_miss:
        keep = ~is_missing.any(axis=1)
        return _wrap(_cov_or_cor(arr[keep], fun), labels)

    # pairwise
    out = np.empty((m, m))
    for i in range(m):
        for j in range(i + 1):
            keep = ~(is_missing[:, i] | is_missing[:, j])
            # 2-column sub-problem; take the off-diagonal. For i == j the two
            # columns are identical so cov -> variance and cor -> 1.
            out[i, j] = _cov_or_cor(arr[keep][:, [i, j]], fun)[0, 1]
            out[j, i] = out[i, j]
    return _wrap(out, labels)


# --------------------------------------------------------------------------
# Test 2 -- exponentially weighted covariance (Week 03 §5)
# --------------------------------------------------------------------------
def ew_weights(m: int, lam: float) -> np.ndarray:
    """
    w_i = (1 - lambda) * lambda^(m - i), i = 1..m, normalised to sum to one.
    The last observation (i = m) has the largest weight (1 - lambda).
    """
    i = np.arange(1, m + 1)
    w = (1.0 - lam) * lam ** (m - i)
    return w / w.sum()


def ew_covar(x, lam: float):
    """
    Exponentially weighted covariance:
        Sigma = sum_i w_i (x_i - mu_w)(x_i - mu_w)'
    where mu_w = sum_i w_i x_i is the weighted mean. No small-sample correction
    because the weights already sum to one.
    """
    arr, labels = _as_array(x)
    w = ew_weights(arr.shape[0], lam)
    xm = np.sqrt(w)[:, None] * (arr - w @ arr)   # sqrt(w) * demeaned rows
    return _wrap(xm.T @ xm, labels)


def ew_correlation(x, lam: float):
    """EW correlation: EW covariance rescaled to unit diagonal."""
    return cov_to_corr(ew_covar(x, lam))


def ew_cov_var_corr(x, lam_var: float, lam_corr: float):
    """
    Covariance built from EW variances at one lambda and EW correlations at
    another:  Sigma = D(lam_var)^1/2  R(lam_corr)  D(lam_var)^1/2.
    """
    arr, labels = _as_array(x)
    sd = np.sqrt(np.diag(_as_array(ew_covar(arr, lam_var))[0]))
    corr = _as_array(ew_correlation(arr, lam_corr))[0]
    return _wrap(sd[:, None] * corr * sd[None, :], labels)

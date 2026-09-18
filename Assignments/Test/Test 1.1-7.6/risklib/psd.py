"""
psd.py -- fixing non-PSD matrices and the PSD Cholesky root (Week 03).

  near_psd            Rebonato & Jackel eigenvalue clip + rescale (tests 3.1, 3.2)
  higham_nearest_psd  Higham's alternating projections (tests 3.3, 3.4)
  chol_psd            Cholesky root that tolerates zero eigenvalues (test 4.1)

Both repair functions accept either a covariance or a correlation matrix. If the
diagonal is not all ones they convert to correlation first, repair, and then put
the variances back, so the diagonal of the input is preserved exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_array(x):
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float), list(x.columns)
    return np.asarray(x, dtype=float), None


def _wrap(out, labels):
    return out if labels is None else pd.DataFrame(out, index=labels, columns=labels)


def _is_correlation(a: np.ndarray) -> bool:
    return np.allclose(np.diag(a), 1.0)


# --------------------------------------------------------------------------
# Rebonato & Jackel (Week 03 §7.1)
# --------------------------------------------------------------------------
def near_psd(a, epsilon: float = 0.0):
    """
    1. scale to correlation (if needed)
    2. eigen-decompose, floor the eigenvalues at `epsilon`
    3. rescale each row of the eigenvector matrix so the reconstructed matrix
       has a unit diagonal:  t_i = 1 / sum_j v_ij^2 lambda_j
    4. B = T^1/2 V Lambda^1/2 ;  out = B B'
    5. scale the variances back in
    """
    arr, labels = _as_array(a)
    out = arr.copy()

    inv_sd = None
    if not _is_correlation(out):
        inv_sd = 1.0 / np.sqrt(np.diag(out))
        out = inv_sd[:, None] * out * inv_sd[None, :]

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)
    t = 1.0 / ((vecs * vecs) @ vals)           # one scale factor per row
    b = np.sqrt(t)[:, None] * vecs * np.sqrt(vals)[None, :]
    out = b @ b.T

    if inv_sd is not None:
        sd = 1.0 / inv_sd
        out = sd[:, None] * out * sd[None, :]
    return _wrap(out, labels)


# --------------------------------------------------------------------------
# Higham (Week 03 §7.2)
# --------------------------------------------------------------------------
def _proj_psd(a: np.ndarray, w_half: np.ndarray) -> np.ndarray:
    """P_S: project onto the PSD cone in the W-norm (eigenvalue floor at 0)."""
    wa = w_half @ a @ w_half
    vals, vecs = np.linalg.eigh(wa)
    aplus = (vecs * np.maximum(vals, 0.0)) @ vecs.T
    iw = np.linalg.inv(w_half)
    return iw @ aplus @ iw


def _proj_unit_diag(a: np.ndarray) -> np.ndarray:
    """P_U: force the diagonal to one, leave everything else alone."""
    out = a.copy()
    np.fill_diagonal(out, 1.0)
    return out


def _wgt_norm(a: np.ndarray, w_half: np.ndarray) -> float:
    """Squared weighted Frobenius norm ||W^1/2 A W^1/2||_F^2."""
    wa = w_half @ a @ w_half
    return float(np.sum(wa * wa))


def higham_nearest_psd(a, weights=None, epsilon: float = 1e-9,
                       max_iter: int = 100, tol: float = 1e-9, verbose: bool = False):
    """
    Alternating projections with Dykstra's correction:
        R_k = Y_{k-1} - dS_{k-1}
        X_k = P_S(R_k)          (PSD projection)
        dS_k = X_k - R_k        (correction carried to the next step)
        Y_k = P_U(X_k)          (unit diagonal)
    Stop when the change in the weighted norm of (Y_k - Y_0) is below `tol`
    and the smallest eigenvalue is above -epsilon.
    """
    arr, labels = _as_array(a)
    n = arr.shape[0]
    w = np.eye(n) if weights is None else np.asarray(weights, dtype=float)
    w_half = np.sqrt(w)  # W is diagonal in practice

    yk = arr.copy()
    inv_sd = None
    if not _is_correlation(yk):
        inv_sd = 1.0 / np.sqrt(np.diag(yk))
        yk = inv_sd[:, None] * yk * inv_sd[None, :]

    y0 = yk.copy()
    delta_s = np.zeros_like(yk)
    norm_prev = np.inf
    i = 1
    while i <= max_iter:
        rk = yk - delta_s
        xk = _proj_psd(rk, w_half)
        delta_s = xk - rk
        yk = _proj_unit_diag(xk)

        norm = _wgt_norm(yk - y0, w_half)
        min_eig = np.linalg.eigvalsh(yk).min()
        if abs(norm - norm_prev) < tol and min_eig > -epsilon:
            break
        norm_prev = norm
        i += 1

    if verbose:
        if i < max_iter:
            print(f"Higham converged in {i} iterations.")
        else:
            print(f"Higham convergence failed after {i - 1} iterations.")

    if inv_sd is not None:
        sd = 1.0 / inv_sd
        yk = sd[:, None] * yk * sd[None, :]
    return _wrap(yk, labels)


# --------------------------------------------------------------------------
# Cholesky for PSD matrices (Week 03 §2.3)
# --------------------------------------------------------------------------
def chol_psd(a, epsilon: float = -1e-8):
    """
    Lower-triangular root L with L L' = a, allowing zero eigenvalues.

    Standard Cholesky, except that when the diagonal residual a_jj - sum l_jk^2
    lands in [epsilon, 0] it is treated as exactly zero (a zero eigenvalue) and
    the rest of that column is set to zero rather than dividing by zero. A
    residual below `epsilon` means the matrix is genuinely not PSD -> ValueError.
    """
    arr, labels = _as_array(a)
    n = arr.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = root[j, :j] @ root[j, :j]
        temp = arr[j, j] - s
        if epsilon <= temp <= 0.0:
            temp = 0.0
        if temp < 0.0:
            raise ValueError(f"matrix is not PSD: negative pivot {temp:.3e} at column {j}")
        root[j, j] = np.sqrt(temp)

        if root[j, j] == 0.0:
            # zero eigenvalue: nothing below the diagonal in this column
            continue
        ir = 1.0 / root[j, j]
        for i in range(j + 1, n):
            s = root[i, :j] @ root[j, :j]
            root[i, j] = (arr[i, j] - s) * ir
    return _wrap(root, labels)

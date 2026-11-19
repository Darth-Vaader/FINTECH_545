"""
optimizers.py -- risk parity and maximum Sharpe ratio (Week 06).

  risk_parity   equal (or budgeted) component-volatility weights
  max_sr        maximum Sharpe ratio under box constraints on the weights
"""

from __future__ import annotations

import numpy as np
from scipy import optimize


def risk_parity(cov, risk_budget=None):
    """
    Minimise the SSE of (component SD / risk_budget_i) across assets, subject
    to w >= 0 and sum(w) = 1. Component SD = w ⊙ (Σw) / σ_p.

    The objective is scaled by 1e5, matching the Julia Ipopt formulation so the
    optimiser lands on the same point.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if risk_budget is None:
        risk_budget = np.ones(n)
    else:
        risk_budget = np.asarray(risk_budget, dtype=float)
        if risk_budget.shape[0] != n:
            raise ValueError("risk_budget is not the correct size")
    mult = 1.0 / risk_budget

    def sse_csd(w):
        pvol = np.sqrt(w @ cov @ w)
        csd = w * (cov @ w) / pvol
        scaled = mult * csd
        d = scaled - scaled.mean()
        return 1.0e5 * float(d @ d)

    w0 = np.full(n, 1.0 / n)
    bounds = [(0.0, None)] * n
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    res = optimize.minimize(sse_csd, w0, method="SLSQP", bounds=bounds,
                            constraints=cons,
                            options={"ftol": 1e-14, "maxiter": 2000, "disp": False})
    w = np.maximum(res.x, 0.0)
    w = w / w.sum()
    return w, res.message


def max_sr(cov, mean, rf, bounds=None):
    """
    Maximise (w'μ - rf) / sqrt(w'Σw) subject to sum(w)=1 and box constraints.
    Default box is [0, 1] (long-only). `bounds` is an (n x 2) array of
    [lower, upper] per weight when supplied.

    Variance is floored at a tiny positive number so a non-PSD Σ (test 10.3's
    input) cannot send the Sharpe through a complex number mid-optimisation.
    """
    cov = np.asarray(cov, dtype=float)
    mean = np.asarray(mean, dtype=float)
    n = cov.shape[0]
    if bounds is None:
        bounds = np.column_stack([np.zeros(n), np.ones(n)])
    else:
        bounds = np.asarray(bounds, dtype=float)

    def neg_sr(w):
        vol2 = float(w @ cov @ w)
        vol = np.sqrt(max(vol2, 1e-18))
        return -(float(w @ mean) - rf) / vol

    w0 = np.full(n, 1.0 / n)
    # project the start onto the box
    w0 = np.clip(w0, bounds[:, 0], bounds[:, 1])
    w0 = w0 / w0.sum()

    box = [(float(bounds[i, 0]), float(bounds[i, 1])) for i in range(n)]
    cons = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    res = optimize.minimize(neg_sr, w0, method="SLSQP", bounds=box,
                            constraints=cons,
                            options={"ftol": 1e-14, "maxiter": 2000, "disp": False})
    return res.x, res.message

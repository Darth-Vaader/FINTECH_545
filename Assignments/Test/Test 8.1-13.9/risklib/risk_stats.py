"""
risk_stats.py -- VaR, Expected Shortfall, and portfolio aggregation (Week 04/05).

  VaR(a) / VaR(dist)   Value at Risk at level alpha (default 5%), as a positive loss
  ES(a)  / ES(dist)    Expected Shortfall at the same level
  agg_risk             per-group and Total VaR / ES from a simulated PnL table

Sign convention (Week 04): both metrics are reported as positive numbers for a
loss. For a distribution that is the return itself, VaR = -F^{-1}(alpha).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import integrate


def VaR(a, alpha: float = 0.05) -> float:
    """
    Sample VaR: -average of the floor(n*alpha) and ceil(n*alpha) order
    statistics (1-based), matching the Julia RiskStats.jl convention.
    Distribution VaR: -quantile(d, alpha).
    """
    if hasattr(a, "ppf"):
        return float(-a.ppf(alpha))
    x = np.sort(np.asarray(a, dtype=float).ravel())
    n = x.size
    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))
    # Julia is 1-based; floor can be 0 only for tiny n*alpha, which we don't hit.
    v = 0.5 * (x[nup - 1] + x[ndn - 1])
    return float(-v)


def ES(a, alpha: float = 0.05) -> float:
    """
    Sample ES: -mean of observations at or below the VaR threshold (same sort
    as VaR, so the threshold is identical).
    Distribution ES: -1/alpha * integral_{-inf}^{-VaR} x f(x) dx, via quad.
    """
    if hasattr(a, "ppf") and hasattr(a, "pdf"):
        v = VaR(a, alpha=alpha)
        lo = float(a.ppf(1e-12))
        hi = -v
        integral, _ = integrate.quad(lambda x: x * a.pdf(x), lo, hi, limit=200)
        return float(-integral / alpha)

    x = np.sort(np.asarray(a, dtype=float).ravel())
    n = x.size
    nup = int(np.ceil(n * alpha))
    ndn = int(np.floor(n * alpha))
    v = 0.5 * (x[nup - 1] + x[ndn - 1])
    return float(-x[x <= v].mean())


def agg_risk(pnl_by_stock: dict, current_value: dict) -> pd.DataFrame:
    """
    Build the VaR95 / ES95 / Pct table that Julia's aggRisk produces for a
    single grouping column (Stock) plus the Total row.

    pnl_by_stock: {stock_name: 1-d array of dollar PnL across iterations}
    current_value: {stock_name: starting dollar value}
    """
    rows = []
    total_pnl = None
    total_value = 0.0
    for stock, pnl in pnl_by_stock.items():
        pnl = np.asarray(pnl, dtype=float)
        cv = float(current_value[stock])
        var95 = VaR(pnl, alpha=0.05)
        es95 = ES(pnl, alpha=0.05)
        rows.append({
            "Stock": stock,
            "VaR95": var95,
            "ES95": es95,
            "VaR95_Pct": var95 / cv,
            "ES95_Pct": es95 / cv,
        })
        total_pnl = pnl if total_pnl is None else total_pnl + pnl
        total_value += cv

    var95 = VaR(total_pnl, alpha=0.05)
    es95 = ES(total_pnl, alpha=0.05)
    rows.append({
        "Stock": "Total",
        "VaR95": var95,
        "ES95": es95,
        "VaR95_Pct": var95 / total_value,
        "ES95_Pct": es95 / total_value,
    })
    return pd.DataFrame(rows)

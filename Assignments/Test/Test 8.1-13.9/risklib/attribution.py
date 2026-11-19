"""
attribution.py -- Carino ex-post return and volatility attribution (Week 07).

  expost_factor(w, stock_returns, factor_returns, betas)
      -> DataFrame with rows TotalReturn / Return Attribution / Vol Attribution
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expost_factor(w, stock_returns: pd.DataFrame, factor_returns: pd.DataFrame,
                 betas: np.ndarray) -> pd.DataFrame:
    """
    Walk the weights forward through the return history, attribute each period's
    portfolio return to the factors (and an Alpha residual) with Carino's
    linking coefficient, then attribute realised portfolio volatility by
    regressing the weighted factor contributions on the portfolio return.
    """
    stocks = list(stock_returns.columns)
    factors = list(factor_returns.columns)
    mat_r = stock_returns[stocks].to_numpy(dtype=float)
    ff_r = factor_returns[factors].to_numpy(dtype=float)
    betas = np.asarray(betas, dtype=float)
    w = np.asarray(w, dtype=float)

    n = mat_r.shape[0]
    p_return = np.empty(n)
    resid_return = np.empty(n)
    weights = np.empty((n, len(w)))
    factor_weights = np.empty((n, len(factors)))
    last_w = w.copy()

    for i in range(n):
        weights[i] = last_w
        # factor exposure of the portfolio this period
        factor_weights[i] = (betas * last_w[:, None]).sum(axis=0)
        last_w = last_w * (1.0 + mat_r[i])
        p_r = last_w.sum()
        last_w = last_w / p_r
        p_return[i] = p_r - 1.0
        resid_return[i] = p_return[i] - factor_weights[i] @ ff_r[i]

    # Carino linking
    total_ret = float(np.exp(np.log1p(p_return).sum()) - 1.0)
    k = np.log1p(total_ret) / total_ret
    carino_k = np.log1p(p_return) / p_return / k

    attrib = ff_r * factor_weights * carino_k[:, None]
    alpha_attrib = resid_return * carino_k

    # TotalReturn row: cumulative return of each factor series, of Alpha, of Portfolio
    # Return Attribution row: sum of linked contributions (Portfolio column = total_ret)
    # Vol Attribution row: component SD from the B * sd(pReturn) regression
    up = pd.DataFrame(ff_r, columns=factors)
    up["Alpha"] = resid_return
    up["Portfolio"] = p_return

    new_factors = factors + ["Alpha"]
    attribution = {"Value": ["TotalReturn", "Return Attribution"]}
    for s in new_factors + ["Portfolio"]:
        tr = float(np.exp(np.log1p(up[s].to_numpy()).sum()) - 1.0)
        atr = float(attrib[:, factors.index(s)].sum()) if s in factors else (
            float(alpha_attrib.sum()) if s == "Alpha" else tr
        )
        attribution[s] = [tr, atr]

    # Volatility attribution: Y = [weighted factor returns | residual], X = [1, pReturn]
    y = np.column_stack([ff_r * factor_weights, resid_return])
    x = np.column_stack([np.ones(n), p_return])
    b = np.linalg.lstsq(x, y, rcond=None)[0][1]   # slope on pReturn only
    c_sd = b * p_return.std(ddof=1)

    vol_row = {"Value": "Vol Attribution"}
    for i, s in enumerate(new_factors):
        vol_row[s] = float(c_sd[i])
    vol_row["Portfolio"] = float(p_return.std(ddof=1))

    out = pd.DataFrame(attribution)
    out = pd.concat([out, pd.DataFrame([vol_row])], ignore_index=True)
    return out

"""
returns.py -- price series to return series (tests 6.1, 6.2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def return_calculate(prices: pd.DataFrame, method: str = "DISCRETE",
                     date_column: str = "date") -> pd.DataFrame:
    """
    Convert a price DataFrame (one date column + one column per asset) into
    returns.

      DISCRETE (arithmetic):  r_t = P_t / P_{t-1} - 1
      LOG:                    r_t = ln(P_t / P_{t-1})

    The first date is dropped (no prior price), so the result has one fewer row.
    """
    if date_column not in prices.columns:
        raise ValueError(f"date_column {date_column!r} not in DataFrame: {list(prices.columns)}")

    asset_cols = [c for c in prices.columns if c != date_column]
    p = prices[asset_cols].to_numpy(dtype=float)
    ratio = p[1:] / p[:-1]

    method = method.upper()
    if method == "DISCRETE":
        r = ratio - 1.0
    elif method == "LOG":
        r = np.log(ratio)
    else:
        raise ValueError(f"method {method!r} must be 'LOG' or 'DISCRETE'")

    out = pd.DataFrame(r, columns=asset_cols)
    out.insert(0, date_column, prices[date_column].iloc[1:].to_numpy())
    return out

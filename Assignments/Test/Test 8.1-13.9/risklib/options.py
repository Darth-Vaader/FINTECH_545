"""
options.py -- GBSM European and CRR American binomial trees (Week 08/09).

  gbsm(call, S, K, T, rf, b, ivol, include_greeks=False)
  bt_american(...)     continuous-dividend (cost-of-carry b) form
  bt_american_div(...) discrete cash-dividend form
  american_greeks_fd   finite-difference Greeks matching the Julia FiniteDiff
                       convention used to generate testout12_2.csv
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class GBSM:
    value: float
    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    c_rho: float = 0.0


def gbsm(call: bool, underlying, strike, ttm, rf, b, ivol,
         include_greeks: bool = False) -> GBSM:
    """
    Generalised Black-Scholes-Merton. Cost-of-carry b specialises the model:
      b = rf          Black-Scholes 1973
      b = rf - q      Merton 1973 with continuous dividend yield q
      b = 0           Black 1976 futures
    """
    sqrt_t = np.sqrt(ttm)
    d1 = (np.log(underlying / strike) + (b + 0.5 * ivol ** 2) * ttm) / (ivol * sqrt_t)
    d2 = d1 - ivol * sqrt_t
    e_brt = np.exp((b - rf) * ttm)
    e_rt = np.exp(-rf * ttm)

    if call:
        delta = e_brt * norm.cdf(d1)
        value = underlying * delta - strike * e_rt * norm.cdf(d2)
    else:
        delta = e_brt * (norm.cdf(d1) - 1.0)
        value = strike * e_rt * norm.cdf(-d2) - underlying * e_brt * norm.cdf(-d1)

    out = GBSM(value=float(value), delta=float(delta))
    if not include_greeks:
        return out

    out.gamma = float(norm.pdf(d1) * e_brt / (underlying * ivol * sqrt_t))
    out.vega = float(underlying * e_brt * norm.pdf(d1) * sqrt_t)
    if call:
        out.theta = float(
            -underlying * e_brt * norm.pdf(d1) * ivol / (2 * sqrt_t)
            - (b - rf) * underlying * e_brt * norm.cdf(d1)
            - rf * strike * e_rt * norm.cdf(d2)
        )
        out.rho = float(ttm * strike * e_rt * norm.cdf(d2))
        out.c_rho = float(ttm * underlying * e_brt * norm.cdf(d1))
    else:
        out.theta = float(
            -underlying * e_brt * norm.pdf(d1) * ivol / (2 * sqrt_t)
            + (b - rf) * underlying * e_brt * norm.cdf(-d1)
            + rf * strike * e_rt * norm.cdf(-d2)
        )
        out.rho = float(-ttm * strike * e_rt * norm.cdf(-d2))
        out.c_rho = float(-ttm * underlying * e_brt * norm.cdf(-d1))
    return out


# --------------------------------------------------------------------------
# CRR American binomial -- continuous cost of carry (layer form, O(n^2))
# --------------------------------------------------------------------------
def bt_american(call: bool, underlying, strike, ttm, rf, b, ivol, n: int) -> float:
    """CRR tree with n steps. Early exercise checked at every node."""
    dt = ttm / n
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(b * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1.0 if call else -1.0

    # terminal layer: i = 0..n up-moves
    i = np.arange(n + 1)
    price = underlying * (u ** i) * (d ** (n - i))
    values = np.maximum(0.0, z * (price - strike))

    for j in range(n - 1, -1, -1):
        values = df * (pu * values[1:] + pd * values[:-1])
        i = np.arange(j + 1)
        price = underlying * (u ** i) * (d ** (j - i))
        values = np.maximum(values, z * (price - strike))
    return float(values[0])


# --------------------------------------------------------------------------
# CRR American binomial -- discrete cash dividends
# --------------------------------------------------------------------------
def bt_american_div(call: bool, underlying, strike, ttm, rf,
                    div_amts, div_times, ivol, n: int) -> float:
    """
    Discrete-dividend American. `div_times` are integer step indices on the
    n-step grid. At the first dividend date the tree hands off to a recursive
    call on the ex-dividend underlying with the remaining dividends shifted.
    """
    div_amts = list(div_amts)
    div_times = [int(t) for t in div_times]
    if not div_amts or not div_times or div_times[0] > n:
        return bt_american(call, underlying, strike, ttm, rf, rf, ivol, n)

    first = div_times[0]
    dt = ttm / n
    u = np.exp(ivol * np.sqrt(dt))
    d = 1.0 / u
    pu = (np.exp(rf * dt) - d) / (u - d)
    pd = 1.0 - pu
    df = np.exp(-rf * dt)
    z = 1.0 if call else -1.0

    rem_amts = div_amts[1:]
    rem_times = [t - first for t in div_times[1:]]
    rem_n = n - first
    rem_ttm = ttm - first * dt

    # values at the dividend date: recurse on the ex-div tree for each node
    i = np.arange(first + 1)
    price = underlying * (u ** i) * (d ** (first - i))
    values = np.empty(first + 1)
    for k in range(first + 1):
        val_no = bt_american_div(
            call, price[k] - div_amts[0], strike, rem_ttm,
            rf, rem_amts, rem_times, ivol, rem_n,
        )
        values[k] = max(val_no, max(0.0, z * (price[k] - strike)))

    # roll back from the dividend date to today
    for j in range(first - 1, -1, -1):
        values = df * (pu * values[1:] + pd * values[:-1])
        i = np.arange(j + 1)
        price = underlying * (u ** i) * (d ** (j - i))
        values = np.maximum(values, z * (price - strike))
    return float(values[0])


# --------------------------------------------------------------------------
# Finite-difference Greeks for the continuous-dividend American
# --------------------------------------------------------------------------
def _fd_eps(x: float) -> float:
    """FiniteDiff.jl central-difference step: cbrt(eps) * max(1, |x|)."""
    return np.cbrt(np.finfo(float).eps) * max(1.0, abs(x))


def american_greeks_fd(call: bool, underlying, strike, ttm, rf, b, ivol, n: int = 500):
    """
    Value plus FD Greeks. Gradient via central differences on
    (S, K, T, rf, b, ivol); gamma via a fixed step of 1.5 on S, matching the
    Julia generation script.
    """
    parms = np.array([underlying, strike, ttm, rf, b, ivol], dtype=float)

    def f(p):
        return bt_american(call, p[0], p[1], p[2], p[3], p[4], p[5], n)

    v = f(parms)
    grad = np.empty(6)
    for i in range(6):
        e = _fd_eps(parms[i])
        pp, pm = parms.copy(), parms.copy()
        pp[i] += e
        pm[i] -= e
        grad[i] = (f(pp) - f(pm)) / (2 * e)

    d = 1.5
    gamma = (f(parms + np.array([d, 0, 0, 0, 0, 0]))
             + f(parms + np.array([-d, 0, 0, 0, 0, 0]))
             - 2 * v) / (d ** 2)

    return {
        "value": v,
        "delta": grad[0],
        "gamma": gamma,
        "vega": grad[5],
        "rho": grad[3],
        "theta": grad[2],   # ∂V/∂T as stored in testout12_2; not calendar theta
    }

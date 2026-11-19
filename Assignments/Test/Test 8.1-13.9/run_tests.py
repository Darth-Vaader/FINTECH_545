"""
run_tests.py -- FinTech 545 functional tests 8.1 through 13.9.

    python run_tests.py            # run everything
    python run_tests.py 8.1 13.7   # run selected tests

Tests 1.1–7.6 live in the sibling folder ../Test 1.1-7.6/.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

import risklib as rl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

NSIM = 100_000
NSIM_VAR = 100_000


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, name))


def matrix(name: str) -> np.ndarray:
    return load(name).to_numpy(dtype=float)


def row(**kwargs) -> pd.DataFrame:
    return pd.DataFrame({k: [v] for k, v in kwargs.items()})


# --------------------------------------------------------------------------
# tests 8 -- VaR / ES
# --------------------------------------------------------------------------
def _t_df_scale(d):
    from scipy.stats import t as t_dist
    df = d.args[0] if d.args else d.kwds["df"]
    scale = d.kwds.get("scale", 1.0)
    return df, scale, t_dist


def t8_1():
    fd = rl.fit_normal(matrix("test7_1.csv")[:, 0])
    d = fd.error_model
    return row(**{
        "VaR Absolute": rl.VaR(d),
        "VaR Diff from Mean": -norm(0, d.std()).ppf(0.05),
    })


def t8_2():
    fd = rl.fit_general_t(matrix("test7_2.csv")[:, 0])
    d = fd.error_model
    df, scale, t_dist = _t_df_scale(d)
    return row(**{
        "VaR Absolute": rl.VaR(d),
        "VaR Diff from Mean": -t_dist(df=df, loc=0.0, scale=scale).ppf(0.05),
    })


def t8_3():
    fd = rl.fit_general_t(matrix("test7_2.csv")[:, 0])
    rng = np.random.default_rng(8)
    sim = fd.eval(rng.random(NSIM_VAR))
    return row(**{
        "VaR Absolute": rl.VaR(sim),
        "VaR Diff from Mean": rl.VaR(sim - sim.mean()),
    })


def t8_4():
    fd = rl.fit_normal(matrix("test7_1.csv")[:, 0])
    d = fd.error_model
    return row(**{
        "ES Absolute": rl.ES(d),
        "ES Diff from Mean": rl.ES(norm(0, d.std())),
    })


def t8_5():
    fd = rl.fit_general_t(matrix("test7_2.csv")[:, 0])
    d = fd.error_model
    df, scale, t_dist = _t_df_scale(d)
    return row(**{
        "ES Absolute": rl.ES(d),
        "ES Diff from Mean": rl.ES(t_dist(df=df, loc=0.0, scale=scale)),
    })


def t8_6():
    fd = rl.fit_general_t(matrix("test7_2.csv")[:, 0])
    rng = np.random.default_rng(8)
    sim = fd.eval(rng.random(NSIM_VAR))
    return row(**{
        "ES Absolute": rl.ES(sim),
        "ES Diff from Mean": rl.ES(sim - sim.mean()),
    })


# --------------------------------------------------------------------------
# test 9 -- Gaussian copula portfolio VaR/ES
# --------------------------------------------------------------------------
def _portfolio_risk(sim_ret: pd.DataFrame, portfolio: pd.DataFrame) -> pd.DataFrame:
    pnl, cv = {}, {}
    for _, r in portfolio.iterrows():
        s = r["Stock"]
        cv[s] = float(r["currentValue"])
        pnl[s] = cv[s] * sim_ret[s].to_numpy(dtype=float)
    return rl.agg_risk(pnl, cv)


def t9_1():
    cin = load("test9_1_returns.csv")
    models = {"A": rl.fit_normal(cin["A"].to_numpy()),
              "B": rl.fit_general_t(cin["B"].to_numpy())}
    u = np.column_stack([models["A"].u, models["B"].u])
    spcor = pd.DataFrame(u).corr(method="spearman").to_numpy()
    u_sim = norm.cdf(rl.simulate_pca(spcor, NSIM, seed=1234))
    sim_ret = pd.DataFrame({
        "A": models["A"].eval(u_sim[:, 0]),
        "B": models["B"].eval(u_sim[:, 1]),
    })
    portfolio = pd.DataFrame({"Stock": ["A", "B"], "currentValue": [2000.0, 3000.0]})
    return _portfolio_risk(sim_ret, portfolio)


# --------------------------------------------------------------------------
# tests 10 -- risk parity / max Sharpe
# --------------------------------------------------------------------------
def t10_1():
    w, _ = rl.risk_parity(matrix("test5_2.csv"))
    return pd.DataFrame({"W": w})


def t10_2():
    w, _ = rl.risk_parity(matrix("test5_2.csv"), risk_budget=[1, 1, 1, 1, 0.5])
    return pd.DataFrame({"W": w})


def t10_3():
    # test_setup.jl uses test5_3.csv (non-PSD), not test5_2 as Tests.xlsx says
    means = load("test10_3_means.csv")["Mean"].to_numpy()
    w, _ = rl.max_sr(matrix("test5_3.csv"), means, rf=0.04)
    return pd.DataFrame({"W": w})


def t10_4():
    means = load("test10_3_means.csv")["Mean"].to_numpy()
    bounds = np.column_stack([np.full(5, 0.1), np.full(5, 0.5)])
    w, _ = rl.max_sr(matrix("test5_3.csv"), means, rf=0.04, bounds=bounds)
    return pd.DataFrame({"W": w})


# --------------------------------------------------------------------------
# tests 11 -- ex-post attribution
# --------------------------------------------------------------------------
def t11_1():
    returns = load("test11_1_returns.csv")
    w = load("test11_1_weights.csv")["W"].to_numpy()
    attr = rl.expost_factor(w, returns, returns, np.eye(3))
    return attr.drop(columns=["Alpha"])


def t11_2():
    w = load("test11_2_weights.csv")["W"].to_numpy()
    stock = load("test11_2_stock_returns.csv")
    factors = load("test11_2_factor_returns.csv")
    beta = load("test11_2_beta.csv").drop(columns=["Stock"]).to_numpy()
    return rl.expost_factor(w, stock, factors, beta)


# --------------------------------------------------------------------------
# tests 12 -- options
# --------------------------------------------------------------------------
def t12_1():
    opts = load("test12_1.csv").dropna(subset=["ID"])
    rows = []
    for _, o in opts.iterrows():
        g = rl.gbsm(
            o["Option Type"] == "Call",
            o["Underlying"], o["Strike"],
            o["DaysToMaturity"] / o["DayPerYear"],
            o["RiskFreeRate"],
            o["RiskFreeRate"] - o["DividendRate"],
            o["ImpliedVol"],
            include_greeks=True,
        )
        rows.append({"ID": o["ID"], "Value": g.value, "Delta": g.delta,
                     "Gamma": g.gamma, "Vega": g.vega, "Rho": g.rho, "Theta": g.theta})
    return pd.DataFrame(rows)


def t12_2():
    opts = load("test12_1.csv").dropna(subset=["ID"])
    rows = []
    for _, o in opts.iterrows():
        call = o["Option Type"] == "Call"
        b = o["RiskFreeRate"] - o["DividendRate"]
        g = rl.american_greeks_fd(
            call, o["Underlying"], o["Strike"],
            o["DaysToMaturity"] / o["DayPerYear"],
            o["RiskFreeRate"], b, o["ImpliedVol"], n=500,
        )
        rows.append({"ID": o["ID"], "Value": g["value"], "Delta": g["delta"],
                     "Gamma": g["gamma"], "Vega": g["vega"], "Rho": g["rho"],
                     "Theta": g["theta"]})
    return pd.DataFrame(rows)


def t12_3():
    opts = load("test12_3.csv").dropna(subset=["ID"])
    rows = []
    for _, o in opts.iterrows():
        div_dates = [int(v) for v in str(o["DividendDates"]).split(",")]
        div_amts = [float(v) for v in str(o["DividendAmts"]).split(",")]
        n = int(o["DaysToMaturity"] * 2)
        div_dates = [d * 2 for d in div_dates]
        v = rl.bt_american_div(
            o["Option Type"] == "Call",
            o["Underlying"], o["Strike"],
            o["DaysToMaturity"] / o["DayPerYear"],
            o["RiskFreeRate"],
            div_amts, div_dates,
            o["ImpliedVol"], n,
        )
        rows.append({"ID": o["ID"], "Value": v})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# tests 13 -- multivariate t / t copula
# --------------------------------------------------------------------------
def t13_1():
    return pd.DataFrame(rl.kendall_correlation(matrix("test13_returns.csv")))


def t13_2():
    mu, *_ = rl.fit_multivariate_t(matrix("test13_returns.csv"))
    return pd.DataFrame({"mu": mu})


def t13_3():
    _, s, *_ = rl.fit_multivariate_t(matrix("test13_returns.csv"))
    return pd.DataFrame(s)


def t13_4():
    *_, nu, ll = rl.fit_multivariate_t(matrix("test13_returns.csv"))
    return row(nu=nu, ll=ll)


def _fitted_margins_and_u():
    x = matrix("test13_returns.csv")
    models = [rl.fit_general_t(x[:, j]) for j in range(x.shape[1])]
    u = np.column_stack([m.u for m in models])
    return models, u


def t13_5():
    _, u = _fitted_margins_and_u()
    _, ll = rl.fit_gaussian_copula(u)
    return row(ll=ll)


def t13_6():
    _, u = _fitted_margins_and_u()
    _, nu, ll = rl.fit_t_copula(u)
    return row(nu=nu, ll=ll)


def t13_7():
    _, u = _fitted_margins_and_u()
    _, ll_g = rl.fit_gaussian_copula(u)
    _, _, ll_t = rl.fit_t_copula(u)
    n_obs = u.shape[0]
    return pd.DataFrame({
        "Copula": ["Gaussian", "T"],
        "LL": [ll_g, ll_t],
        "K": [0, 1],
        "AICC": [rl.copula_aicc(ll_g, 0, n_obs), rl.copula_aicc(ll_t, 1, n_obs)],
        "BIC": [rl.copula_bic(ll_g, 0, n_obs), rl.copula_bic(ll_t, 1, n_obs)],
    })


def t13_8():
    _, u = _fitted_margins_and_u()
    rc, nu_c, _ = rl.fit_t_copula(u)
    n = rc.shape[0]
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "I": i + 1, "J": j + 1,
                "Rho": rc[i, j],
                "Lambda": rl.tail_dependence_t(rc[i, j], nu_c),
            })
    return pd.DataFrame(rows)


def t13_9():
    x = matrix("test13_returns.csv")
    names = list(load("test13_returns.csv").columns)
    models = [rl.fit_general_t(x[:, j]) for j in range(x.shape[1])]
    u = np.column_stack([m.u for m in models])
    rc, nu_c, _ = rl.fit_t_copula(u)
    sim_u = rl.simulate_t_copula(rc, nu_c, NSIM, seed=13)
    sim_ret = pd.DataFrame(
        {names[j]: models[j].eval(sim_u[:, j]) for j in range(len(names))}
    )
    return _portfolio_risk(sim_ret, load("test13_portfolio.csv"))


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
EXACT, OPTIM, SIM = 1e-8, 1e-4, 1.5e-1
OPTIM_LOOSE = 5e-3

ANALYTICAL_CHECK = {
    "8.3": ("8.2", 1e-1),
    "8.6": ("8.5", 1e-1),
}

TESTS = {
    "8.1": (t8_1, "testout8_1.csv", EXACT),
    "8.2": (t8_2, "testout8_2.csv", OPTIM),
    "8.3": (t8_3, "testout8_3.csv", SIM),
    "8.4": (t8_4, "testout8_4.csv", EXACT),
    "8.5": (t8_5, "testout8_5.csv", OPTIM),
    "8.6": (t8_6, "testout8_6.csv", SIM),
    "9.1": (t9_1, "testout9_1.csv", SIM),
    "10.1": (t10_1, "testout10_1.csv", OPTIM_LOOSE),
    "10.2": (t10_2, "testout10_2.csv", OPTIM_LOOSE),
    "10.3": (t10_3, "testout10_3.csv", OPTIM_LOOSE),
    "10.4": (t10_4, "testout10_4.csv", OPTIM_LOOSE),
    "11.1": (t11_1, "testout11_1.csv", EXACT),
    "11.2": (t11_2, "testout11_2.csv", EXACT),
    "12.1": (t12_1, "testout12_1.csv", EXACT),
    "12.2": (t12_2, "testout12_2.csv", OPTIM_LOOSE),
    "12.3": (t12_3, "testout12_3.csv", EXACT),
    "13.1": (t13_1, "testout13_1.csv", EXACT),
    "13.2": (t13_2, "testout13_2.csv", EXACT),
    "13.3": (t13_3, "testout13_3.csv", EXACT),
    "13.4": (t13_4, "testout13_4.csv", OPTIM),
    "13.5": (t13_5, "testout13_5.csv", OPTIM),
    "13.6": (t13_6, "testout13_6.csv", OPTIM),
    "13.7": (t13_7, "testout13_7.csv", OPTIM),
    "13.8": (t13_8, "testout13_8.csv", OPTIM),
    "13.9": (t13_9, "testout13_9.csv", SIM),
}


def compare(got: pd.DataFrame, exp: pd.DataFrame, rtol) -> tuple[bool, str]:
    if list(got.columns) != list(exp.columns) and got.shape == exp.shape:
        got = got.copy()
        got.columns = exp.columns
    if list(got.columns) != list(exp.columns):
        return False, f"columns differ: {list(got.columns)} vs {list(exp.columns)}"
    if got.shape != exp.shape:
        return False, f"shape differs: {got.shape} vs {exp.shape}"

    ok = True
    notes = []
    for c in exp.columns:
        if pd.api.types.is_numeric_dtype(exp[c]):
            g = got[c].to_numpy(dtype=float)
            e = exp[c].to_numpy(dtype=float)
            if not np.allclose(g, e, rtol=rtol, atol=max(1e-8, rtol * 1e-2)):
                ok = False
        else:
            if not (got[c].astype(str).to_numpy() == exp[c].astype(str).to_numpy()).all():
                ok = False
                notes.append(f"non-numeric column {c!r} differs")

    num = [c for c in exp.columns if pd.api.types.is_numeric_dtype(exp[c])]
    g = got[num].to_numpy(dtype=float)
    e = exp[num].to_numpy(dtype=float)
    abs_diff = float(np.nanmax(np.abs(g - e))) if g.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(g - e) / np.maximum(np.abs(e), 1e-15)
        rel = rel[np.isfinite(rel)]
    rel_diff = float(rel.max()) if rel.size else 0.0
    notes.insert(0, f"max abs diff {abs_diff:.3e}, max rel diff {rel_diff:.3e}")
    return ok, "; ".join(notes)


def run(names):
    width = max(len(n) for n in names)
    n_pass = 0
    for name in names:
        fn, exp_file, rtol = TESTS[name]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = fn()
            got = got if isinstance(got, pd.DataFrame) else pd.DataFrame(got)
            exp = load(exp_file)
            if list(got.columns) != list(exp.columns) and got.shape == exp.shape:
                got.columns = exp.columns
            for c in got.columns:
                if c in exp.columns and pd.api.types.is_numeric_dtype(exp[c]):
                    got[c] = pd.to_numeric(got[c], errors="coerce")
            got.to_csv(os.path.join(OUT, exp_file), index=False)
            ok, note = compare(got, exp, rtol)
            if name in ANALYTICAL_CHECK:
                ref_name, ref_tol = ANALYTICAL_CHECK[name]
                ref = TESTS[ref_name][0]()
                g = got.select_dtypes(include=[np.number]).to_numpy(float)
                r = ref.select_dtypes(include=[np.number]).to_numpy(float)
                rel = np.max(np.abs(g - r) / np.maximum(np.abs(r), 1e-15))
                ok = bool(rel <= ref_tol)
                note += f"; vs analytical {ref_name} max rel {rel:.3e}"
        except Exception as exc:                # noqa: BLE001
            ok, note = False, f"ERROR {type(exc).__name__}: {exc}"
        n_pass += ok
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}  {note}")
    print(f"\n{n_pass}/{len(names)} passed")
    return n_pass == len(names)


if __name__ == "__main__":
    selected = sys.argv[1:] or list(TESTS)
    unknown = [s for s in selected if s not in TESTS]
    if unknown:
        sys.exit(f"unknown test(s): {unknown}. Available: {list(TESTS)}")
    sys.exit(0 if run(selected) else 1)

"""
run_tests.py -- FinTech 545 functional tests 1.1 through 7.6.

    python run_tests.py            # run everything
    python run_tests.py 2.1 3.3    # run selected tests

For each test: read the input CSV from data/, run the risklib function, write our
result to output/<same name as the expected file>, and compare against the
instructor's data/testout*.csv. Prints PASS / FAIL with the largest absolute and
relative difference.

Tolerances
----------
Deterministic calculations (1.x, 2.x, 3.x, 4.1, 6.x, 7.1, 7.5) are compared at
1e-8 relative. Numerical optimisers (7.2, 7.3, 7.4, 7.6) at 1e-4 relative: a
different optimiser lands within that of the same maximum.

Monte Carlo tests (5.x) cannot match the expected file digit for digit because
the random draws differ. The sample covariance of N draws from a normal has
standard error  se_ij = sqrt((s_ii s_jj + s_ij^2) / N)  per entry, so each entry
is required to sit within MC_SIGMAS standard errors of the expected value
(multiplied by sqrt(2), as both files are independent Monte Carlo estimates),
and within MC_SIGMAS standard errors of the covariance it was meant to
reproduce. A relative tolerance would be the wrong yardstick here: entries the
input sets to (near) zero have unbounded relative error however good the draw.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

import risklib as rl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

NSIM = 100_000


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(DATA, name))


def matrix(name: str) -> np.ndarray:
    return load(name).to_numpy(dtype=float)


def labelled(arr, like: pd.DataFrame) -> pd.DataFrame:
    """Square matrix -> DataFrame with the same column headers as `like`."""
    return pd.DataFrame(np.asarray(arr), columns=like.columns)


def row(**kwargs) -> pd.DataFrame:
    """One-row DataFrame, e.g. row(mu=..., sigma=...)."""
    return pd.DataFrame({k: [float(v)] for k, v in kwargs.items()})


# --------------------------------------------------------------------------
# the tests: name -> (function producing a DataFrame, expected file, rtol)
# --------------------------------------------------------------------------
def t1_1():
    x = load("test1.csv")
    return rl.missing_cov(x, skip_miss=True, fun="cov")

def t1_2():
    x = load("test1.csv")
    return rl.missing_cov(x, skip_miss=True, fun="cor")

def t1_3():
    x = load("test1.csv")
    return rl.missing_cov(x, skip_miss=False, fun="cov")

def t1_4():
    x = load("test1.csv")
    return rl.missing_cov(x, skip_miss=False, fun="cor")

def t2_1():
    return rl.ew_covar(load("test2.csv"), 0.97)

def t2_2():
    return rl.ew_correlation(load("test2.csv"), 0.94)

def t2_3():
    return rl.ew_cov_var_corr(load("test2.csv"), lam_var=0.97, lam_corr=0.94)

def t3_1():
    return rl.near_psd(load("testout_1.3.csv"))

def t3_2():
    return rl.near_psd(load("testout_1.4.csv"))

def t3_3():
    return rl.higham_nearest_psd(load("testout_1.3.csv"))

def t3_4():
    return rl.higham_nearest_psd(load("testout_1.4.csv"))

def t4_1():
    return rl.chol_psd(load("testout_3.1.csv"))

def _sim_cov(sim, like):
    return labelled(np.cov(sim, rowvar=False, ddof=1), like)

def t5_1():
    cin = load("test5_1.csv")
    return _sim_cov(rl.simulate_normal(NSIM, cin), cin)

def t5_2():
    cin = load("test5_2.csv")
    return _sim_cov(rl.simulate_normal(NSIM, cin), cin)

def t5_3():
    cin = load("test5_3.csv")
    return _sim_cov(rl.simulate_normal(NSIM, cin, fix_method=rl.near_psd), cin)

def t5_4():
    cin = load("test5_3.csv")
    return _sim_cov(rl.simulate_normal(NSIM, cin, fix_method=rl.higham_nearest_psd), cin)

def t5_5():
    cin = load("test5_2.csv")
    return _sim_cov(rl.simulate_pca(cin, NSIM, pct_exp=0.99), cin)

def t6_1():
    return rl.return_calculate(load("test6.csv"), method="DISCRETE", date_column="Date")

def t6_2():
    return rl.return_calculate(load("test6.csv"), method="LOG", date_column="Date")

def t7_1():
    fd = rl.fit_normal(matrix("test7_1.csv")[:, 0])
    return row(mu=fd.params["mu"], sigma=fd.params["sigma"])

def t7_2():
    fd = rl.fit_general_t(matrix("test7_2.csv")[:, 0])
    return row(mu=fd.params["mu"], sigma=fd.params["sigma"], nu=fd.params["nu"])

def t7_3():
    cin = load("test7_3.csv")
    fd = rl.fit_regression_t(cin["y"], cin.drop(columns="y"))
    return row(**fd.params)

def t7_4():
    x = matrix("test7_2.csv")[:, 0]
    fd = rl.fit_general_t(x)
    return row(AICC=rl.aicc(fd, x))

def t7_5():
    fd = rl.fit_nig_moments(matrix("test7_5.csv")[:, 0])
    return row(**fd.params)

def t7_6():
    fd = rl.fit_nig_mle(matrix("test7_5.csv")[:, 0])
    return row(**fd.params)


EXACT, OPTIM, MONTE_CARLO = 1e-8, 1e-4, "mc"
MC_SIGMAS = 5.0   # allowed distance in standard errors for Monte Carlo tests

TESTS = {
    "1.1": (t1_1, "testout_1.1.csv", EXACT),
    "1.2": (t1_2, "testout_1.2.csv", EXACT),
    "1.3": (t1_3, "testout_1.3.csv", EXACT),
    "1.4": (t1_4, "testout_1.4.csv", EXACT),
    "2.1": (t2_1, "testout_2.1.csv", EXACT),
    "2.2": (t2_2, "testout_2.2.csv", EXACT),
    "2.3": (t2_3, "testout_2.3.csv", EXACT),
    "3.1": (t3_1, "testout_3.1.csv", EXACT),
    "3.2": (t3_2, "testout_3.2.csv", EXACT),
    "3.3": (t3_3, "testout_3.3.csv", EXACT),
    "3.4": (t3_4, "testout_3.4.csv", EXACT),
    "4.1": (t4_1, "testout_4.1.csv", EXACT),
    "5.1": (t5_1, "testout_5.1.csv", MONTE_CARLO),
    "5.2": (t5_2, "testout_5.2.csv", MONTE_CARLO),
    "5.3": (t5_3, "testout_5.3.csv", MONTE_CARLO),
    "5.4": (t5_4, "testout_5.4.csv", MONTE_CARLO),
    "5.5": (t5_5, "testout_5.5.csv", MONTE_CARLO),
    "6.1": (t6_1, "testout6_1.csv", EXACT),
    "6.2": (t6_2, "testout6_2.csv", EXACT),
    "7.1": (t7_1, "testout7_1.csv", EXACT),
    "7.2": (t7_2, "testout7_2.csv", OPTIM),
    "7.3": (t7_3, "testout7_3.csv", OPTIM),
    "7.4": (t7_4, "testout7_4.csv", OPTIM),
    "7.5": (t7_5, "testout7_5.csv", EXACT),
    "7.6": (t7_6, "testout7_6.csv", OPTIM),
}

# For the Monte Carlo tests, the covariance the simulation is meant to reproduce.
# 5.3 / 5.4 target the REPAIRED matrix, 5.5 targets the 99%-variance truncation,
# so only 5.1 and 5.2 should reproduce their input exactly.
def _mc_target(name):
    if name == "5.1":
        return matrix("test5_1.csv")
    if name == "5.2":
        return matrix("test5_2.csv")
    if name == "5.3":
        return np.asarray(rl.near_psd(matrix("test5_3.csv")))
    if name == "5.4":
        return np.asarray(rl.higham_nearest_psd(matrix("test5_3.csv")))
    return None


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
def cov_std_err(cov: np.ndarray, nsim: int) -> np.ndarray:
    """Standard error of each entry of a sample covariance from nsim normal draws."""
    d = np.diag(cov)
    return np.sqrt((np.outer(d, d) + cov ** 2) / nsim)


def compare_mc(got: np.ndarray, ref: np.ndarray, nsim: int, both_mc: bool) -> tuple[bool, float]:
    """
    Is every entry of `got` within MC_SIGMAS standard errors of `ref`?
    both_mc=True when `ref` is itself a simulation, which doubles the variance.
    Returns (ok, largest distance in standard errors).
    """
    se = cov_std_err(ref, nsim) * (np.sqrt(2.0) if both_mc else 1.0)
    z = np.abs(got - ref) / se
    return bool(np.all(z <= MC_SIGMAS)), float(z.max())


def compare(got: pd.DataFrame, exp: pd.DataFrame, rtol) -> tuple[bool, str]:
    if rtol == MONTE_CARLO:
        g, e = got.to_numpy(dtype=float), exp.to_numpy(dtype=float)
        ok, z = compare_mc(g, e, NSIM, both_mc=True)
        return ok, f"max abs diff {np.abs(g - e).max():.3e} = {z:.2f} std errors vs expected"

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
            if not np.allclose(g, e, rtol=rtol, atol=1e-12):
                ok = False
        else:                                  # e.g. the Date column
            if not (got[c].astype(str).to_numpy() == exp[c].astype(str).to_numpy()).all():
                ok = False
                notes.append(f"non-numeric column {c!r} differs")

    num = [c for c in exp.columns if pd.api.types.is_numeric_dtype(exp[c])]
    g = got[num].to_numpy(dtype=float)
    e = exp[num].to_numpy(dtype=float)
    abs_diff = np.nanmax(np.abs(g - e)) if g.size else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(g - e) / np.abs(e)
        rel = rel[np.isfinite(rel)]
    rel_diff = rel.max() if rel.size else 0.0
    notes.insert(0, f"max abs diff {abs_diff:.3e}, max rel diff {rel_diff:.3e}")
    return ok, "; ".join(notes)


def run(names):
    width = max(len(n) for n in names)
    n_pass = 0
    for name in names:
        fn, exp_file, rtol = TESTS[name]
        try:
            got = fn()
            got = got if isinstance(got, pd.DataFrame) else pd.DataFrame(got)
            exp = load(exp_file)
            if list(got.columns) != list(exp.columns) and got.shape == exp.shape:
                # matrix outputs carry the input's labels; align to expected names
                got.columns = exp.columns
            got.to_csv(os.path.join(OUT, exp_file), index=False)
            ok, note = compare(got, exp, rtol)
            target = _mc_target(name)
            if target is not None:
                ok_t, z = compare_mc(got.to_numpy(float), target, NSIM, both_mc=False)
                ok = ok and ok_t
                note += f"; {z:.2f} std errors vs target cov"
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

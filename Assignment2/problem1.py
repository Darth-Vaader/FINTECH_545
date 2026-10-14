"""
Problem 1 -- Correlations from mismatched histories.

problem1.csv holds 250 days of returns for A, B, C, D and an index IDX built from
them with weights 0.40 / 0.30 / 0.20 / 0.10 plus a small tracking error. Blank
cells are days an asset did not trade. We

  predict  : tabulate joint observation counts, bound the smallest eigenvalue of
             the TRUE correlation matrix from the tracking error, and rank the
             pairs by how noisy their correlation estimate must be;
  fit      : complete-case vs pairwise correlation, eigenvalues, Cholesky; a
             covariance from pairwise correlations + full-history sd; the
             variance of the long-IDX / short-basket portfolio; Rebonato-Jackel
             and Higham repairs;
  reconcile: read the negative variance through the PSD definition, locate
             where Higham moved the matrix and why, and compare the size of
             the repair with the estimator gap.

Run:  python problem1.py
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
from common import C1, C2, C3

SERIES = ["A", "B", "C", "D", "IDX"]
BASKET_W = np.array([0.40, 0.30, 0.20, 0.10])
# Long one unit of IDX, short the weighted basket: the TRACKING portfolio.
TRACK_W = np.r_[-BASKET_W, 1.0]


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
def part_predict(d: pd.DataFrame) -> dict:
    common.header("Problem 1 -- Predict: who trades when")
    obs = d[SERIES].notna()
    joint = (obs.astype(int).T @ obs.astype(int))          # days both observed
    n_complete = int(obs.all(axis=1).sum())

    print("  Observations per series:")
    for s in SERIES:
        first = int(d.loc[obs[s], "Day"].iloc[0])
        print(f"    {s:<4} {int(obs[s].sum()):>4} days   (first traded day {first})")
    print(f"\n  Days on which all five trade (complete rows): {n_complete} of {len(d)}")
    print("\n  Jointly observed days, pair by pair:")
    common.print_matrix(joint.values, SERIES, fmt="{:>9d}")

    # Where the holes are, because the two calendars fail in different ways:
    # C has scattered single-day holidays; D simply does not exist before day 151.
    c_missing = d.loc[~obs["C"], "Day"].astype(int).tolist()
    print(f"\n  C's holidays ({len(c_missing)} days):")
    print(f"    {c_missing[:11]}")
    print(f"    {c_missing[11:]}")
    print(f"  D's first day: {int(d.loc[obs['D'], 'Day'].iloc[0])}; "
          f"before that every D cell is blank.")

    # (b) How close to singular is the TRUE correlation matrix? IDX is the basket
    # plus a tracking error, so the portfolio long IDX / short the basket has a
    # tiny variance. In standardised units that portfolio is v = D w, and the
    # Rayleigh quotient v'Cv / v'v bounds the smallest eigenvalue from above:
    #   lambda_min(C) <= Var(tracking error) / sum_i (w_i sigma_i)^2.
    # This uses only the construction of IDX and each series' own sd -- no
    # correlation has been computed yet.
    full = d.dropna(subset=SERIES)
    te = full["IDX"].to_numpy() - full[["A", "B", "C", "D"]].to_numpy() @ BASKET_W
    sd = d[SERIES].std()                                     # full-history sd
    v = TRACK_W * sd.to_numpy()
    bound = te.var(ddof=1) / (v @ v)
    print("\n  (b) IDX against its own basket, on the complete rows:")
    print(f"      sd of (IDX - basket) = {te.std(ddof=1):.5f}  vs  sd of IDX = {sd['IDX']:.5f}")
    print(f"      => tracking error is {100 * te.std(ddof=1) / sd['IDX']:.1f}% of the index's "
          f"own volatility")
    print(f"      Rayleigh bound on the smallest eigenvalue of the TRUE correlation")
    print(f"      matrix: lambda_min <= Var(TE) / sum (w_i sigma_i)^2 = {bound:.5f}")
    print(f"      A 5x5 correlation matrix has eigenvalues summing to 5; one of them")
    print(f"      is at most {bound:.4f}. The true matrix is a hair from singular.")

    # (c) Sampling noise of a correlation from n paired days is about (1-rho^2)
    # /sqrt(n) <= 1/sqrt(n). Rank the pairs by joint n.
    print("\n  (c) Pairs ranked by jointly observed days (fewest first), with the")
    print("      worst-case standard error 1/sqrt(n) of a correlation built on them:")
    rows = []
    for a, b in itertools.combinations(SERIES, 2):
        n = int(joint.loc[a, b])
        rows.append((f"{a}-{b}", n, 1 / np.sqrt(n)))
    rows.sort(key=lambda r: r[1])
    print(f"      {'pair':<8}{'n':>6}{'1/sqrt(n)':>12}")
    for name, n, se in rows:
        print(f"      {name:<8}{n:>6}{se:>12.3f}")
    print(f"      The noise on a D pair (~{1 / np.sqrt(joint.loc['C', 'D']):.2f}) is "
          f"roughly {1 / np.sqrt(joint.loc['C', 'D']) / bound:.0f}x the room the true")
    print(f"      matrix has before an eigenvalue goes negative ({bound:.4f}).")

    return {"joint": joint, "n_complete": n_complete, "sd": sd, "te": te,
            "bound": bound, "obs": obs}


# --------------------------------------------------------------------------
# Fit (d): both correlation estimators, eigenvalues, Cholesky
# --------------------------------------------------------------------------
def pairwise_corr(d: pd.DataFrame) -> np.ndarray:
    """Pairwise-complete Pearson correlation: cell (i, j) is computed on the rows
    where BOTH i and j are present, with means and sds taken on those same rows
    (Week 03 s3.3 method 2). Different cells use different rows, which is why
    the result is not guaranteed PSD."""
    x = d[SERIES].to_numpy(dtype=float)
    m = len(SERIES)
    out = np.eye(m)
    for i in range(m):
        for j in range(i):
            keep = ~(np.isnan(x[:, i]) | np.isnan(x[:, j]))
            out[i, j] = out[j, i] = np.corrcoef(x[keep, i], x[keep, j])[0, 1]
    return out


def complete_corr(d: pd.DataFrame) -> np.ndarray:
    """Listwise deletion: drop every row with ANY blank, correlate what is left
    (Week 03 s3.3 method 1). A Gram matrix of standardised complete data, so PSD."""
    return np.corrcoef(d[SERIES].dropna().to_numpy(dtype=float), rowvar=False)


def try_cholesky(c: np.ndarray, label: str) -> None:
    """numpy's PD Cholesky, then the course's PSD-tolerant version, which names
    the pivot that goes negative."""
    try:
        np.linalg.cholesky(c)
        print(f"    {label:<14} numpy cholesky : OK (matrix is positive definite)")
    except np.linalg.LinAlgError as e:
        print(f"    {label:<14} numpy cholesky : FAILS ({e})")
    try:
        common.chol_psd(c)
        print(f"    {'':<14} chol_psd       : OK")
    except ValueError as e:
        print(f"    {'':<14} chol_psd       : FAILS -- {e}")


def part_d(d: pd.DataFrame) -> dict:
    common.header("Problem 1(d) -- complete-case vs pairwise correlation")
    c_cc = complete_corr(d)
    c_pw = pairwise_corr(d)
    e_cc = np.linalg.eigvalsh(c_cc)
    e_pw = np.linalg.eigvalsh(c_pw)

    common.print_matrix(c_cc, SERIES, title=f"Complete case ({len(d.dropna())} rows):")
    print()
    common.print_matrix(c_pw, SERIES, title="Pairwise (each cell on its own rows):")
    print("\n  Eigenvalues (ascending):")
    print(f"    complete case : {np.array2string(e_cc, precision=4, floatmode='fixed')}")
    print(f"    pairwise      : {np.array2string(e_pw, precision=4, floatmode='fixed')}")
    print(f"    smallest: complete {e_cc[0]:+.5f}   pairwise {e_pw[0]:+.5f}")
    print(f"    as a fraction of the largest eigenvalue: complete {e_cc[0] / e_cc[-1]:+.1e},")
    print(f"    pairwise {e_pw[0] / e_pw[-1]:+.1e}. Week 03 s3.6: -1e-16 of the largest is")
    print(f"    arithmetic; -3.5e-3 of the largest is a statement about the data.")
    print("\n  Cholesky factorisation attempts:")
    try_cholesky(c_cc, "complete case")
    try_cholesky(c_pw, "pairwise")
    return {"c_cc": c_cc, "c_pw": c_pw, "e_cc": e_cc, "e_pw": e_pw}


# --------------------------------------------------------------------------
# Fit (e): covariance from pairwise correlation + full-history sd; tracking var
# --------------------------------------------------------------------------
def covariance_from(corr: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Sigma = D C D, Week 03 s3.4 item 7: correlations from whatever estimator,
    variances from the FULL history of each series."""
    return np.outer(sd, sd) * corr


def tracking_variance(cov: np.ndarray) -> float:
    """Fact 7 (Week 02 s2.1.1) in matrix form: var(w'X) = w' Sigma w."""
    return float(TRACK_W @ cov @ TRACK_W)


def part_e(P: dict, D: dict) -> dict:
    common.header("Problem 1(e) -- covariance from pairwise corr + full-history sd")
    sd = P["sd"].to_numpy()
    print("  Full-history standard deviation of each series (its own n):")
    for s, v, n in zip(SERIES, sd, P["obs"].sum()):
        print(f"    {s:<4} sd = {v:.6f}   (n = {int(n)})")
    cov_pw = covariance_from(D["c_pw"], sd)
    cov_cc = covariance_from(D["c_cc"], sd)
    var_pw = tracking_variance(cov_pw)
    var_cc = tracking_variance(cov_cc)
    print(f"\n  Tracking portfolio w = {TRACK_W.tolist()}  (long IDX, short basket)")
    print(f"    w' Sigma w  with PAIRWISE correlations      = {var_pw:+.4e}")
    print(f"    w' Sigma w  with COMPLETE-CASE correlations = {var_cc:+.4e}")
    print(f"    realised var of (IDX - basket), complete rows = {P['te'].var(ddof=1):+.4e}")
    if var_pw < 0:
        print(f"\n    The pairwise number is NEGATIVE: sd would be sqrt({var_pw:.3e}), "
              f"undefined.")
        print(f"    A portfolio that really exists has been assigned a variance below 0.")
    # How far the tracking direction is from the offending eigenvector.
    vals, vecs = np.linalg.eigh(D["c_pw"])
    v = TRACK_W * sd
    cos = abs(v @ vecs[:, 0]) / np.linalg.norm(v)
    print(f"\n    In standardised units the tracking portfolio is v = D w; the cosine")
    print(f"    between v and the eigenvector of the negative eigenvalue is {cos:.3f}.")
    print(f"    The portfolio the question asks about IS the direction in which the")
    print(f"    pairwise matrix fails.")
    return {"cov_pw": cov_pw, "cov_cc": cov_cc, "var_pw": var_pw, "var_cc": var_cc,
            "cos": cos}


# --------------------------------------------------------------------------
# Fit (f): repairs
# --------------------------------------------------------------------------
def part_f(P: dict, D: dict) -> dict:
    common.header("Problem 1(f) -- repairing the pairwise matrix")
    c_pw = D["c_pw"]
    sd = P["sd"].to_numpy()
    c_rj = common.near_psd(c_pw)
    c_hi, iters = common.higham_nearest_corr(c_pw)
    out = {}
    print(f"  {'method':<20}{'min eig':>12}{'||C - C_pw||_F':>16}{'tracking var':>15}")
    print("  " + "-" * 63)
    for name, c in [("pairwise (input)", c_pw), ("Rebonato-Jackel", c_rj),
                    ("Higham", c_hi)]:
        e = np.linalg.eigvalsh(c)[0]
        fd = common.frob(c, c_pw)
        tv = tracking_variance(covariance_from(c, sd))
        out[name] = {"corr": c, "min_eig": e, "frob": fd, "track_var": tv}
        print(f"  {name:<20}{e:>+12.2e}{fd:>16.5f}{tv:>+15.3e}")
    print("  " + "-" * 63)
    print(f"  Higham converged in {iters} iterations (W = I, tol 1e-9).")
    print(f"\n  Cholesky on the repaired matrices. Both repairs FLOOR the bad eigenvalue")
    print(f"  at zero rather than lifting it above zero, so the result is PSD but")
    print(f"  singular. numpy's strictly-PD routine still refuses it; the Week 03")
    print(f"  s3.2.2 zero-column patch (chol_psd) factors it fine:")
    try_cholesky(c_rj, "Rebonato-Jackel")
    try_cholesky(c_hi, "Higham")
    rj_sd = np.sqrt(max(out['Rebonato-Jackel']['track_var'], 0))
    hi_sd = np.sqrt(max(out['Higham']['track_var'], 0))
    print(f"\n  Tracking sd after repair: RJ {rj_sd:.5f}, Higham {hi_sd:.5f}; realised "
          f"{P['te'].std(ddof=1):.5f}.")
    print(f"  The repaired matrices put a zero eigenvalue almost exactly along the")
    print(f"  tracking direction, so the variance that is left comes from the small")
    print(f"  part of v = D w that leaks into the other eigenvectors. Small, positive,")
    print(f"  and the same order as the realised tracking variance -- usable.")
    out["iters"] = iters
    return out


# --------------------------------------------------------------------------
# Reconcile (g), (h), (i)
# --------------------------------------------------------------------------
def part_ghi(P: dict, D: dict, E: dict, F: dict) -> pd.DataFrame:
    common.header("Problem 1(g) -- the negative variance, read through the PSD definition")
    print("  Week 02 s2.2.1: Sigma is PSD iff w' Sigma w >= 0 for every w, because")
    print("  w' Sigma w IS the variance of the portfolio w'X and a variance cannot be")
    print(f"  negative. The pairwise Sigma gives w' Sigma w = {E['var_pw']:+.3e} for the")
    print("  tracking portfolio, so it fails the definition at a portfolio that exists.")
    print("  The eigenvalue view says the same thing: the smallest eigenvalue is")
    print(f"  {D['e_pw'][0]:+.4f}, and the eigenvector it belongs to points (cosine "
          f"{E['cos']:.2f}) along")
    print("  the tracking portfolio. The assembled entries are mutually inconsistent --")
    print("  the index is correlated with each stock a little MORE than its basket")
    print("  weights and the stock-stock correlations can jointly support.")

    common.header("Problem 1(h) -- where Higham moved the matrix")
    c_pw, c_hi = D["c_pw"], F["Higham"]["corr"]
    c_rj = F["Rebonato-Jackel"]["corr"]
    vals, vecs = np.linalg.eigh(c_pw)
    v0 = vecs[:, 0]                        # eigenvector of the negative eigenvalue
    rows = []
    for i, j in itertools.combinations(range(5), 2):
        rows.append({"pair": f"{SERIES[i]}-{SERIES[j]}",
                     "n": int(P["joint"].iloc[i, j]),
                     "pairwise": c_pw[i, j],
                     "Higham": c_hi[i, j],
                     "dHigham": c_hi[i, j] - c_pw[i, j],
                     "dRJ": c_rj[i, j] - c_pw[i, j],
                     "v_i*v_j": v0[i] * v0[j],
                     "complete": D["c_cc"][i, j],
                     "gap_cc_pw": D["c_cc"][i, j] - c_pw[i, j]})
    tbl = pd.DataFrame(rows).sort_values("dHigham", key=np.abs, ascending=False)
    print("  Entries sorted by |Higham change|. v_i*v_j is the outer product of the")
    print("  negative eigenvector -- the direction a rank-one correction must take:")
    print(f"  {'pair':<8}{'n':>5}{'pairwise':>10}{'Higham':>9}{'change':>9}"
          f"{'RJ chg':>9}{'v_i*v_j':>9}{'complete':>10}{'cc-pw':>8}")
    for _, r in tbl.iterrows():
        print(f"  {r['pair']:<8}{r['n']:>5}{r['pairwise']:>10.4f}{r['Higham']:>9.4f}"
              f"{r['dHigham']:>+9.4f}{r['dRJ']:>+9.4f}{r['v_i*v_j']:>+9.3f}"
              f"{r['complete']:>10.4f}{r['gap_cc_pw']:>+8.3f}")
    corr_sign = np.corrcoef(tbl["dHigham"], tbl["v_i*v_j"])[0, 1]
    corr_n = np.corrcoef(tbl["dHigham"].abs(), tbl["n"])[0, 1]
    print(f"\n  corr(Higham change, v_i*v_j)  = {corr_sign:+.3f}   <- the repair follows")
    print(f"  corr(|Higham change|, n)      = {corr_n:+.3f}   the eigenvector, not n")
    top = tbl.iloc[0]
    print(f"  Largest move: {top['pair']} ({top['dHigham']:+.4f}, built on n = {top['n']}).")
    print("  The repair knows nothing about sample sizes. With W = I every entry is")
    print("  equally cheap to move, so it finds the nearest PSD matrix in the plain")
    print("  Frobenius norm: shrink the component along the negative eigenvector")
    print("  until that eigenvalue reaches zero. The entries that load most on that")
    print("  eigenvector (the IDX row, and the stock pairs that must offset it) move")
    print("  the most, whether or not they were the noisy ones.")

    common.header("Problem 1(i) -- size of the repair vs size of the estimator gap")
    f_hi = F["Higham"]["frob"]
    f_rj = F["Rebonato-Jackel"]["frob"]
    f_est = common.frob(D["c_cc"], c_pw)
    f_hi_rj = common.frob(c_hi, c_rj)
    off = ~np.eye(5, dtype=bool)
    print(f"  ||Higham - pairwise||_F           = {f_hi:.5f}   "
          f"(max entry {np.abs(c_hi - c_pw)[off].max():.4f})")
    print(f"  ||RJ - pairwise||_F               = {f_rj:.5f}   "
          f"(max entry {np.abs(c_rj - c_pw)[off].max():.4f})")
    print(f"  ||Higham - RJ||_F                 = {f_hi_rj:.5f}")
    print(f"  ||complete-case - pairwise||_F    = {f_est:.5f}   "
          f"(max entry {np.abs(D['c_cc'] - c_pw)[off].max():.4f})")
    print(f"  ratio estimator gap / Higham repair = {f_est / f_hi:.1f}x")
    print(f"\n  Tracking variance under each matrix (x 1e-6):")
    for name, val in [("pairwise", E["var_pw"]), ("Rebonato-Jackel", F["Rebonato-Jackel"]["track_var"]),
                      ("Higham", F["Higham"]["track_var"]), ("complete case", E["var_cc"]),
                      ("realised (81 rows)", P["te"].var(ddof=1))]:
        print(f"    {name:<20}{val * 1e6:>+10.3f}")
    return tbl


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def figure(d: pd.DataFrame, P: dict, D: dict, F: dict, tbl: pd.DataFrame) -> None:
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0),
                           gridspec_kw={"width_ratios": [1.5, 1.0, 1.15]})

    # Panel 1: trading calendar raster.
    obs = P["obs"].to_numpy().T.astype(float)
    ax[0].imshow(obs, aspect="auto", cmap="Blues", vmin=-0.3, vmax=1.2,
                 interpolation="nearest", extent=[0.5, len(d) + 0.5, 4.5, -0.5])
    ax[0].set_yticks(range(5))
    ax[0].set_yticklabels(SERIES)
    ax[0].set_xlabel("day")
    ax[0].grid(False)
    ax[0].set_title(f"Who trades when ({P['n_complete']} complete days of {len(d)})")
    ax[0].axvline(150.5, color=C2, linestyle="--", linewidth=1.2)
    ax[0].annotate("D lists", xy=(151, 3), xytext=(60, 3.0), color=C2, fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", color=C2, lw=1))

    # Panel 2: eigenvalues of each matrix.
    mats = [("complete", D["c_cc"], C3), ("pairwise", D["c_pw"], C2),
            ("R&J", F["Rebonato-Jackel"]["corr"], C1), ("Higham", F["Higham"]["corr"], common.INK2)]
    for k, (name, c, col) in enumerate(mats):
        e = np.linalg.eigvalsh(c)
        ax[1].scatter([k] * 5, e, color=col, s=28, zorder=3)
        ax[1].annotate(f"{e[0]:+.4f}", xy=(k, e[0]), xytext=(0, -12),
                       textcoords="offset points", ha="center", fontsize=7.5, color=col)
    ax[1].axhline(0, color=common.INK, linewidth=0.8)
    ax[1].set_xticks(range(4))
    ax[1].set_xticklabels([m[0] for m in mats])
    ax[1].set_yscale("symlog", linthresh=0.01)
    ax[1].set_ylabel("eigenvalue (symlog)")
    ax[1].set_title("Eigenvalues: only pairwise is < 0")

    # Panel 3: Higham change per entry, annotated with joint n.
    delta = F["Higham"]["corr"] - D["c_pw"]
    lim = np.abs(delta).max()
    im = ax[2].imshow(delta, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax[2].set_xticks(range(5)); ax[2].set_yticks(range(5))
    ax[2].set_xticklabels(SERIES); ax[2].set_yticklabels(SERIES)
    ax[2].grid(False)
    for i in range(5):
        for j in range(5):
            if i != j:
                ax[2].text(j, i, f"{delta[i, j]:+.3f}\nn={int(P['joint'].iloc[i, j])}",
                           ha="center", va="center", fontsize=7, color=common.INK)
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03)
    ax[2].set_title("Higham minus pairwise (n = joint days)")

    fig.suptitle("Problem 1 -- a pairwise correlation matrix that assigns a negative "
                 "variance to a real portfolio", fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem1")


def main() -> dict:
    d = common.load("problem1")
    P = part_predict(d)
    D = part_d(d)
    E = part_e(P, D)
    F = part_f(P, D)
    tbl = part_ghi(P, D, E, F)
    figure(d, P, D, F, tbl)
    return {"P": P, "D": D, "E": E, "F": F, "tbl": tbl}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

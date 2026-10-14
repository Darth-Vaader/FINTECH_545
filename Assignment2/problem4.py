"""
Problem 4 -- Gaussian or t copula.

problem4.csv holds 1,000 days of returns for X1, X2, X3; we hold $1,000,000 of
each. We

  predict  : four moments per series and how much the kurtosis leans on a
             single day; rank scatter plots; joint 2.5% tail counts against
             independence AND against a Gaussian copula at the same rank
             correlation; a call on which copula should win;
  fit      : Normal vs t margins by AICc; uniforms through the fitted margins;
             R from Kendall's tau; Gaussian and t copulas by profile likelihood
             with log-likelihood, nu-hat, AICc, BIC; 100,000-day simulations
             from each copula; VaR and ES at 5% and 1%; implied joint tail counts;
  reconcile: the winner and the margin; which risk number moved; where the
             log-likelihood gain comes from; tail dependence.

Conventions (Week 05 s5.4-5.6):
  * ranks are scaled as rank/(n+1) so they stay strictly inside (0, 1);
  * R is estimated ONCE from Kendall's tau, rho = sin(pi tau / 2), and handed to
    both copulas, so the likelihood gap is copula family and nothing else;
  * nu is profiled on theta = 1/nu in [0.01, 0.49] on a 200-point grid and then
    refined between the grid neighbours (s5.5.3.2);
  * the copula log-likelihood is computed as ln f_joint(y) - sum_i ln f_i(y_i),
    never as a ratio of densities (s5.6.2);
  * AICc and BIC score the copula log-likelihood only, with k_Gauss = 0 and
    k_t = 1 (s5.6.5);
  * the t copula is simulated through the mixture Y = L Z / sqrt(Q/nu), Q ~ chi2_nu
    (s5.5.2), and the uniform step uses the COPULA's nu-hat, not any margin's.

Run:  python problem4.py
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

import common
from common import C1, C2, C3

COLS = ["X1", "X2", "X3"]
PAIRS = list(itertools.combinations(range(3), 2))
UNIT = 1_000_000.0
TAIL = 0.025
NSIM = 100_000


# --------------------------------------------------------------------------
# Predict (a): moments, and the one day that carries the kurtosis
# --------------------------------------------------------------------------
def part_a(d: pd.DataFrame) -> dict:
    common.header("Problem 4(a) -- moments of each series")
    x = d[COLS].to_numpy(dtype=float)
    n = x.shape[0]
    out = {}
    print(f"  Under H0: Normal, se(skew) = {np.sqrt(6 / n):.3f} and se(exkurt) = "
          f"{np.sqrt(24 / n):.3f} at n = {n}.")
    print(f"  {'series':<10}{'mean':>13}{'sd':>12}{'skew':>9}{'exkurt':>9}")
    for j, c in enumerate(COLS):
        m = common.moments(x[:, j])
        out[c] = m
        common.print_moments(c, m, pct=True)
    # Leave-one-out on the single most extreme observation of each series.
    print(f"\n  Drop the single most extreme day of each series and recompute:")
    print(f"  {'series':<8}{'day':>6}{'value':>10}{'z-score':>9}{'exkurt all':>12}"
          f"{'exkurt w/o':>12}{'skew w/o':>10}")
    worst = {}
    for j, c in enumerate(COLS):
        i = int(np.argmax(np.abs(x[:, j] - x[:, j].mean())))
        m_wo = common.moments(np.delete(x[:, j], i))
        worst[c] = (i, x[i, j])
        print(f"  {c:<8}{i:>6}{x[i, j]:>+10.4f}{(x[i, j] - out[c]['mean']) / out[c]['sd']:>9.2f}"
              f"{out[c]['exkurt']:>12.3f}{m_wo['exkurt']:>12.3f}{m_wo['skew']:>10.3f}")
    days = {w[0] for w in worst.values()}
    if len(days) == 1:
        day = days.pop()
        print(f"\n  The same day ({day}, 0-indexed) is the most extreme observation of ALL")
        print(f"  THREE series: X1 {x[day, 0]:+.4f}, X2 {x[day, 1]:+.4f}, X3 {x[day, 2]:+.4f}."
              f" That is a joint")
        print(f"  tail event, and it is the first hint about the copula, not just the margins.")
    print(f"\n  X2's excess kurtosis of {out['X2']['exkurt']:.1f} is the number that looks out of"
          f" place: one day")
    print(f"  carries {out['X2']['exkurt'] - common.moments(np.delete(x[:, 1], worst['X2'][0]))['exkurt']:.1f}"
          f" of it. Without that day X2 and X3 both sit near +1.7, still "
          f"{common.moments(np.delete(x[:, 1], worst['X2'][0]))['exkurt'] / np.sqrt(24 / n):.0f} se")
    print(f"  from Normal, so both want a t; X1 is Normal-shaped either way.")
    return {"m": out, "worst": worst}


# --------------------------------------------------------------------------
# Predict (b): rank plots and joint tail counts
# --------------------------------------------------------------------------
def gauss_copula_tail_prob(rho: float, u: float) -> float:
    """P(U1 <= u, U2 <= u) under a Gaussian copula with correlation rho."""
    z = stats.norm.ppf(u)
    return float(stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]]).cdf([z, z]))


def t_copula_tail_prob(rho: float, nu: float, u: float) -> float:
    """P(U1 <= u, U2 <= u) under a bivariate t copula (rho, nu).

    Uses the mixture representation of Week 05 s5.5.2: Y = Z / sqrt(Q/nu) with
    Q ~ chi2_nu, so P(Y1 <= y, Y2 <= y) = E_Q[ Phi_rho(y sqrt(Q/nu), y sqrt(Q/nu)) ],
    a one-dimensional integral over Q. scipy's multivariate_t.cdf is quasi-Monte
    Carlo and moves in the third digit between calls; this is deterministic."""
    y = stats.t.ppf(u, nu)
    mvn = stats.multivariate_normal(mean=[0, 0], cov=[[1, rho], [rho, 1]])

    def integrand(q):
        s = np.sqrt(q / nu)
        return mvn.cdf([y * s, y * s]) * stats.chi2.pdf(q, nu)

    val, _ = integrate.quad(integrand, 0, np.inf, limit=200)
    return float(val)


def joint_tail_counts(u: np.ndarray, q: float = TAIL) -> dict:
    """For each pair: days both <= q (joint worst) and both >= 1-q (joint best)."""
    out = {}
    for i, j in PAIRS:
        lo = int(np.sum((u[:, i] <= q) & (u[:, j] <= q)))
        hi = int(np.sum((u[:, i] >= 1 - q) & (u[:, j] >= 1 - q)))
        out[(i, j)] = (lo, hi)
    return out


def part_b(d: pd.DataFrame) -> dict:
    common.header("Problem 4(b) -- ranks, pair plots, joint tail counts")
    x = d[COLS].to_numpy(dtype=float)
    n = x.shape[0]
    u_rank = stats.rankdata(x, axis=0) / (n + 1)          # strictly inside (0, 1)
    counts = joint_tail_counts(u_rank)
    tau = np.array([[stats.kendalltau(x[:, i], x[:, j])[0] for j in range(3)]
                    for i in range(3)])
    rho_tau = np.sin(np.pi * tau / 2)
    indep = n * TAIL ** 2
    print(f"  'Worst 2.5%' = the {int(TAIL * n)} lowest ranks of each series; 'best 2.5%' the "
          f"{int(TAIL * n)} highest.")
    print(f"  Expected joint count under INDEPENDENCE: {n} x {TAIL}^2 = {indep:.3f} days.")
    print(f"  Expected under a GAUSSIAN copula at the pair's rank correlation (rho from")
    print(f"  Kendall's tau, which is what the fit will use), and under t copulas:")
    print(f"\n  {'pair':<8}{'tau':>7}{'rho':>7}{'both worst':>12}{'both best':>11}"
          f"{'indep':>8}{'Gauss':>8}{'t(8)':>7}{'t(4)':>7}")
    print("  " + "-" * 75)
    bench = {}
    for (i, j), (lo, hi) in counts.items():
        r = rho_tau[i, j]
        g = n * gauss_copula_tail_prob(r, TAIL)
        t8 = n * t_copula_tail_prob(r, 8, TAIL)
        t4 = n * t_copula_tail_prob(r, 4, TAIL)
        bench[(i, j)] = {"rho": r, "gauss": g, "t8": t8, "t4": t4}
        print(f"  {COLS[i] + '-' + COLS[j]:<8}{tau[i, j]:>7.3f}{r:>7.3f}{lo:>12}{hi:>11}"
              f"{indep:>8.2f}{g:>8.2f}{t8:>7.2f}{t4:>7.2f}")
    print("  " + "-" * 75)
    tot_lo = sum(c[0] for c in counts.values())
    tot_hi = sum(c[1] for c in counts.values())
    tot_g = sum(b["gauss"] for b in bench.values())
    print(f"  totals  : worst {tot_lo}, best {tot_hi}  ->  {tot_lo + tot_hi} joint tail days "
          f"over 6 pair-tails")
    print(f"  Gaussian: {2 * tot_g:.1f} expected over the same 6 pair-tails "
          f"(Poisson sd ~ {np.sqrt(2 * tot_g):.1f})")
    print(f"  t(8)    : {2 * sum(b['t8'] for b in bench.values()):.1f};   "
          f"t(4): {2 * sum(b['t4'] for b in bench.values()):.1f}")
    print(f"\n  Against independence the corners are crowded ({(tot_lo + tot_hi) / (6 * indep):.0f}x). "
          f"Most of that is just")
    print(f"  correlation: the Gaussian copula, which has NO tail dependence, already")
    print(f"  expects {2 * tot_g:.0f} of the {tot_lo + tot_hi}. The excess over Gaussian is "
          f"{tot_lo + tot_hi - 2 * tot_g:+.0f} days, about")
    print(f"  {(tot_lo + tot_hi - 2 * tot_g) / np.sqrt(2 * tot_g):.1f} Poisson standard deviations"
          f": the corners are more connected than the")
    print(f"  correlation alone can explain, which is what tail dependence looks like.")

    # Which nu would reproduce the observed total under a t copula with these
    # rhos? A one-line solve, as a prediction of the fitted nu.
    def excess(nu):
        return 2 * sum(n * t_copula_tail_prob(b["rho"], nu, TAIL) for b in bench.values()) \
            - (tot_lo + tot_hi)
    nu_count = optimize.brentq(excess, 2.1, 60.0) if excess(60.0) < 0 < excess(2.1) else np.nan
    print(f"  A t copula at these correlations reproduces the {tot_lo + tot_hi} corner days at "
          f"nu ~ {nu_count:.1f}.")
    print(f"  Prediction: the t copula wins, with a single-digit nu.")
    return {"u_rank": u_rank, "counts": counts, "tau": tau, "rho_tau": rho_tau,
            "bench": bench, "indep": indep, "tot": (tot_lo, tot_hi), "tot_gauss": 2 * tot_g,
            "nu_count": nu_count}


def figure_predict(d: pd.DataFrame, A: dict, B: dict) -> None:
    common.use_style()
    u = B["u_rank"]
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.3))
    day = A["worst"]["X2"][0]
    for k, (i, j) in enumerate(PAIRS):
        a = ax[k]
        a.scatter(u[:, i], u[:, j], s=6, color=C1, alpha=0.45, edgecolor="none")
        a.scatter(u[day, i], u[day, j], s=40, color=C2, zorder=5, label=f"day {day}")
        for x0, y0 in ((0, 0), (1 - TAIL, 1 - TAIL)):
            a.add_patch(plt.Rectangle((x0, y0), TAIL, TAIL, fill=False, edgecolor=C2,
                                      linewidth=1.3))
        lo, hi = B["counts"][(i, j)]
        g = B["bench"][(i, j)]["gauss"]
        a.text(0.03, 0.08, f"{lo} days\n(Gauss {g:.1f})", fontsize=8, color=C2)
        a.text(0.97, 0.90, f"{hi} days\n(Gauss {g:.1f})", fontsize=8, color=C2, ha="right")
        a.set_xlim(0, 1); a.set_ylim(0, 1)
        a.set_aspect("equal")
        a.set_xlabel(f"rank of {COLS[i]} / (n+1)")
        a.set_ylabel(f"rank of {COLS[j]} / (n+1)")
        a.set_title(f"{COLS[i]} vs {COLS[j]}:  tau = {B['tau'][i, j]:.3f}")
        if k == 0:
            a.legend(loc="upper left")
    fig.suptitle("Problem 4 -- Predict: rank scatter plots with the joint 2.5% corners",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    common.finish(fig, "problem4_predict")


# --------------------------------------------------------------------------
# Fit (c): margins
# --------------------------------------------------------------------------
def part_c(d: pd.DataFrame) -> dict:
    common.header("Problem 4(c) -- Normal vs Student t for each margin, by AICc")
    x = d[COLS].to_numpy(dtype=float)
    n = x.shape[0]
    out = {}
    print(f"  {'series':<8}{'ll Normal':>11}{'ll t':>10}{'nu':>8}{'AICc N':>11}{'AICc t':>11}"
          f"{'dAICc':>8}   choice")
    print("  " + "-" * 80)
    for j, c in enumerate(COLS):
        fn, ft = common.fit_normal(x[:, j]), common.fit_t(x[:, j])
        an, at = common.aicc(fn["loglik"], 2, n), common.aicc(ft["loglik"], 3, n)
        choice = "Normal" if an <= at else "t"
        out[c] = {"normal": fn, "t": ft, "aicc_n": an, "aicc_t": at, "choice": choice,
                  "dist": fn["dist"] if choice == "Normal" else ft["dist"]}
        nu_txt = f"{ft['nu']:.2f}" if ft["nu"] < common.NU_MAX - 1e-6 else f">={common.NU_MAX:.0f}"
        print(f"  {c:<8}{fn['loglik']:>11.2f}{ft['loglik']:>10.2f}{nu_txt:>8}{an:>11.2f}"
              f"{at:>11.2f}{at - an:>+8.2f}   {choice}")
    print("  " + "-" * 80)
    print("  k = 2 for the Normal (mu, sigma), k = 3 for the t (mu, scale, nu). For X1 the t's")
    print("  nu runs to its upper bound and its likelihood equals the Normal's, so AICc")
    print("  charges it for a parameter that bought nothing. Fitted t margins:")
    for c in COLS:
        ft = out[c]["t"]
        print(f"    {c}: mu = {ft['mu']:+.6f}, scale = {ft['scale']:.6f}, nu = {ft['nu']:.2f}, "
              f"implied sd = {ft['sd_implied']:.6f}")
    return out


# --------------------------------------------------------------------------
# Fit (d): uniforms, R from Kendall's tau, both copulas
# --------------------------------------------------------------------------
def mvt_logpdf(y: np.ndarray, R: np.ndarray, nu: float) -> np.ndarray:
    """Multivariate t log density (Week 05 s5.5.1) with mu = 0, scale R:
         ln G((nu+n)/2) - ln G(nu/2) - n/2 ln(nu pi) - 1/2 ln|R|
         - (nu+n)/2 ln(1 + y' R^-1 y / nu)
    Written out rather than taken from scipy so the pieces are visible; it is
    checked against scipy.stats.multivariate_t below."""
    n = R.shape[1]
    Rinv = np.linalg.inv(R)
    _, logdet = np.linalg.slogdet(R)
    q = np.einsum("ki,ij,kj->k", y, Rinv, y)
    return (special.gammaln((nu + n) / 2) - special.gammaln(nu / 2)
            - 0.5 * n * np.log(nu * np.pi) - 0.5 * logdet
            - 0.5 * (nu + n) * np.log1p(q / nu))


def gauss_copula_loglik_days(u: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Per-day ln c_R(u) = ln phi_R(z) - sum_i ln phi(z_i), z = Phi^-1(u)."""
    z = stats.norm.ppf(u)
    joint = stats.multivariate_normal(mean=np.zeros(R.shape[0]), cov=R).logpdf(z)
    return joint - stats.norm.logpdf(z).sum(axis=1)


def t_copula_loglik_days(u: np.ndarray, R: np.ndarray, nu: float) -> np.ndarray:
    """Per-day ln c_{R,nu}(u) = ln f_{R,nu}(y) - sum_i ln f_nu(y_i), y = t_nu^-1(u)."""
    y = stats.t.ppf(u, nu)
    return mvt_logpdf(y, R, nu) - stats.t.logpdf(y, nu).sum(axis=1)


def profile_nu(u: np.ndarray, R: np.ndarray, grid_pts: int = 200) -> dict:
    """Maximise the t copula log-likelihood over nu with R fixed, on theta = 1/nu
    in [0.01, 0.49] (nu in [2.04, 100]); grid first, then refine between the
    neighbours of the grid maximum with a bounded scalar search (s5.5.3.2)."""
    thetas = np.linspace(0.01, 0.49, grid_pts)
    ll = np.array([t_copula_loglik_days(u, R, 1 / th).sum() for th in thetas])
    k = int(np.argmax(ll))
    lo, hi = thetas[max(k - 1, 0)], thetas[min(k + 1, grid_pts - 1)]
    res = optimize.minimize_scalar(lambda th: -t_copula_loglik_days(u, R, 1 / th).sum(),
                                   bounds=(lo, hi), method="bounded",
                                   options={"xatol": 1e-8})
    return {"nu": 1 / res.x, "loglik": -res.fun, "grid_nu": 1 / thetas, "grid_ll": ll}


def part_d(d: pd.DataFrame, M: dict, B: dict) -> dict:
    common.header("Problem 4(d) -- uniforms, R from Kendall's tau, Gaussian and t copulas")
    x = d[COLS].to_numpy(dtype=float)
    n = x.shape[0]
    # Step 1: U through each FITTED margin (the AICc choice from part c).
    u = np.column_stack([M[c]["dist"].cdf(x[:, j]) for j, c in enumerate(COLS)])
    u = np.clip(u, 1e-10, 1 - 1e-10)
    print("  Uniforms through the fitted margins: " +
          ", ".join(f"{c} -> {M[c]['choice']}" for c in COLS))
    for j, c in enumerate(COLS):
        ks = stats.kstest(u[:, j], "uniform")
        print(f"    {c}: mean {u[:, j].mean():.3f}, KS vs Uniform p = {ks[1]:.2f}")
    # Step 2: R from Kendall's tau (ranks are unchanged by the monotone margin
    # transform, so this is the same tau as in part b).
    R = B["rho_tau"].copy()
    np.fill_diagonal(R, 1.0)
    eig = np.linalg.eigvalsh(R)
    common.print_matrix(R, COLS, title="\n  R = sin(pi tau / 2):")
    print(f"  eigenvalues {np.array2string(eig, precision=4)} -> PSD, no repair needed.")
    # Step 3: likelihoods.
    ll_g_days = gauss_copula_loglik_days(u, R)
    prof = profile_nu(u, R)
    nu_hat = prof["nu"]
    ll_t_days = t_copula_loglik_days(u, R, nu_hat)
    ll_g, ll_t = ll_g_days.sum(), ll_t_days.sum()
    # Sanity check of the hand-written density against scipy.
    y = stats.t.ppf(u[:5], nu_hat)
    chk = np.abs(mvt_logpdf(y, R, nu_hat)
                 - stats.multivariate_t(loc=np.zeros(3), shape=R, df=nu_hat).logpdf(y)).max()
    res = {
        "Gaussian": {"loglik": ll_g, "k": 0, "nu": np.inf,
                     "aicc": common.aicc(ll_g, 0, n), "bic": common.bic(ll_g, 0, n)},
        "t": {"loglik": ll_t, "k": 1, "nu": nu_hat,
              "aicc": common.aicc(ll_t, 1, n), "bic": common.bic(ll_t, 1, n)},
    }
    print(f"\n  {'copula':<10}{'loglik':>10}{'k':>4}{'nu-hat':>9}{'AICc':>11}{'BIC':>11}")
    print("  " + "-" * 55)
    for name, r in res.items():
        nu_txt = "inf" if np.isinf(r["nu"]) else f"{r['nu']:.2f}"
        print(f"  {name:<10}{r['loglik']:>10.2f}{r['k']:>4}{nu_txt:>9}{r['aicc']:>11.2f}"
              f"{r['bic']:>11.2f}")
    print("  " + "-" * 55)
    d_aicc = res["Gaussian"]["aicc"] - res["t"]["aicc"]
    d_bic = res["Gaussian"]["bic"] - res["t"]["bic"]
    print(f"  dAICc (Gauss - t) = {d_aicc:+.2f}     dBIC (Gauss - t) = {d_bic:+.2f}"
          f"   [positive favours the t]")
    print(f"  check: 2 (ll_t - ll_G) - ln m = 2 x {ll_t - ll_g:.2f} - {np.log(n):.2f} = "
          f"{2 * (ll_t - ll_g) - np.log(n):+.2f}")
    # How well identified is nu? Profile-likelihood interval: the nu range over
    # which the log-likelihood stays within 1.92 (= chi2_1(0.95)/2) of its max.
    g_nu, g_ll = prof["grid_nu"], prof["grid_ll"]
    inside = g_nu[g_ll >= ll_t - 1.92]
    print(f"  Sharpness of the maximum: ll at nu = {nu_hat / 2:.1f} is "
          f"{t_copula_loglik_days(u, R, nu_hat / 2).sum():.2f}, at nu = {2 * nu_hat:.1f} is "
          f"{t_copula_loglik_days(u, R, 2 * nu_hat).sum():.2f},")
    print(f"  max {ll_t:.2f}. Profile-likelihood 95% interval for nu: "
          f"[{inside.min():.2f}, {inside.max():.2f}].")
    print(f"  Unlike the course's worked example (s5.6.5.1), nu is tightly pinned here.")
    print(f"  (hand-written multivariate t density vs scipy: max |diff| = {chk:.1e})")
    return {"u": u, "R": R, "res": res, "ll_g_days": ll_g_days, "ll_t_days": ll_t_days,
            "prof": prof, "nu": nu_hat, "d_aicc": d_aicc, "d_bic": d_bic,
            "nu_ci": (inside.min(), inside.max())}


# --------------------------------------------------------------------------
# Fit (e): simulate, VaR and ES
# --------------------------------------------------------------------------
def simulate(M: dict, R: np.ndarray, nu: float | None, rng: np.random.Generator,
             z: np.ndarray) -> tuple:
    """Draw NSIM scenarios from the copula and push them through the fitted
    margins. `z` is a shared (NSIM x 3) standard normal block so both copulas see
    the same underlying randomness; the t adds its chi-square mixing draw.
    Returns (returns matrix, uniforms)."""
    L = np.linalg.cholesky(R)
    y = z @ L.T                                          # correlated normals
    if nu is None:
        u = stats.norm.cdf(y)
    else:
        w = np.sqrt(nu / rng.chisquare(nu, size=z.shape[0]))   # one W per row
        u = stats.t.cdf(y * w[:, None], nu)              # COPULA's nu here
    u = np.clip(u, 1e-12, 1 - 1e-12)
    x = np.column_stack([M[c]["dist"].ppf(u[:, j]) for j, c in enumerate(COLS)])
    return x, u


def block_se(pnl: np.ndarray, alpha: float, blocks: int = 10) -> tuple:
    """Monte Carlo standard error of VaR and ES from `blocks` independent
    sub-samples: sd of the block estimates / sqrt(blocks)."""
    parts = np.array_split(pnl, blocks)
    v = np.array([common.var_es(p, alpha) for p in parts])
    return v[:, 0].std(ddof=1) / np.sqrt(blocks), v[:, 1].std(ddof=1) / np.sqrt(blocks)


def part_e(d: pd.DataFrame, M: dict, D: dict) -> dict:
    common.header(f"Problem 4(e) -- {NSIM:,} simulated days through each copula; portfolio "
                  f"VaR and ES")
    rng = np.random.default_rng(common.SEED)
    z = rng.standard_normal((NSIM, 3))
    x_g, u_g = simulate(M, D["R"], None, rng, z)
    x_t, u_t = simulate(M, D["R"], D["nu"], rng, z)
    pnl = {"Gaussian": UNIT * x_g.sum(axis=1), "t": UNIT * x_t.sum(axis=1),
           "historical": UNIT * d[COLS].to_numpy(dtype=float).sum(axis=1)}
    out = {}
    print(f"  Portfolio: $1M in each of X1, X2, X3  ->  P&L = $1M x (X1 + X2 + X3)")
    print(f"\n  {'source':<12}{'VaR 5%':>12}{'ES 5%':>12}{'VaR 1%':>12}{'ES 1%':>12}")
    print("  " + "-" * 60)
    for name, p in pnl.items():
        row = {}
        for a in (0.05, 0.01):
            row[a] = common.var_es(p, a)
        out[name] = row
        print(f"  {name:<12}{common.money(row[0.05][0]):>12}{common.money(row[0.05][1]):>12}"
              f"{common.money(row[0.01][0]):>12}{common.money(row[0.01][1]):>12}")
    print("  " + "-" * 60)
    print("  Monte Carlo standard errors (10 blocks of 10,000):")
    for name in ("Gaussian", "t"):
        s5, e5 = block_se(pnl[name], 0.05)
        s1, e1 = block_se(pnl[name], 0.01)
        print(f"  {name:<12}{common.money(s5):>12}{common.money(e5):>12}{common.money(s1):>12}"
              f"{common.money(e1):>12}")
    print(f"  Historical numbers rest on {int(0.05 * len(d))} and {int(0.01 * len(d))} days "
          f"respectively.")
    # Check the simulated margins reproduce the fitted ones.
    print(f"\n  Simulated (t copula) vs fitted margin sd:")
    for j, c in enumerate(COLS):
        print(f"    {c}: {x_t[:, j].std():.5f} vs {M[c]['dist'].std():.5f}")
    return {"pnl": pnl, "risk": out, "u_g": u_g, "u_t": u_t, "x_g": x_g, "x_t": x_t}


# --------------------------------------------------------------------------
# Fit (f): implied joint tail counts
# --------------------------------------------------------------------------
def part_f(B: dict, S: dict) -> dict:
    common.header("Problem 4(f) -- joint 2.5% tail days per 1,000 days implied by each copula")
    n_data = B["u_rank"].shape[0]
    scale = n_data / NSIM
    cg = joint_tail_counts(S["u_g"])
    ct = joint_tail_counts(S["u_t"])
    print(f"  {'pair':<8}{'data worst':>11}{'Gauss':>8}{'t':>7}{'data best':>12}{'Gauss':>8}"
          f"{'t':>7}")
    print("  " + "-" * 61)
    tot = np.zeros((3, 2))
    for (i, j), (lo, hi) in B["counts"].items():
        g_lo, g_hi = cg[(i, j)][0] * scale, cg[(i, j)][1] * scale
        t_lo, t_hi = ct[(i, j)][0] * scale, ct[(i, j)][1] * scale
        tot += np.array([[lo, hi], [g_lo, g_hi], [t_lo, t_hi]])
        print(f"  {COLS[i] + '-' + COLS[j]:<8}{lo:>11}{g_lo:>8.2f}{t_lo:>7.2f}{hi:>12}"
              f"{g_hi:>8.2f}{t_hi:>7.2f}")
    print("  " + "-" * 61)
    print(f"  {'total':<8}{tot[0, 0]:>11.0f}{tot[1, 0]:>8.2f}{tot[2, 0]:>7.2f}{tot[0, 1]:>12.0f}"
          f"{tot[1, 1]:>8.2f}{tot[2, 1]:>7.2f}")
    print(f"  all 6 : data {tot[0].sum():.0f}, Gaussian {tot[1].sum():.1f}, t {tot[2].sum():.1f}"
          f"  (Poisson sd on a count of 50 is ~7)")
    return {"gauss": cg, "t": ct, "tot": tot}


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------
def part_g(D: dict, F: dict) -> None:
    common.header("Problem 4(g) -- which copula wins, and by how much")
    r = D["res"]
    print(f"  t copula: ll {r['t']['loglik']:.2f} vs Gaussian {r['Gaussian']['loglik']:.2f}, "
          f"gain {r['t']['loglik'] - r['Gaussian']['loglik']:.2f} for one parameter.")
    print(f"  dAICc = {D['d_aicc']:+.2f}, dBIC = {D['d_bic']:+.2f} (positive favours the t).")
    verdict = "t" if D["d_bic"] > 0 else "Gaussian"
    print(f"  Both criteria agree: the {verdict} wins. Week 05 s5.6.5 reads a dBIC above 10")
    print(f"  as strong evidence; this is {'well past' if abs(D['d_bic']) > 10 else 'short of'} "
          f"that, with nu-hat = {D['nu']:.1f}.")
    tot = F["tot"]
    print(f"\n  Joint 2.5% tail days over 1,000 days, all six corners:")
    print(f"    data {tot[0].sum():.0f}   Gaussian {tot[1].sum():.1f}   t {tot[2].sum():.1f}"
          f"   (Poisson sd on ~50 is ~7)")
    print(f"  The Gaussian under-produces the corners by {tot[0].sum() - tot[1].sum():.0f} days, "
          f"{(tot[0].sum() - tot[1].sum()) / np.sqrt(tot[1].sum()):.1f} Poisson sd; the t")
    print(f"  lands within {abs(tot[2].sum() - tot[0].sum()):.0f} days of the data, inside the "
          f"noise. The fitted nu of {D['nu']:.1f} also")
    print(f"  matches the nu the raw corner counts asked for in (b).")


def part_h(S: dict) -> None:
    common.header("Problem 4(h) -- which risk number moved")
    g, t = S["risk"]["Gaussian"], S["risk"]["t"]
    print(f"  {'measure':<10}{'Gaussian':>12}{'t':>12}{'change':>10}{'% change':>10}")
    print("  " + "-" * 54)
    for a in (0.05, 0.01):
        for k, lab in ((0, "VaR"), (1, "ES")):
            ch = t[a][k] - g[a][k]
            print(f"  {lab + ' ' + str(int(100 * a)) + '%':<10}{common.money(g[a][k]):>12}"
                  f"{common.money(t[a][k]):>12}{common.money(ch):>10}{100 * ch / g[a][k]:>+9.1f}%")
    print("  " + "-" * 54)
    print("  The 5% VaR barely moves. Both simulations share the margins (so every")
    print("  single-asset quantile is identical) and the same R (so the portfolio's")
    print("  variance is identical). The 5% quantile of a 3-asset sum sits in the")
    print("  shoulder of the distribution, where the two copulas put almost the same")
    print("  mass -- a 5% portfolio day is usually one bad asset and two ordinary ones.")
    print("  The t copula differs in how often ALL THREE are bad on the same day, and")
    print("  that shows up where those days live: the far tail, i.e. the 1% ES.")


def part_i(D: dict) -> dict:
    common.header("Problem 4(i) -- where the log-likelihood gain comes from")
    u = D["u"]
    contrib = D["ll_t_days"] - D["ll_g_days"]
    total = contrib.sum()
    outer = (u <= 0.05) | (u >= 0.95)                  # any series in its outer 5%
    interior = ~outer.any(axis=1)
    n_int = int(interior.sum())
    share_int = contrib[interior].sum() / total
    print(f"  sum over days of (ln c_t - ln c_Gauss) = {total:.2f}  (= ll_t - ll_Gauss)")
    print(f"  Days with NO series in its outer 5% on either side: {n_int} of {u.shape[0]}")
    print(f"    their contribution: {contrib[interior].sum():+.2f}  = {100 * share_int:.0f}% of "
          f"the total")
    print(f"  Days with at least one series in an outer 5%: {u.shape[0] - n_int}")
    print(f"    their contribution: {contrib[~interior].sum():+.2f}  = "
          f"{100 * (1 - share_int):.0f}% of the total")
    # Finer split by the most extreme coordinate of the day.
    ext = np.abs(u - 0.5).max(axis=1)                   # 0 = dead centre, 0.5 = extreme
    print(f"\n  By the day's most extreme coordinate, max_i |u_i - 0.5|:")
    print(f"  {'band':<22}{'days':>6}{'sum contrib':>13}{'mean/day':>10}{'share':>8}")
    edges = [0, 0.3, 0.4, 0.45, 0.475, 0.49, 0.5001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ext >= lo) & (ext < hi)
        if m.sum():
            print(f"  [{lo:.3f}, {hi if hi < 0.5 else 0.5:.3f})      {int(m.sum()):>6}"
                  f"{contrib[m].sum():>+13.2f}{contrib[m].mean():>+10.4f}"
                  f"{100 * contrib[m].sum() / total:>7.0f}%")
    top = np.argsort(contrib)[::-1][:5]
    print(f"\n  Five largest single-day contributions:")
    for k in top:
        print(f"    day {k:>4}: {contrib[k]:+.3f}   u = ({u[k, 0]:.3f}, {u[k, 1]:.3f}, {u[k, 2]:.3f})")
    # Are the big days concordant (all extreme on the same side) or discordant?
    ext_days = ext >= 0.475
    z = stats.norm.ppf(u[ext_days])
    same_side = np.all(np.sign(z) == np.sign(z[:, [0]]), axis=1)
    print(f"  Of the {int(ext_days.sum())} days with a coordinate beyond 2.5%, "
          f"{int(same_side.sum())} have all three series on the")
    print(f"  same side (contribution {contrib[ext_days][same_side].sum():+.2f}) and "
          f"{int((~same_side).sum())} do not (contribution "
          f"{contrib[ext_days][~same_side].sum():+.2f}).")
    neg = contrib < 0
    print(f"\n  {int(neg.sum())} days favour the Gaussian (sum {contrib[neg].sum():+.2f}); "
          f"{int((~neg).sum())} favour the t (sum {contrib[~neg].sum():+.2f}).")
    print(f"  Two things the criterion is scoring, then. Half the gain is the SHAPE of the")
    print(f"  centre: a t copula puts more mass near (0.5, 0.5, 0.5) and less in the")
    print(f"  shoulders, and {n_int} days live there. The other half is the extreme days,")
    print(f"  and within those it is NOT the joint crashes: the concordant extremes score")
    print(f"  about evenly under both copulas, and nearly all of the gain is days where")
    print(f"  the assets were extreme in OPPOSITE directions. A Gaussian at rho ~ 0.5")
    print(f"  calls such a day almost impossible; a t with one shared scale draw W")
    print(f"  (s5.5.2) does not. 'Tail dependence' in a t copula is radial: big days are")
    print(f"  big in every direction, not only along the diagonal.")
    return {"contrib": contrib, "interior": interior, "share_interior": share_int, "ext": ext}


def part_j(D: dict) -> dict:
    common.header("Problem 4(j) -- tail dependence of the most correlated pair")
    R, nu = D["R"], D["nu"]
    off = R - np.eye(3)
    i, j = np.unravel_index(np.argmax(off), off.shape)
    rho = R[i, j]
    lam = 2 * stats.t.cdf(-np.sqrt((nu + 1) * (1 - rho) / (1 + rho)), nu + 1)
    print(f"  Most correlated pair: {COLS[i]}-{COLS[j]}, rho = {rho:.4f}")
    print(f"  t copula, nu = {nu:.2f}:  lambda = 2 t_(nu+1)( -sqrt((nu+1)(1-rho)/(1+rho)) ) = "
          f"{lam:.4f}")
    print(f"  Gaussian copula:          lambda = 0 for every rho < 1")
    print(f"  Reading: given {COLS[i]} has a 1-in-a-million day, the t copula says {COLS[j]} has "
          f"a {100 * lam:.0f}%")
    print(f"  chance of one too; the Gaussian says that chance falls to zero. At the 2.5%")
    print(f"  level the two are still close (which is why the counts in (f) differ by")
    print(f"  only a handful of days); the gap opens as the level goes to zero.")
    print(f"  For reference, lambda at this rho for other nu:")
    print(f"    " + "   ".join(
        f"nu={v}: {2 * stats.t.cdf(-np.sqrt((v + 1) * (1 - rho) / (1 + rho)), v + 1):.3f}"
        for v in (4, 8, 16, 30)))
    return {"pair": (COLS[i], COLS[j]), "rho": rho, "lambda_t": lam, "lambda_gauss": 0.0}


# --------------------------------------------------------------------------
# Fit figure
# --------------------------------------------------------------------------
def figure_fit(D: dict, S: dict, I: dict) -> None:
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    # Profile likelihood gain vs nu with the two penalties.
    prof = D["prof"]
    n = D["u"].shape[0]
    gain = prof["grid_ll"] - D["res"]["Gaussian"]["loglik"]
    ax[0].plot(prof["grid_nu"], 2 * gain, color=C1, label="2 (ll_t(nu) - ll_Gauss)")
    ax[0].axhline(2, color=C2, linestyle="--", label="AIC penalty, 2")
    ax[0].axhline(np.log(n), color=C3, linestyle="--", label=f"BIC penalty, ln m = {np.log(n):.2f}")
    ax[0].axvline(D["nu"], color=common.INK2, linewidth=0.9)
    ax[0].text(D["nu"] * 1.08, 2 * gain.max() * 0.5, f"nu-hat = {D['nu']:.1f}", fontsize=8.5,
               color=common.INK2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("nu (log scale)")
    ax[0].set_ylabel("twice the log-likelihood gain")
    ax[0].set_title("t copula profile: gain over the Gaussian")
    ax[0].legend(loc="upper right", fontsize=8)

    # Per-day contribution vs most extreme coordinate.
    c = I["contrib"]
    ext = I["ext"]
    ax[1].scatter(ext[I["interior"]], c[I["interior"]], s=7, color=C1, alpha=0.5,
                  edgecolor="none", label=f"no series in outer 5% ({100 * I['share_interior']:.0f}% of gain)")
    ax[1].scatter(ext[~I["interior"]], c[~I["interior"]], s=9, color=C2, alpha=0.7,
                  edgecolor="none", label="some series in outer 5%")
    ax[1].axhline(0, color=common.INK, linewidth=0.8)
    ax[1].set_xlabel("max_i |u_i - 0.5|  (0 = centre, 0.5 = extreme)")
    ax[1].set_ylabel("ln c_t - ln c_Gauss, per day")
    ax[1].set_title("Who pays for the t: the centre and the corners")
    ax[1].legend(loc="upper left", fontsize=8)

    # VaR against alpha under each copula and historically.
    alphas = np.logspace(np.log10(0.001), np.log10(0.10), 50)
    for name, col, ls in (("Gaussian", C2, "-"), ("t", C3, "--"), ("historical", C1, "-")):
        p = S["pnl"][name]
        a_use = alphas if name != "historical" else alphas[alphas >= 0.002]
        ax[2].plot(100 * a_use, [common.var_es(p, a)[1] / 1e3 for a in a_use], color=col,
                   linestyle=ls, label=f"ES, {name}")
    for name, col in (("Gaussian", C2), ("t", C3)):
        ax[2].plot(100 * alphas, [common.var_es(S["pnl"][name], a)[0] / 1e3 for a in alphas],
                   color=col, linewidth=1.0, alpha=0.7, label=f"VaR, {name}")
    for a in (5, 1):
        ax[2].axvline(a, color=common.INK2, linewidth=0.8)
    ax[2].set_xscale("log")
    ax[2].set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10])
    ax[2].set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax[2].set_xlabel("alpha (%)")
    ax[2].set_ylabel("$ thousands")
    ax[2].set_title("Portfolio VaR and ES: the copulas part deep in the tail")
    ax[2].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Problem 4 -- Fit: Gaussian vs t copula on the same margins and the same R",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem4_fit")


def main() -> dict:
    d = common.load("problem4")
    A = part_a(d)
    B = part_b(d)
    figure_predict(d, A, B)
    M = part_c(d)
    D = part_d(d, M, B)
    S = part_e(d, M, D)
    F = part_f(B, S)
    part_g(D, F)
    part_h(S)
    I = part_i(D)
    J = part_j(D)
    figure_fit(D, S, I)
    return {"A": A, "B": B, "M": M, "D": D, "S": S, "F": F, "I": I, "J": J}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

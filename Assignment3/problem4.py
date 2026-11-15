"""
Problem 4 -- A Sharpe ratio that sells insurance.

problem4.csv: 240 monthly total returns for EQ (equity index), BOND, STK (a
single stock) and PUTW (sells a one-month put on EQ struck 5% below spot each
month, fully collateralised at the risk free rate). rf = 3% a year.

  predict  : (a) moments and annualised Sharpe ratios, with the single month
             that drives the odd number; (b) how much PUTW the long-only
             tangency portfolio will hold, and whether its 1% ES will look Normal.
  fit      : (c) max-Sharpe portfolios: unconstrained, long only, long only
             with PUTW <= 20%; long-only frontier and CML; (d) Sharpe, historical
             and Normal VaR/ES at 5% and 1% for each; (e) CAPM of PUTW on EQ,
             and beta on the months EQ's excess return was below -5%.
  reconcile: (f) what the optimizer bought, and why ES sees it; (g) is the
             alpha skill; (h) the standard errors of the means and a bootstrap
             of the tangency weights.

Conventions:
  * monthly risk free rate rf_m = 3%/12 = 0.25% (simple; the difference from
    1.03^(1/12) - 1 = 0.2466% is printed and changes no conclusion);
  * annualised Sharpe = (mean_m - rf_m) / sd_m x sqrt(12) (Week 07; i.i.d.
    scaling of the mean by 12 and the sd by sqrt(12));
  * mean and covariance are the plain sample moments (sd with n - 1), the
    inputs the Week 07 optimiser takes;
  * weights sum to one; the unconstrained solution is the closed form
    w ~ Sigma^{-1}(mu - rf) normalised to sum to one; the constrained ones are
    solved with SLSQP (scipy) -- the role Ipopt plays in library/optimizers.jl;
  * VaR / ES are monthly, in percent of portfolio value, positive = loss;
    historical numbers use the RiskStats.jl estimator (common.var_es).

Run:  python problem4.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

import common
from common import C1, C2, C3, C4, INK2

RF_A = 0.03
RF_M = RF_A / 12.0
ASSETS = ["EQ", "BOND", "STK", "PUTW"]


def prepare() -> dict:
    d = common.load("problem4")
    d["Month"] = pd.to_datetime(d.Month)
    R = d[ASSETS].to_numpy(dtype=float)
    return {"d": d, "R": R, "mu": R.mean(axis=0), "cov": np.cov(R, rowvar=False)}


def sharpe_m(r) -> float:
    r = np.asarray(r, dtype=float)
    return (r.mean() - RF_M) / r.std(ddof=1)


# --------------------------------------------------------------------------
# (a) moments and Sharpe ratios
# --------------------------------------------------------------------------
def part_a(D: dict) -> pd.DataFrame:
    d = D["d"]
    rows = []
    for a in ASSETS:
        m = common.moments(d[a])
        rows.append({"asset": a, "mean/mo": m["mean"], "sd/mo": m["sd"], "skew": m["skew"],
                     "exkurt": m["exkurt"], "ann mean": 12 * m["mean"],
                     "ann sd": np.sqrt(12) * m["sd"],
                     "Sharpe (ann)": np.sqrt(12) * sharpe_m(d[a]),
                     "min": d[a].min(), "max": d[a].max()})
    out = pd.DataFrame(rows).set_index("asset")
    common.header("Problem 4(a) - monthly moments and annualised Sharpe ratios")
    print(f"rf_m = {RF_M:.5f} (3%/12); compounded alternative {1.03 ** (1 / 12) - 1:.5f}\n")
    common.show(out, {"mean/mo": "{:.5f}", "sd/mo": "{:.5f}", "skew": "{:+.3f}",
                      "exkurt": "{:.3f}", "ann mean": "{:.4f}", "ann sd": "{:.4f}",
                      "Sharpe (ann)": "{:.3f}", "min": "{:+.4f}", "max": "{:+.4f}"})

    # PUTW's distribution: how many months are exactly the premium + interest?
    p = d.PUTW.to_numpy()
    mode = pd.Series(p).mode()[0]
    at = np.abs(p - mode) < 1e-9
    print(f"\nPUTW: {int(at.sum())} of {len(p)} months return exactly {mode:.6f} "
          f"(premium + interest, put expired worthless).")
    print(f"      {int(np.sum(p < mode - 1e-9))} months below that (the put paid off);"
          f" {int(np.sum(p > mode + 1e-9))} months above it.")

    # Leave-one-out on the most extreme month of each asset
    common.header("Problem 4(a) - how much comes from a single month (leave the worst month out)")
    loo = []
    for a in ASSETS:
        x = d[a].to_numpy()
        i = int(np.argmin(x))
        xm = np.delete(x, i)
        loo.append({"asset": a, "worst month": d.Month.iloc[i].strftime("%Y-%m"),
                    "worst ret": x[i], "EQ that month": d.EQ.iloc[i],
                    "skew all": stats.skew(x, bias=False), "skew ex-worst": stats.skew(xm, bias=False),
                    "exkurt all": stats.kurtosis(x, bias=False),
                    "exkurt ex-worst": stats.kurtosis(xm, bias=False),
                    "SR all": np.sqrt(12) * sharpe_m(x), "SR ex-worst": np.sqrt(12) * sharpe_m(xm)})
    loo = pd.DataFrame(loo).set_index("asset")
    common.show(loo, {"worst ret": "{:+.4f}", "EQ that month": "{:+.4f}", "skew all": "{:+.2f}",
                      "skew ex-worst": "{:+.2f}", "exkurt all": "{:.2f}",
                      "exkurt ex-worst": "{:.2f}", "SR all": "{:.3f}", "SR ex-worst": "{:.3f}"})
    # PUTW worst months, all of them
    worst = d.nsmallest(6, "PUTW")[["Month", "EQ", "PUTW"]].copy()
    worst["Month"] = worst.Month.dt.strftime("%Y-%m")
    print("\nPUTW's six worst months and what EQ did:")
    common.show(worst.set_index("Month"), "{:+.4f}")
    D["moments"] = out
    return out


def figure_predict(D: dict) -> None:
    d = D["d"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    ax = axes[0]
    ax.scatter(d.EQ, d.PUTW, s=9, color=C1, alpha=0.7)
    x = np.linspace(d.EQ.min(), d.EQ.max(), 200)
    ax.axvline(-0.05, color=INK2, lw=0.8, ls=":")
    ax.text(-0.048, d.PUTW.min(), " EQ = -5%", fontsize=7.5, color=INK2)
    ax.set_xlabel("EQ monthly return")
    ax.set_ylabel("PUTW monthly return")
    ax.set_title("PUTW against EQ: a short put payoff")
    ax = axes[1]
    for a, c in zip(ASSETS, (C1, C3, C4, C2)):
        ax.plot(d.Month, np.cumprod(1 + d[a]), color=c, lw=1.2, label=a)
    ax.set_yscale("log")
    ax.set_title("growth of $1 (log scale)")
    ax.legend()
    fig.tight_layout()
    common.finish(fig, "problem4_predict")


# --------------------------------------------------------------------------
# (c) optimisation
# --------------------------------------------------------------------------
def port_stats(w, mu, cov) -> tuple:
    m = float(w @ mu)
    s = float(np.sqrt(w @ cov @ w))
    return m, s, (m - RF_M) / s


def max_sr(mu, cov, bounds=None) -> np.ndarray:
    """Maximise (w'mu - rf)/sqrt(w'Sigma w) s.t. sum w = 1 and bounds.
    bounds=None -> closed form tangency portfolio (no bounds)."""
    n = len(mu)
    if bounds is None:
        z = np.linalg.solve(cov, mu - RF_M)
        return z / z.sum()
    obj = lambda w: -(w @ mu - RF_M) / np.sqrt(w @ cov @ w)
    best = None
    # several starts; the problem is quasi-concave but SLSQP can stall at a bound
    for w0 in [np.full(n, 1 / n)] + [np.eye(n)[i] * 0.7 + 0.3 / n for i in range(n)]:
        w0 = np.clip(w0, [b[0] for b in bounds], [b[1] for b in bounds])
        w0 = w0 / w0.sum()
        res = optimize.minimize(obj, w0, method="SLSQP", bounds=bounds,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                                options={"ftol": 1e-14, "maxiter": 2000})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def frontier(mu, cov, n_pts: int = 60) -> pd.DataFrame:
    """Long-only efficient frontier: min w'Sigma w s.t. sum w = 1, w'mu = R,
    w >= 0, for R from the min-variance portfolio's return to max(mu)."""
    n = len(mu)
    bounds = [(0, 1)] * n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    w_mv = optimize.minimize(lambda w: w @ cov @ w, np.full(n, 1 / n), method="SLSQP",
                             bounds=bounds, constraints=cons, options={"ftol": 1e-15}).x
    rows = []
    for tgt in np.linspace(w_mv @ mu, mu.max(), n_pts):
        c2 = cons + [{"type": "eq", "fun": lambda w, t=tgt: w @ mu - t}]
        w = optimize.minimize(lambda w: w @ cov @ w, w_mv, method="SLSQP", bounds=bounds,
                              constraints=c2, options={"ftol": 1e-15, "maxiter": 1000}).x
        rows.append({"ret": w @ mu, "sd": np.sqrt(w @ cov @ w), **dict(zip(ASSETS, w))})
    return pd.DataFrame(rows)


def part_c(D: dict) -> dict:
    mu, cov = D["mu"], D["cov"]
    W = {
        "Unconstrained": max_sr(mu, cov),
        "Long only": max_sr(mu, cov, [(0, 1)] * 4),
        "Long only, PUTW<=20%": max_sr(mu, cov, [(0, 1)] * 3 + [(0, 0.2)]),
    }
    rows = []
    for k, w in W.items():
        m, s, sr = port_stats(w, mu, cov)
        rows.append({"portfolio": k, **dict(zip(ASSETS, w)), "mean/mo": m, "sd/mo": s,
                     "Sharpe (ann)": sr * np.sqrt(12)})
    out = pd.DataFrame(rows).set_index("portfolio")
    common.header("Problem 4(c) - maximum Sharpe ratio portfolios")
    common.show(out, {a: "{:+.4f}" for a in ASSETS} |
                {"mean/mo": "{:.5f}", "sd/mo": "{:.5f}", "Sharpe (ann)": "{:.3f}"})
    D["W"] = W
    D["front"] = frontier(mu, cov)
    return W


def figure_frontier(D: dict) -> None:
    mu, cov, F, W = D["mu"], D["cov"], D["front"], D["W"]
    ann_m = lambda m: 12 * m
    ann_s = lambda s: np.sqrt(12) * s
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(ann_s(F.sd), ann_m(F.ret), color=C1, lw=2, label="long-only efficient frontier")
    m, s, sr = port_stats(W["Long only"], mu, cov)
    xs = np.linspace(0, ann_s(np.sqrt(np.diag(cov)).max()) * 1.02, 50)
    ax.plot(xs, RF_A + sr * np.sqrt(12) * xs, color=C2, lw=1.4, ls="--",
            label=f"CML (long-only tangency, SR {sr * np.sqrt(12):.2f})")
    for a, c in zip(ASSETS, (C1, C3, C4, C2)):
        i = ASSETS.index(a)
        ax.scatter(ann_s(np.sqrt(cov[i, i])), ann_m(mu[i]), s=40, color=c, zorder=5,
                   edgecolor="white")
        ax.annotate(a, (ann_s(np.sqrt(cov[i, i])), ann_m(mu[i])), textcoords="offset points",
                    xytext=(6, -3), fontsize=9)
    for k, mk in (("Long only", "*"), ("Long only, PUTW<=20%", "D")):
        m, s, _ = port_stats(W[k], mu, cov)
        ax.scatter(ann_s(s), ann_m(m), marker=mk, s=110 if mk == "*" else 45, color="black",
                   zorder=6, label=k + " tangency")
    ax.scatter(0, RF_A, color=INK2, s=20, zorder=5)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("annualised standard deviation")
    ax.set_ylabel("annualised mean return")
    ax.set_title("Problem 4(c): long-only frontier and capital market line")
    ax.legend(loc="lower right")
    common.finish(fig, "problem4_frontier")


# --------------------------------------------------------------------------
# (d) risk of the three portfolios
# --------------------------------------------------------------------------
def part_d(D: dict) -> pd.DataFrame:
    R, W = D["R"], D["W"]
    rows = []
    for k, w in W.items():
        p = R @ w
        m, s = p.mean(), p.std(ddof=1)
        rec = {"portfolio": k, "Sharpe (ann)": np.sqrt(12) * sharpe_m(p)}
        for a in (0.05, 0.01):
            hv, he = common.var_es(p, a)
            tag = int(a * 100)
            rec[f"hist VaR{tag}"] = hv
            rec[f"norm VaR{tag}"] = common.var_normal(s, a, m)
            rec[f"hist ES{tag}"] = he
            rec[f"norm ES{tag}"] = common.es_normal(s, a, m)
        rec["ES1 hist/norm"] = rec["hist ES1"] / rec["norm ES1"]
        rec["skew"] = stats.skew(p, bias=False)
        rec["exkurt"] = stats.kurtosis(p, bias=False)
        rec["worst month"] = p.min()
        rows.append(rec)
    # The individual assets, for reference
    for i, a in enumerate(ASSETS):
        p = R[:, i]
        m, s = p.mean(), p.std(ddof=1)
        rec = {"portfolio": f"(asset) {a}", "Sharpe (ann)": np.sqrt(12) * sharpe_m(p)}
        for al in (0.05, 0.01):
            hv, he = common.var_es(p, al)
            tag = int(al * 100)
            rec[f"hist VaR{tag}"] = hv
            rec[f"norm VaR{tag}"] = common.var_normal(s, al, m)
            rec[f"hist ES{tag}"] = he
            rec[f"norm ES{tag}"] = common.es_normal(s, al, m)
        rec["ES1 hist/norm"] = rec["hist ES1"] / rec["norm ES1"]
        rec["skew"] = stats.skew(p, bias=False)
        rec["exkurt"] = stats.kurtosis(p, bias=False)
        rec["worst month"] = p.min()
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("portfolio")
    common.header("Problem 4(d) - monthly VaR and ES (% of value, positive = loss)")
    cols1 = ["Sharpe (ann)", "hist VaR5", "norm VaR5", "hist ES5", "norm ES5"]
    cols2 = ["hist VaR1", "norm VaR1", "hist ES1", "norm ES1", "ES1 hist/norm"]
    cols3 = ["skew", "exkurt", "worst month"]
    pct = lambda v: f"{100 * v:.2f}%"
    for cols in (cols1, cols2, cols3):
        sub = out[cols].copy()
        for c in cols:
            if c in ("Sharpe (ann)", "ES1 hist/norm", "skew", "exkurt"):
                sub[c] = sub[c].map(lambda v: f"{v:.3f}")
            else:
                sub[c] = sub[c].map(pct)
        print(sub.to_string())
        print()
    print("Historical: RiskStats.jl estimator on 240 months (1%: N a = 2.4 -> average of"
          "\nthe 2nd and 3rd worst months; ES = mean of the months at or below that).")
    D["risk"] = out
    return out


# --------------------------------------------------------------------------
# (e) CAPM regressions
# --------------------------------------------------------------------------
def part_e(D: dict) -> dict:
    d = D["d"]
    y = d.PUTW.to_numpy() - RF_M
    x = d.EQ.to_numpy() - RF_M
    X = np.column_stack([np.ones_like(x), x])
    full = common.ols(y, X, ["alpha", "beta"])
    dn = x < -0.05
    down = common.ols(y[dn], X[dn], ["alpha", "beta"])
    up = common.ols(y[~dn], X[~dn], ["alpha", "beta"])
    common.header("Problem 4(e) - CAPM of PUTW excess return on EQ excess return")
    for lab, f, n in (("all 240 months", full, len(y)),
                      (f"EQ excess < -5% ({dn.sum()} months)", down, dn.sum()),
                      (f"EQ excess >= -5% ({(~dn).sum()} months)", up, (~dn).sum())):
        print(f"\n{lab}")
        print(f"  alpha = {f['beta'][0]:+.5f}/mo ({12 * f['beta'][0]:+.4f}/yr)  "
              f"se {f['se'][0]:.5f}  t = {f['t'][0]:+.2f}")
        print(f"  beta  = {f['beta'][1]:+.4f}  se {f['se'][1]:.4f}  t = {f['t'][1]:+.2f}  "
              f"R^2 = {f['r2']:.3f}")
    # What a beta-consistent alpha would be: the crash-beta charged on all months
    eq_ex_mean = x.mean()
    print(f"\nmean EQ excess return {eq_ex_mean:+.5f}/mo; mean PUTW excess {y.mean():+.5f}/mo")
    print(f"alpha if PUTW were charged the downside beta on every month: "
          f"{y.mean() - down['beta'][1] * eq_ex_mean:+.5f}/mo")
    D["capm"] = {"full": full, "down": down, "up": up, "n_down": int(dn.sum())}
    return D["capm"]


# --------------------------------------------------------------------------
# (h) estimation error
# --------------------------------------------------------------------------
def part_h(D: dict, n_boot: int = 2000) -> dict:
    R, mu, cov = D["R"], D["mu"], D["cov"]
    n = R.shape[0]
    se = R.std(axis=0, ddof=1) / np.sqrt(n)
    tab = pd.DataFrame({"mean/mo": mu, "se(mean)": se, "t (mean>rf)": (mu - RF_M) / se,
                        "ann mean": 12 * mu, "ann 95% CI lo": 12 * (mu - 1.96 * se),
                        "ann 95% CI hi": 12 * (mu + 1.96 * se)}, index=ASSETS)
    common.header("Problem 4(h) - standard error of each mean monthly return (sd/sqrt(240))")
    common.show(tab, {"mean/mo": "{:.5f}", "se(mean)": "{:.5f}", "t (mean>rf)": "{:+.2f}",
                      "ann mean": "{:+.4f}", "ann 95% CI lo": "{:+.4f}", "ann 95% CI hi": "{:+.4f}"})

    # Sensitivity: move each mean by one standard error and re-solve.
    common.header("Problem 4(h) - long-only tangency weights when one mean moves by +/- 1 se")
    base = max_sr(mu, cov, [(0, 1)] * 4)
    rows = [{"shock": "none", **dict(zip(ASSETS, base))}]
    for i, a in enumerate(ASSETS):
        for sgn in (-1, 1):
            m2 = mu.copy()
            m2[i] += sgn * se[i]
            rows.append({"shock": f"{a} {'+' if sgn > 0 else '-'}1se",
                         **dict(zip(ASSETS, max_sr(m2, cov, [(0, 1)] * 4)))})
    common.show(pd.DataFrame(rows).set_index("shock"), "{:.3f}")

    # Bootstrap: resample the 240 months with replacement, re-estimate mu and
    # Sigma, re-solve the long-only tangency.
    rng = np.random.default_rng(common.SEED)
    boot = np.empty((n_boot, 4))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        Rb = R[idx]
        boot[b] = max_sr(Rb.mean(axis=0), np.cov(Rb, rowvar=False), [(0, 1)] * 4)
    bt = pd.DataFrame(boot, columns=ASSETS)
    summ = pd.DataFrame({"point": base, "boot mean": bt.mean(), "boot sd": bt.std(),
                         "5%": bt.quantile(0.05), "50%": bt.median(), "95%": bt.quantile(0.95),
                         "P(w > 1%)": (bt > 0.01).mean()}, index=ASSETS)
    common.header(f"Problem 4(h) - bootstrap of the long-only tangency weights ({n_boot} resamples)")
    common.show(summ, "{:.3f}")
    # How often does the resample contain the crash month(s)?
    worst = int(np.argmin(D["d"].PUTW.to_numpy()))
    print(f"\nShare of resamples that hold the weight PUTW > 50%: {(bt.PUTW > 0.5).mean():.1%}")
    D["boot"] = bt
    return {"se": tab, "boot": summ}


def figure_boot(D: dict) -> None:
    bt = D["boot"]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    parts = ax.violinplot([bt[a] for a in ASSETS], showmedians=True, widths=0.8)
    for pc, c in zip(parts["bodies"], (C1, C3, C4, C2)):
        pc.set_facecolor(c)
        pc.set_alpha(0.45)
    base = max_sr(D["mu"], D["cov"], [(0, 1)] * 4)
    ax.scatter(np.arange(1, 5), base, color="black", zorder=5, s=25, label="point estimate")
    ax.set_xticks(np.arange(1, 5), ASSETS)
    ax.set_ylabel("long-only tangency weight")
    ax.set_title("Problem 4(h): bootstrapped tangency weights (resampled months)")
    ax.legend(loc="upper left")
    common.finish(fig, "problem4_boot")


def run_all() -> None:
    D = prepare()
    part_a(D)
    figure_predict(D)
    part_c(D)
    figure_frontier(D)
    part_d(D)
    part_e(D)
    part_h(D)
    figure_boot(D)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    common.use_style()
    run_all()

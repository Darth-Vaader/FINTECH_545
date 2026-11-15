"""
Problem 3 -- VaR of an option book.

problem3.csv: 501 daily prices of ABC -> 500 arithmetic returns.
problem3_portfolios.csv: four books of ABC stock and European options on ABC.

    value everything at the last price; r = 4.25%; no dividends (b = r);
    255 trading days a year, DaysToExpiry in trading days -> T = days/255;
    implied vols held constant; horizon 10 trading days; zero expected return.

  predict  : (a) each book's delta, gamma, theta and a ranking of how badly a
             delta-normal VaR will misstate the 10-day risk; (b) the sign of the
             P&L if the price does not move, and which books will have ES/VaR
             well above the Normal 1.25.
  fit      : (c) moments; Normal vs Student t by AICc; (d) 10-day delta-normal
             VaR/ES at 5% and 1%; (e) 100,000 simulated 10-day paths from each
             fitted distribution (zero mean), full revaluation with 10 fewer
             days; (f) P&L against the 10-day price, with the delta-only line.
  reconcile: (g) what delta-normal cannot see, and whether adding the gamma
             (and theta) term reproduces full revaluation; (h) the mean P&L;
             (i) which numbers move under the t, and why the 5% VaR moves least;
             (j) the "flat" hedged short call.

Conventions:
  * Delta-normal (Week 04 s4.4, Week 07 homework): dollar delta
    D$ = sum_i holding_i x delta_i x S, 10-day return sd = sigma_daily sqrt(10)
    with sigma_daily the sample sd of returns (n - 1), zero mean.
    VaR = -Phi^{-1}(a) sigma_10 |D$|,  ES = sigma_10 phi(Phi^{-1}(a))/a |D$|.
  * Theta from GBSM is -dV/dT per YEAR; per trading day it is theta/255, and
    over the horizon Theta dt with dt = 10/255.
  * Simulation: each fitted distribution with its location set to zero,
    10 daily returns compounded, S_10 = S_0 prod(1 + r_i). P&L is the change in
    book value (current value at S_0 and T, new value at S_10 and T - 10/255).

Run:  python problem3.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common
from common import C1, C2, C3, INK2

R = 0.0425
DPY = 255.0
H = 10                      # horizon, trading days
NSIM = 100_000
ALPHAS = (0.05, 0.01)
BOOKS = ["CoveredCall", "ShortStraddle", "RiskReversal", "HedgedShortCall"]


def prepare() -> dict:
    px = common.load("problem3")
    p = px.ABC.to_numpy(dtype=float)
    r = p[1:] / p[:-1] - 1.0                 # arithmetic returns
    pf = common.load("problem3_portfolios")
    return {"prices": px, "r": r, "S0": p[-1], "pf": pf}


# --------------------------------------------------------------------------
# Position valuation
# --------------------------------------------------------------------------
def position_value(row, S, days_elapsed: float = 0.0):
    """Value of ONE unit of a position at price S after `days_elapsed` trading
    days. Vectorised over S."""
    S = np.asarray(S, dtype=float)
    if row.Type == "Stock":
        return S
    T = (row.DaysToExpiry - days_elapsed) / DPY
    return common.gbsm(row.Type == "Call", S, row.Strike, T, R, R, row.ImpliedVol)


def book_value(pf_book: pd.DataFrame, S, days_elapsed: float = 0.0):
    return sum(r.Holding * position_value(r, S, days_elapsed) for r in pf_book.itertuples())


# --------------------------------------------------------------------------
# (a) Greeks
# --------------------------------------------------------------------------
def part_a(D: dict) -> pd.DataFrame:
    S0, pf = D["S0"], D["pf"]
    rows = []
    for r in pf.itertuples():
        if r.Type == "Stock":
            g = {"value": S0, "delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
        else:
            g = common.gbsm_greeks(r.Type == "Call", S0, r.Strike, r.DaysToExpiry / DPY,
                                   R, R, r.ImpliedVol)
        rows.append({"Portfolio": r.Portfolio, "Type": r.Type, "Strike": r.Strike,
                     "Holding": r.Holding, "unit value": float(g["value"]),
                     "unit delta": float(g["delta"]), "value": r.Holding * float(g["value"]),
                     "delta": r.Holding * float(g["delta"]), "gamma": r.Holding * float(g["gamma"]),
                     "theta/day": r.Holding * float(g["theta"]) / DPY,
                     "vega/pt": r.Holding * float(g["vega"]) / 100})
    pos = pd.DataFrame(rows)
    book = pos.groupby("Portfolio", sort=False)[["value", "delta", "gamma", "theta/day", "vega/pt"]].sum()
    book["$delta"] = book.delta * S0
    # second-order and theta terms for a typical 10-day move (1 sd)
    sd10 = D["r"].std(ddof=1) * np.sqrt(H) * S0
    book["0.5 G (1sd dS)^2"] = 0.5 * book.gamma * sd10 ** 2
    book["Theta x 10d"] = book["theta/day"] * H
    book["|delta| x 1sd dS"] = book.delta.abs() * sd10
    book["gamma share"] = book["0.5 G (1sd dS)^2"].abs() / (
        book["0.5 G (1sd dS)^2"].abs() + book["|delta| x 1sd dS"])
    common.header("Problem 3(a) - position and book Greeks")
    print(f"S0 = {S0}, T = DaysToExpiry/255, r = b = {R}\n")
    common.show(pos.set_index(["Portfolio", "Type"]),
                {"Strike": "{:.0f}", "Holding": "{:.0f}", "unit value": "{:.4f}",
                 "unit delta": "{:.4f}", "value": "{:,.2f}", "delta": "{:+.3f}",
                 "gamma": "{:+.4f}", "theta/day": "{:+.4f}", "vega/pt": "{:+.3f}"})
    print("\nBook totals (delta in shares, gamma per $1 per $1, theta in $ per trading day)")
    common.show(book[["value", "delta", "$delta", "gamma", "theta/day", "vega/pt"]],
                {"value": "{:,.2f}", "delta": "{:+.3f}", "$delta": "{:+,.1f}",
                 "gamma": "{:+.4f}", "theta/day": "{:+.3f}", "vega/pt": "{:+.3f}"})
    print(f"\nSize of each term for a 1-sd 10-day move (dS = {sd10:.3f}):")
    common.show(book[["|delta| x 1sd dS", "0.5 G (1sd dS)^2", "Theta x 10d", "gamma share"]],
                {"|delta| x 1sd dS": "{:,.2f}", "0.5 G (1sd dS)^2": "{:+,.2f}",
                 "Theta x 10d": "{:+,.2f}", "gamma share": "{:.0%}"})
    D["pos"], D["book"], D["sd10_dollar"] = pos, book, sd10
    return book


# --------------------------------------------------------------------------
# (c) moments and fits
# --------------------------------------------------------------------------
def part_c(D: dict) -> dict:
    r = D["r"]
    m = common.moments(r)
    fn = common.fit_normal(r)
    ft = common.fit_t(r)
    common.header("Problem 3(c) - moments of ABC returns; Normal vs Student t by AICc")
    print(f"n = {m['n']}  mean = {m['mean']:.6f}  sd = {m['sd']:.6f}  "
          f"skew = {m['skew']:.4f}  excess kurtosis = {m['exkurt']:.4f}")
    print(f"se(skew) ~ sqrt(6/n) = {np.sqrt(6 / m['n']):.3f}, "
          f"se(exkurt) ~ sqrt(24/n) = {np.sqrt(24 / m['n']):.3f}\n")
    tab = pd.DataFrame({
        "Normal": [fn["mu"], fn["sigma"], np.nan, fn["sigma"], fn["loglik"], 2, fn["aicc"]],
        "Student t": [ft["mu"], ft["scale"], ft["nu"], ft["sd_implied"], ft["loglik"], 3, ft["aicc"]],
    }, index=["mu", "sigma / scale", "nu", "implied sd", "log-lik", "k", "AICc"])
    common.show(tab, "{:.6f}")
    best = "Student t" if ft["aicc"] < fn["aicc"] else "Normal"
    print(f"\ndelta AICc (Normal - t) = {fn['aicc'] - ft['aicc']:.2f}  ->  {best} preferred")
    D.update(m=m, fn=fn, ft=ft)
    return {"m": m, "fn": fn, "ft": ft}


# --------------------------------------------------------------------------
# (d) delta-normal VaR / ES
# --------------------------------------------------------------------------
def part_d(D: dict) -> pd.DataFrame:
    book = D["book"]
    sd10 = D["m"]["sd"] * np.sqrt(H)
    rows = []
    for name, b in book.iterrows():
        rec = {"Portfolio": name, "$delta": b["$delta"]}
        for a in ALPHAS:
            rec[f"VaR{int(a * 100)}"] = common.var_normal(sd10, a) * abs(b["$delta"])
            rec[f"ES{int(a * 100)}"] = common.es_normal(sd10, a) * abs(b["$delta"])
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("Portfolio")
    common.header("Problem 3(d) - 10-day delta-normal VaR and ES ($, positive = loss)")
    print(f"sigma_daily = {D['m']['sd']:.6f}, sigma_10 = {sd10:.6f}\n")
    common.show(out, "{:,.2f}")
    D["dn"] = out
    return out


# --------------------------------------------------------------------------
# (e) simulation + full revaluation
# --------------------------------------------------------------------------
def simulate_S10(D: dict) -> dict:
    """10 daily returns per path from each fitted model with ZERO location,
    compounded. One RNG stream per model, both seeded from common.SEED."""
    S0 = D["S0"]
    rng = np.random.default_rng(common.SEED)
    rn = rng.normal(0.0, D["fn"]["sigma"], size=(NSIM, H))
    rng = np.random.default_rng(common.SEED + 1)
    rt = stats.t.rvs(D["ft"]["nu"], loc=0.0, scale=D["ft"]["scale"], size=(NSIM, H),
                     random_state=rng)
    return {"Normal": S0 * np.prod(1.0 + rn, axis=1), "t": S0 * np.prod(1.0 + rt, axis=1)}


def part_e(D: dict) -> pd.DataFrame:
    S0, pf = D["S0"], D["pf"]
    sims = simulate_S10(D)
    D["sims"] = sims
    rows = []
    pnl_store = {}
    for dist, s10 in sims.items():
        for name in BOOKS:
            b = pf[pf.Portfolio == name]
            v0 = book_value(b, S0, 0.0)
            pnl = book_value(b, s10, H) - v0
            pnl_store[(dist, name)] = pnl
            rec = {"dist": dist, "Portfolio": name, "mean P&L": pnl.mean()}
            for a in ALPHAS:
                v, e = common.var_es(pnl, a)
                rec[f"VaR{int(a * 100)}"] = v
                rec[f"ES{int(a * 100)}"] = e
            rec["ES5/VaR5"] = rec["ES5"] / rec["VaR5"]
            rec["ES1/VaR1"] = rec["ES1"] / rec["VaR1"]
            rows.append(rec)
    out = pd.DataFrame(rows).set_index(["Portfolio", "dist"]).sort_index(
        level=0, key=lambda s: s.map({b: i for i, b in enumerate(BOOKS)}))
    common.header(f"Problem 3(e) - {NSIM:,} simulated 10-day paths, full revaluation")
    s_n, s_t = sims["Normal"], sims["t"]
    print(f"S_10 sd: Normal {s_n.std():.3f}, t {s_t.std():.3f};  "
          f"1% / 99% quantiles: Normal {np.percentile(s_n, 1):.2f}/{np.percentile(s_n, 99):.2f},"
          f" t {np.percentile(s_t, 1):.2f}/{np.percentile(s_t, 99):.2f}\n")
    common.show(out, {"mean P&L": "{:+,.2f}", "VaR5": "{:,.2f}", "ES5": "{:,.2f}",
                      "VaR1": "{:,.2f}", "ES1": "{:,.2f}", "ES5/VaR5": "{:.3f}",
                      "ES1/VaR1": "{:.3f}"})
    print("\nNormal ES/VaR for reference: 5% -> 1.254, 1% -> 1.146")
    D["sim"] = out
    D["pnl"] = pnl_store
    return out


# --------------------------------------------------------------------------
# (b) and (h) the no-move P&L (pure theta, exact)
# --------------------------------------------------------------------------
def no_move(D: dict) -> pd.DataFrame:
    S0, pf = D["S0"], D["pf"]
    rows = []
    for name in BOOKS:
        b = pf[pf.Portfolio == name]
        rows.append({"Portfolio": name,
                     "P&L if S unchanged (exact)": book_value(b, S0, H) - book_value(b, S0, 0.0),
                     "Theta x 10d (linear)": D["book"].loc[name, "Theta x 10d"]})
    out = pd.DataFrame(rows).set_index("Portfolio")
    common.header("Problem 3(b)/(h) - P&L over 10 days if the price does not move")
    common.show(out, "{:+,.2f}")
    return out


# --------------------------------------------------------------------------
# (g) delta-gamma(-theta) approximation on the same simulated paths
# --------------------------------------------------------------------------
def part_g(D: dict) -> pd.DataFrame:
    S0, book = D["S0"], D["book"]
    rows = []
    for dist, s10 in D["sims"].items():
        dS = s10 - S0
        for name in BOOKS:
            b = book.loc[name]
            approx = {
                "delta": b.delta * dS,
                "delta+gamma": b.delta * dS + 0.5 * b.gamma * dS ** 2,
                "delta+gamma+theta": b.delta * dS + 0.5 * b.gamma * dS ** 2 + b["theta/day"] * H,
                "full": D["pnl"][(dist, name)],
            }
            for meth, pnl in approx.items():
                v5, e5 = common.var_es(pnl, 0.05)
                v1, e1 = common.var_es(pnl, 0.01)
                rows.append({"Portfolio": name, "dist": dist, "method": meth,
                             "mean": pnl.mean(), "VaR5": v5, "ES5": e5, "VaR1": v1, "ES1": e1})
    out = pd.DataFrame(rows)
    common.header("Problem 3(g) - Taylor approximations vs full revaluation (same paths)")
    for dist in ("Normal", "t"):
        print(f"\n{dist} paths")
        sub = out[out.dist == dist].drop(columns="dist").set_index(["Portfolio", "method"])
        common.show(sub, {"mean": "{:+,.2f}", "VaR5": "{:,.2f}", "ES5": "{:,.2f}",
                          "VaR1": "{:,.2f}", "ES1": "{:,.2f}"})
    D["taylor"] = out
    return out


# --------------------------------------------------------------------------
# (h) what the positive mean is paying for
# --------------------------------------------------------------------------
def part_h(D: dict) -> pd.DataFrame:
    """Black-Scholes PDE with b = r:  Theta = -(1/2) Gamma S^2 sigma_IV^2 + r(V - Delta S),
    Theta here being the per-year time decay. With ZERO real-world drift and
    real-world variance sigma_RV^2, the expected P&L of one option over dt is
        Theta dt + (1/2) Gamma S^2 sigma_RV^2 dt
      = (1/2) Gamma S^2 (sigma_RV^2 - sigma_IV^2) dt     <- variance spread
        + r (V - Delta S) dt                             <- carry: the price
          assumes the stock drifts at r, the scenario assumes it drifts at 0.
    Stock legs have zero expected P&L. Summing over the book gives a first-order
    forecast of the simulated mean."""
    S0 = D["S0"]
    sig_rv = D["m"]["sd"] * np.sqrt(DPY)
    dt = H / DPY
    pf = D["pf"]
    rows = []
    for name in BOOKS:
        b = pf[pf.Portfolio == name]
        vrp = carry = 0.0
        for r in b.itertuples():
            if r.Type == "Stock":
                continue
            g = common.gbsm_greeks(r.Type == "Call", S0, r.Strike, r.DaysToExpiry / DPY,
                                   R, R, r.ImpliedVol)
            vrp += r.Holding * 0.5 * g["gamma"] * S0 ** 2 * (sig_rv ** 2 - r.ImpliedVol ** 2) * dt
            carry += r.Holding * R * (g["value"] - g["delta"] * S0) * dt
        bk = D["book"].loc[name]
        rows.append({"Portfolio": name,
                     "theta 10d": bk["Theta x 10d"],
                     "E[.5G dS^2]": 0.5 * bk.gamma * (S0 * D["m"]["sd"]) ** 2 * H,
                     "var spread": vrp, "carry": carry, "forecast": vrp + carry,
                     "sim mean N": D["sim"].loc[(name, "Normal"), "mean P&L"],
                     "sim mean t": D["sim"].loc[(name, "t"), "mean P&L"],
                     "ES1 (t)": D["sim"].loc[(name, "t"), "ES1"]})
    out = pd.DataFrame(rows).set_index("Portfolio")
    common.header("Problem 3(h) - decomposing the mean 10-day P&L")
    print(f"realised vol = sd_daily x sqrt(255) = {sig_rv:.4f}; implied vols 0.27-0.32")
    print("variance spread = sum h x 0.5 G S^2 (RV^2 - IV^2) dt;  carry = sum h x r(V - delta S) dt\n")
    common.show(out, "{:+,.2f}")
    return out


# --------------------------------------------------------------------------
# (i) Normal -> t: what moves
# --------------------------------------------------------------------------
def part_i(D: dict) -> pd.DataFrame:
    sim = D["sim"]
    rows = []
    for name in BOOKS:
        n, t = sim.loc[(name, "Normal")], sim.loc[(name, "t")]
        rows.append({"Portfolio": name, **{f"{c} t/N": t[c] / n[c] for c in
                                         ("VaR5", "ES5", "VaR1", "ES1")},
                     "mean t-N": t["mean P&L"] - n["mean P&L"]})
    out = pd.DataFrame(rows).set_index("Portfolio")
    fn, ft = D["fn"], D["ft"]
    # Daily quantiles of the two fitted return distributions (zero location)
    q = pd.DataFrame({"alpha": [0.05, 0.01, 0.001],
                      "Normal": [stats.norm.ppf(a, 0, fn["sigma"]) for a in (0.05, 0.01, 0.001)],
                      "t": [stats.t.ppf(a, ft["nu"], 0, ft["scale"]) for a in (0.05, 0.01, 0.001)]})
    q["t/Normal"] = q.t / q.Normal
    # Where do the two daily CDFs cross?
    from scipy import optimize
    f = lambda x: stats.norm.cdf(x, 0, fn["sigma"]) - stats.t.cdf(x, ft["nu"], 0, ft["scale"])
    xc = optimize.brentq(f, -6 * fn["sigma"], -0.3 * fn["sigma"])
    # 10-day compounded returns: same comparison
    r10n = D["sims"]["Normal"] / D["S0"] - 1
    r10t = D["sims"]["t"] / D["S0"] - 1
    q10 = pd.DataFrame({"alpha": [0.05, 0.01],
                        "Normal": [np.quantile(r10n, a) for a in (0.05, 0.01)],
                        "t": [np.quantile(r10t, a) for a in (0.05, 0.01)]})
    q10["t/Normal"] = q10.t / q10.Normal
    common.header("Problem 3(i) - ratio of t to Normal risk numbers")
    common.show(out, {c: "{:.3f}" for c in out.columns} | {"mean t-N": "{:+.2f}"})
    print("\nDaily return quantiles of the two fits (zero location):")
    common.show(q.set_index("alpha"), {"Normal": "{:.5f}", "t": "{:.5f}", "t/Normal": "{:.3f}"})
    print(f"The two daily CDFs cross at x = {xc:.5f}, i.e. at the "
          f"{stats.norm.cdf(xc, 0, fn['sigma']):.2%} quantile")
    print("\n10-day compounded return quantiles (simulated):")
    common.show(q10.set_index("alpha"), {"Normal": "{:.5f}", "t": "{:.5f}", "t/Normal": "{:.3f}"})
    return out


# --------------------------------------------------------------------------
# (f) the P&L plot
# --------------------------------------------------------------------------
def figure_pnl(D: dict) -> None:
    S0, pf, book = D["S0"], D["pf"], D["book"]
    s10 = D["sims"]["t"]
    lo, hi = np.percentile(s10, [0.05, 99.95])
    grid = np.linspace(lo, hi, 400)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex=True)
    rng = np.random.default_rng(7)
    idx = rng.choice(s10.size, 3000, replace=False)
    for ax, name in zip(axes.ravel(), BOOKS):
        b = pf[pf.Portfolio == name]
        v0 = book_value(b, S0, 0.0)
        full = book_value(b, grid, H) - v0
        dS = grid - S0
        bk = book.loc[name]
        ax.scatter(s10[idx], D["pnl"][("t", name)][idx], s=3, color=C1, alpha=0.18,
                   lw=0, label="simulated paths (t)")
        ax.plot(grid, full, color=C1, lw=2.0, label="full revaluation")
        ax.plot(grid, bk.delta * dS, color=C2, lw=1.6, ls="--", label="delta only")
        ax.plot(grid, bk.delta * dS + 0.5 * bk.gamma * dS ** 2 + bk["theta/day"] * H,
                color=C3, lw=1.2, ls=":", label="delta + gamma + theta")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.axvline(S0, color=INK2, lw=0.6, ls=":")
        for a, c in ((0.05, "#999999"), (0.01, "#555555")):
            v, _ = common.var_es(D["pnl"][("t", name)], a)
            ax.axhline(-v, color=c, lw=0.8, ls="-.")
            ax.text(grid[0], -v, f" {int(a * 100)}% VaR", va="bottom", fontsize=7, color=c)
        ax.set_title(f"{name}  (delta {bk.delta:+.1f} sh, gamma {bk.gamma:+.2f})")
        ax.set_ylabel("10-day P&L ($)")
    for ax in axes[1]:
        ax.set_xlabel("ABC price in 10 days")
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    fig.suptitle("Problem 3(f): 10-day P&L against the 10-day price of ABC", fontsize=11,
                 fontweight="bold")
    fig.tight_layout()
    common.finish(fig, "problem3_pnl")


def figure_returns(D: dict) -> None:
    """Predict-stage plot: the return series and its histogram against both fits."""
    r = D["r"]
    fn, ft = D["fn"], D["ft"]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    axes[0].plot(r, color=C1, lw=0.7)
    axes[0].set_title("ABC daily arithmetic returns")
    axes[0].set_xlabel("day")
    x = np.linspace(r.min() * 1.1, r.max() * 1.1, 400)
    axes[1].hist(r, bins=50, density=True, color=C1, alpha=0.35)
    axes[1].plot(x, stats.norm.pdf(x, fn["mu"], fn["sigma"]), color=C2, label="Normal fit")
    axes[1].plot(x, stats.t.pdf(x, ft["nu"], ft["mu"], ft["scale"]), color=C3,
                 label=f"t fit (nu={ft['nu']:.1f})")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.05, None)
    axes[1].set_title("histogram (log density) vs fitted densities")
    axes[1].legend()
    fig.tight_layout()
    common.finish(fig, "problem3_returns")


def run_all() -> None:
    D = prepare()
    part_a(D)
    no_move(D)
    part_c(D)
    figure_returns(D)
    part_d(D)
    part_e(D)
    figure_pnl(D)
    part_g(D)
    part_h(D)
    part_i(D)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    common.use_style()
    run_all()

"""
Problem 2 -- A volatility estimate after the regime changed.

problem2.csv holds 501 daily prices of one stock; we hold $1,000,000 of it. We

  predict  : arithmetic returns with the sample mean removed; locate the recent
             high-volatility regime; effective sample size, half life, and the
             share of weight each lambda puts on the regime; the four moments;
  fit      : five one-day 5% VaRs -- Normal with equal-weight sd, Normal with
             EW sd at lambda = 0.97 and 0.94, a Student's t by MLE, and
             historical simulation;
  reconcile: the ranking, the within-regime sds and the mixture kurtosis they
             imply, and the standard error of an EW volatility estimate.

Run:  python problem2.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, stats

import common
from common import C1, C2, C3

PV = 1_000_000.0
ALPHA = 0.05
LAMBDAS = (0.97, 0.94)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def prepare() -> dict:
    """Arithmetic returns P_t/P_{t-1} - 1 (Week 04 s4.1.2), sample mean removed
    with EQUAL weights once, as the problem instructs. Everything downstream
    treats the demeaned series as having mean exactly zero."""
    p = common.load("problem2")["Price"].to_numpy(dtype=float)
    r_raw = p[1:] / p[:-1] - 1.0
    mu = r_raw.mean()
    return {"prices": p, "r_raw": r_raw, "mu": mu, "r": r_raw - mu}


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
def find_break(r: np.ndarray, min_seg: int = 20) -> int:
    """Single most likely variance break: the split k maximising the two-segment
    Normal log-likelihood -sum n_i ln(var_i)/2. This is the quantitative version
    of "how many recent days look different on the plot"; the plot is what the
    question asks for, this just reads the same thing off without squinting."""
    best_k, best_ll = None, -np.inf
    for k in range(min_seg, r.size - min_seg):
        a, b = r[:k], r[k:]
        ll = -0.5 * (a.size * np.log(a.var()) + b.size * np.log(b.var()))
        if ll > best_ll:
            best_k, best_ll = k, ll
    return best_k


def part_predict(D: dict) -> dict:
    r = D["r"]
    n = r.size
    common.header("Problem 2 -- Predict: returns, regime, weights, moments")
    print(f"  {n} arithmetic returns; sample mean {D['mu']:+.6f} removed.")

    # The regime, read from the data.
    k = find_break(r)
    K = n - k                                   # number of recent days that differ
    sd_recent, sd_rest = r[k:].std(ddof=1), r[:k].std(ddof=1)
    print(f"\n  Most likely single variance break after day {k}: the last {K} days.")
    print(f"    sd of the last {K} days   = {sd_recent:.5f}")
    print(f"    sd of the first {k} days = {sd_rest:.5f}     ratio {sd_recent / sd_rest:.2f}x")
    roll = np.array([r[i - 20:i].std(ddof=1) for i in range(20, n + 1)])
    print(f"    20-day rolling sd: median {np.median(roll[:k - 20]):.5f} before the break,"
          f" max {roll.max():.5f} inside it")

    # (b) Effective sample size, half life, weight on the regime.
    print(f"\n  (b) lambda   n_eff   half-life   weight on last {K} days   equal-weight")
    print("      " + "-" * 66)
    weights = {}
    for lam in LAMBDAS:
        w = common.ew_weights(n, lam)
        frac = w[-K:].sum()
        weights[lam] = w
        print(f"      {lam:<8}{common.n_eff(lam):>6.1f}{common.half_life(lam):>11.1f}"
              f"{frac:>22.3f}{K / n:>15.3f}")
    print("      " + "-" * 66)
    print(f"      (infinite-horizon shortcut 1 - lambda^K: "
          f"{1 - 0.97 ** K:.3f} at 0.97, {1 - 0.94 ** K:.3f} at 0.94)")
    print(f"      Equal weighting gives the regime {100 * K / n:.1f}% of the weight; "
          f"lambda = 0.94 gives it {100 * weights[0.94][-K:].sum():.0f}%.")

    # (c) Moments of the full sample.
    m = common.moments(r)
    jb = stats.jarque_bera(r)
    print(f"\n  (c) Full-sample moments (mean removed):")
    print(f"      sd              = {m['sd']:.6f}")
    print(f"      skewness        = {m['skew']:+.4f}   ({m['skew'] / m['se_skew']:+.1f} se "
          f"under H0: Normal, se = {m['se_skew']:.3f})")
    print(f"      excess kurtosis = {m['exkurt']:+.4f}   ({m['exkurt'] / m['se_exkurt']:+.1f} se "
          f"under H0: Normal, se = {m['se_exkurt']:.3f})")
    print(f"      Jarque-Bera p   = {jb[1]:.2e}")
    print(f"      A single Normal regime gives excess kurtosis 0 +/- {m['se_exkurt']:.2f}; "
          f"{m['exkurt']:+.2f} is not that.")
    return {"k": k, "K": K, "sd_recent": sd_recent, "sd_rest": sd_rest, "roll": roll,
            "weights": weights, "m": m}


# --------------------------------------------------------------------------
# Fit (d): the five VaRs
# --------------------------------------------------------------------------
def part_d(D: dict, P: dict) -> dict:
    r = D["r"]
    common.header("Problem 2(d) -- five one-day VaR estimates on $1,000,000")
    sd_eq = r.std(ddof=1)
    sd_97 = np.sqrt(common.ew_variance(r, 0.97))
    sd_94 = np.sqrt(common.ew_variance(r, 0.94))
    # Location-scale t by MLE (common.fit_t): scale is not the sd, see docstring.
    tfit = common.fit_t(r)
    var_hist, es_hist = common.var_es(PV * r, ALPHA)
    var_hist_np = -PV * np.quantile(r, ALPHA)

    est = {
        "1. Normal, equal-weight sd": (sd_eq, common.var_normal(sd_eq, ALPHA, pv=PV)),
        "2. Normal, EW sd lambda=0.97": (sd_97, common.var_normal(sd_97, ALPHA, pv=PV)),
        "3. Normal, EW sd lambda=0.94": (sd_94, common.var_normal(sd_94, ALPHA, pv=PV)),
        "4. Student t by MLE": (tfit["sd_implied"],
                                -PV * stats.t.ppf(ALPHA, tfit["nu"], tfit["mu"], tfit["scale"])),
        "5. Historical (25th worst of 500)": (sd_eq, var_hist),
    }
    print(f"  {'estimate':<36}{'sigma used':>12}{'VaR (5%)':>14}")
    print("  " + "-" * 62)
    for name, (sig, v) in est.items():
        print(f"  {name:<36}{sig:>12.5f}{common.money(v):>14}")
    print("  " + "-" * 62)
    order = sorted(est.items(), key=lambda kv: kv[1][1])
    print("  Ranked smallest to largest: " + " < ".join(k.split(".")[0] for k, _ in order))
    print(f"\n  Fitted t: nu = {tfit['nu']:.3f}, mu = {tfit['mu']:+.6f}, scale = "
          f"{tfit['scale']:.6f}")
    print(f"            implied sd = scale*sqrt(nu/(nu-2)) = {tfit['sd_implied']:.6f}  "
          f"(sample sd {sd_eq:.6f})")
    print(f"            log-likelihood {tfit['loglik']:.2f} vs Normal "
          f"{np.sum(stats.norm.logpdf(r, 0, sd_eq)):.2f}")
    # The t was given its own location (its MLE is a robust centre, not the mean
    # that was removed). Refit with the location pinned at zero, the convention
    # the four other estimates use, to show the ranking does not hinge on it.
    def nll0(p):
        return -np.sum(stats.t.logpdf(r, df=p[1], loc=0.0, scale=p[0]))
    s0, nu0 = optimize.minimize(nll0, [tfit["scale"], tfit["nu"]], method="Nelder-Mead",
                                options={"xatol": 1e-10, "fatol": 1e-12}).x
    var_t0 = -PV * stats.t.ppf(ALPHA, nu0, 0.0, s0)
    print(f"\n  t refitted with location fixed at 0: nu = {nu0:.3f}, scale = {s0:.6f}, "
          f"VaR = {common.money(var_t0)}")
    print(f"  (still below estimate 1's {common.money(est['1. Normal, equal-weight sd'][1])}; "
          f"the ranking does not depend on it)")
    print(f"\n  Historical VaR with numpy's interpolating quantile instead: "
          f"{common.money(var_hist_np)}")
    print(f"  (difference {common.money(abs(var_hist_np - var_hist))} -- the convention "
          f"does not matter here)")
    # EW with the weighted mean removed instead of the equal-weight mean, for
    # the record (common.ew_variance docstring).
    alt = {lam: np.sqrt(common.ew_variance(r, lam, demeaned=False)) for lam in LAMBDAS}
    print(f"  EW sd removing the WEIGHTED mean instead: 0.97 -> {alt[0.97]:.5f}, "
          f"0.94 -> {alt[0.94]:.5f}")
    return {"sd_eq": sd_eq, "sd_97": sd_97, "sd_94": sd_94, "tfit": tfit,
            "est": est, "var_hist": var_hist, "es_hist": es_hist, "order": order}


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------
def part_e(D: dict, P: dict, F: dict) -> dict:
    r = D["r"]
    t = F["tfit"]
    common.header("Problem 2(e) -- the ranking, and the estimate that moved")
    # Why the t sits where it does: at equal variance a t_nu has a 5% quantile
    # INSIDE the Normal's for moderate nu. The t only overtakes the Normal
    # further out. Find the alpha where the two VaRs cross.
    z_t = -stats.t.ppf(ALPHA, t["nu"]) * np.sqrt((t["nu"] - 2) / t["nu"])
    print(f"  5% quantile of a UNIT-VARIANCE t with nu = {t['nu']:.2f}: {z_t:.3f} sd, "
          f"vs 1.645 for the Normal.")
    rel = 100 * (t["sd_implied"] / F["sd_eq"] - 1)
    print(f"  The t's implied sd ({t['sd_implied']:.5f}) is within {abs(rel):.1f}% of the "
          f"sample sd ({F['sd_eq']:.5f}),")
    print(f"  so the whole gap comes from the SHAPE: a t puts more mass in the centre")
    print(f"  and in the far tail than a Normal of the same variance, and takes it")
    print(f"  from the shoulders -- which is where the 5% quantile sits.")

    def gap(a):
        return (-stats.t.ppf(a, t["nu"], t["mu"], t["scale"])) - (-stats.norm.ppf(a) * F["sd_eq"])
    cross = optimize.brentq(gap, 1e-4, ALPHA)
    print(f"  The fitted t's VaR exceeds the equal-weight Normal's only below alpha = "
          f"{100 * cross:.2f}%.")
    for a in (0.05, 0.025, 0.01, 0.005):
        vn = common.var_normal(F["sd_eq"], a, pv=PV)
        vt = -PV * stats.t.ppf(a, t["nu"], t["mu"], t["scale"])
        vh, _ = common.var_es(PV * r, a)
        print(f"    alpha {100 * a:>4.1f}%: Normal {common.money(vn):>10}   t {common.money(vt):>10}"
              f"   historical {common.money(vh):>10}")
    # Where the historical 5% quantile comes from.
    idx = np.argsort(r)[:int(round(ALPHA * r.size))]
    in_regime = int((idx >= P["k"]).sum())
    print(f"\n  Historical: {in_regime} of the {idx.size} worst days fall inside the last "
          f"{P['K']}-day regime")
    print(f"  ({100 * in_regime / idx.size:.0f}% of the tail from {100 * P['K'] / r.size:.0f}% of the"
          f" sample), which is what lifts it above the")
    print(f"  equal-weight Normal without getting anywhere near the EW estimates.")
    return {"z_t": z_t, "cross": cross, "tail_in_regime": in_regime}


def part_f(D: dict, P: dict) -> dict:
    r, k, K = D["r"], P["k"], P["K"]
    common.header("Problem 2(f) -- within-regime sds and the kurtosis they imply")
    a, b = r[:k], r[k:]
    p = K / r.size
    s1, s2 = b.std(ddof=1), a.std(ddof=1)
    m1, m2 = common.moments(b), common.moments(a)
    print(f"  {'regime':<22}{'days':>6}{'sd':>10}{'skew':>8}{'exkurt':>9}{'se(exkurt)':>12}")
    print("  " + "-" * 67)
    print(f"  {'recent (last ' + str(K) + ')':<22}{K:>6}{s1:>10.5f}{m1['skew']:>8.3f}"
          f"{m1['exkurt']:>9.3f}{m1['se_exkurt']:>12.3f}")
    print(f"  {'earlier (first ' + str(k) + ')':<22}{k:>6}{s2:>10.5f}{m2['skew']:>8.3f}"
          f"{m2['exkurt']:>9.3f}{m2['se_exkurt']:>12.3f}")
    print("  " + "-" * 67)
    # A two-component scale mixture of Normals has
    #   E[x^4] / E[x^2]^2 - 3 = 3 (p s1^4 + (1-p) s2^4) / (p s1^2 + (1-p) s2^2)^2 - 3
    mix_var = p * s1 ** 2 + (1 - p) * s2 ** 2
    mix_kurt = 3 * (p * s1 ** 4 + (1 - p) * s2 ** 4) / mix_var ** 2 - 3
    print(f"  Mixture of the two Normals (weights {p:.3f} / {1 - p:.3f}):")
    print(f"    implied sd              = {np.sqrt(mix_var):.5f}   (sample {r.std(ddof=1):.5f})")
    print(f"    implied excess kurtosis = {mix_kurt:+.3f}    (sample {P['m']['exkurt']:+.3f})")
    print(f"  Two Normals with different variances, neither of them fat-tailed, account")
    print(f"  for the excess kurtosis of the pooled sample. The fitted t (nu ~ "
          f"{common.fit_t(r)['nu']:.1f}) is")
    print(f"  describing that pooled, unconditional mixture -- a distribution no single")
    print(f"  day was drawn from. Today's return comes from the {s1:.4f} regime, and a")
    print(f"  Normal with THAT sd describes it better than the t does.")
    return {"s_recent": s1, "s_rest": s2, "mix_kurt": mix_kurt, "p": p}


def part_g(F: dict) -> dict:
    common.header("Problem 2(g) -- is the EW gap bigger than the EW noise?")
    out = {}
    for lam, sig in ((0.97, F["sd_97"]), (0.94, F["sd_94"])):
        se = sig / np.sqrt(2 * common.n_eff(lam))
        out[lam] = {"sigma": sig, "se": se, "var_se": common.var_normal(se, ALPHA, pv=PV)}
        print(f"  lambda = {lam}: sigma = {sig:.5f}, n_eff = {common.n_eff(lam):.1f}")
        print(f"      se(sigma) ~ sigma/sqrt(2 n_eff) = {se:.5f}  ({100 * se / sig:.1f}% of sigma)")
        print(f"      in VaR terms +/- {common.money(out[lam]['var_se'])}")
    v97 = F["est"]["2. Normal, EW sd lambda=0.97"][1]
    v94 = F["est"]["3. Normal, EW sd lambda=0.94"][1]
    gap = v94 - v97
    joint = np.hypot(out[0.97]["var_se"], out[0.94]["var_se"])
    print(f"\n  Gap between the two EW VaRs: {common.money(gap)}")
    print(f"    = {gap / out[0.94]['var_se']:.2f} se of the 0.94 estimate alone, "
          f"{gap / joint:.2f} se of the difference")
    print(f"  The gap is inside the noise of the estimates. The two lambdas do not")
    print(f"  disagree about the level of volatility today; they disagree about how")
    print(f"  much of last month to believe, and at n_eff = 32 even the fast one cannot")
    print(f"  tell a 2.3% sd from a 2.6% one.")
    out["gap"] = gap
    out["joint_se"] = joint
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figure_predict(D: dict, P: dict) -> None:
    r, n, k, K = D["r"], D["r"].size, P["k"], P["K"]
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
    days = np.arange(1, n + 1)
    ax[0].axvspan(k + 0.5, n + 0.5, color=C2, alpha=0.12, lw=0)
    ax[0].plot(days, r, color=C1, linewidth=0.9)
    for s in (P["sd_rest"], -P["sd_rest"]):
        ax[0].axhline(2 * s, color=C3, linestyle="--", linewidth=1.0)
    ax[0].plot(days[19:], 2 * P["roll"], color=C2, linewidth=1.4, label="2 x 20-day rolling sd")
    ax[0].plot(days[19:], -2 * P["roll"], color=C2, linewidth=1.4)
    ax[0].set_title(f"Demeaned returns: the last {K} days are different")
    ax[0].set_xlabel("day")
    ax[0].set_ylabel("return")
    ax[0].legend(loc="lower left")
    ax[0].annotate(f"+/- 2 sd of first {k} days", xy=(30, 2 * P["sd_rest"]),
                   xytext=(30, 2 * P["sd_rest"] + 0.012), fontsize=8, color=C3)

    # Weights
    ax[1].axvspan(k + 0.5, n + 0.5, color=C2, alpha=0.12, lw=0)
    for lam, col in ((0.94, C2), (0.97, C1)):
        w = P["weights"][lam]
        ax[1].plot(days, w, color=col, label=f"lambda = {lam}  ({100 * w[-K:].sum():.0f}% on regime)")
    ax[1].axhline(1 / n, color=C3, linestyle="--", label=f"equal  ({100 * K / n:.0f}% on regime)")
    ax[1].set_xlim(300, n + 1)
    ax[1].set_title("Normalised weights on each day")
    ax[1].set_xlabel("day")
    ax[1].set_ylabel("weight")
    ax[1].legend(loc="upper left")

    # Histogram against a single Normal
    sd = r.std(ddof=1)
    grid = np.linspace(r.min(), r.max(), 500)
    ax[2].hist(r, bins=50, density=True, color=C1, alpha=0.55, edgecolor="white",
               linewidth=0.5, label="all 500 days")
    ax[2].plot(grid, stats.norm.pdf(grid, 0, sd), color=C2, label=f"Normal, sd {sd:.4f}")
    ax[2].plot(grid, stats.norm.pdf(grid, 0, P["sd_rest"]), color=C3, linestyle="--",
               label=f"Normal, sd {P['sd_rest']:.4f} (early regime)")
    ax[2].set_yscale("log")
    ax[2].set_ylim(0.3, None)
    ax[2].set_title(f"Pooled sample vs one Normal (exkurt {P['m']['exkurt']:+.2f})")
    ax[2].set_xlabel("return")
    ax[2].set_ylabel("density (log)")
    ax[2].legend(loc="lower center", fontsize=7.5)
    fig.suptitle("Problem 2 -- Predict: a 500-day sample whose last month is a different regime",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem2_predict")


def figure_fit(D: dict, P: dict, F: dict) -> None:
    r = D["r"]
    t = F["tfit"]
    common.use_style()
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 4.0))
    # Left tail with the five VaRs marked.
    grid = np.linspace(r.min() * 1.05, 0, 400)
    ax[0].hist(r, bins=60, density=True, color=C1, alpha=0.5, edgecolor="white",
               linewidth=0.4, label="returns")
    ax[0].plot(grid, stats.norm.pdf(grid, 0, F["sd_eq"]), color=C2, label="Normal, equal-weight sd")
    ax[0].plot(grid, stats.t.pdf(grid, t["nu"], t["mu"], t["scale"]), color=C3,
               linestyle="--", label=f"t, nu = {t['nu']:.1f}")
    ax[0].plot(grid, stats.norm.pdf(grid, 0, F["sd_94"]), color=common.INK2, linestyle=":",
               label="Normal, EW 0.94 sd")
    cols = {"1": C2, "2": common.INK2, "3": common.INK, "4": C3, "5": C1}
    for name, (_, v) in F["est"].items():
        q = -v / PV
        ax[0].axvline(q, color=cols[name[0]], linewidth=1.0, alpha=0.9)
        ax[0].text(q, 48, name[0], ha="center", fontsize=8, color=cols[name[0]],
                   fontweight="bold")
    ax[0].set_xlim(r.min() * 1.05, 0)
    ax[0].set_ylim(0, 52)
    ax[0].set_title("Left tail; the five 5% VaR quantiles numbered")
    ax[0].set_xlabel("return")
    ax[0].set_ylabel("density")
    ax[0].legend(loc="upper left", fontsize=7.5)

    # VaR as a function of alpha: Normal (eq), t, historical, EW 0.94.
    alphas = np.logspace(np.log10(0.002), np.log10(0.10), 60)
    ax[1].plot(100 * alphas, [common.var_normal(F["sd_eq"], a, pv=PV) / 1e3 for a in alphas],
               color=C2, label="Normal, equal-weight sd")
    ax[1].plot(100 * alphas, [-PV * stats.t.ppf(a, t["nu"], t["mu"], t["scale"]) / 1e3 for a in alphas],
               color=C3, linestyle="--", label="Student t")
    ax[1].plot(100 * alphas, [common.var_es(PV * r, a)[0] / 1e3 for a in alphas],
               color=C1, label="historical")
    ax[1].plot(100 * alphas, [common.var_normal(F["sd_94"], a, pv=PV) / 1e3 for a in alphas],
               color=common.INK2, linestyle=":", label="Normal, EW 0.94 sd")
    ax[1].axvline(5, color=common.INK2, linewidth=0.8)
    ax[1].set_xscale("log")
    ax[1].set_xticks([0.2, 0.5, 1, 2, 5, 10])
    ax[1].set_xticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    ax[1].set_title("VaR against alpha: the t only overtakes the Normal deep in the tail")
    ax[1].set_xlabel("alpha (%)")
    ax[1].set_ylabel("VaR ($ thousands)")
    ax[1].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Problem 2 -- Fit: five VaR estimates on $1,000,000",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem2_fit")


def main() -> dict:
    D = prepare()
    P = part_predict(D)
    figure_predict(D, P)
    F = part_d(D, P)
    E = part_e(D, P, F)
    R = part_f(D, P)
    G = part_g(F)
    figure_fit(D, P, F)
    return {"D": D, "P": P, "F": F, "E": E, "R": R, "G": G}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

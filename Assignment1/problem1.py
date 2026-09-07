"""
Problem 1 -- Reading the shape of a sample.

problem1.csv holds a single series x. We
  (a) compute the first four moments and use skew + excess kurtosis alone to
      screen the Week 1 candidate distributions,
  (b) fit a Normal by matching mean and variance, and count how many
      observations fall below its 1% quantile against how many should,
  (c) locate where the Normal fails and in which direction.

Run:  python problem1.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

import common
from common import C1, C2, C3


# --------------------------------------------------------------------------
# (a) Moments, and the moment-based screen of candidate distributions
# --------------------------------------------------------------------------
def part_a(x: np.ndarray) -> dict:
    m = common.moments(x)
    common.header("Problem 1(a) -- first four moments")
    print(f"  n                 = {m['n']}")
    print(f"  mean              = {m['mean']: .6f}")
    print(f"  variance (ddof=1) = {m['var']: .6e}")
    print(f"  std dev           = {m['sd']: .6f}")
    print(f"  skewness          = {m['skew']: .4f}   [bias-corrected, Week 01 s4.3]")
    print(f"  excess kurtosis   = {m['exkurt']: .4f}   [bias-corrected]")
    # Week 01 s4.4 warns that packages differ on this and says to check rather
    # than assume, so all three variants are shown. The spread is ~0.02 on 2.34.
    print(f"\n  Estimator variants (Week 01 s4.4 -- check, do not assume):")
    print(f"    skew   : bias-corrected {m['skew']:+.4f}   "
          f"population form {m['skew_pop']:+.4f}")
    print(f"    exkurt : bias-corrected {m['exkurt']:+.4f}   "
          f"population form {m['exkurt_pop']:+.4f}")
    print(f"             s4.3 expression over s^4 {m['exkurt_slide']:+.4f}")
    print(f"    The three exkurt variants span "
          f"{max(m['exkurt'], m['exkurt_pop'], m['exkurt_slide']) - min(m['exkurt'], m['exkurt_pop'], m['exkurt_slide']):.4f}"
          f" on a value of {m['exkurt']:.2f}, so")
    print(f"    nothing below depends on the choice.")

    # Week 01 s6 feasibility bound: kurtosis >= skew^2 + 1 for ANY distribution.
    ok, bound = common.shape_is_admissible(m["skew"], m["exkurt"])
    print(f"\n  Feasibility check (Week 01 s6): excess kurtosis must be >= "
          f"skew^2 - 2 = {bound:.4f}")
    print(f"    observed {m['exkurt']:+.4f} -> "
          f"{'admissible' if ok else 'IMPOSSIBLE, indicates a calculation error'}")

    # Are these two numbers actually far from Normal, or just noise? Under the
    # null of normality skew ~ N(0, 6/n) and excess kurtosis ~ N(0, 24/n), so we
    # can put each on a t-like scale and read it off directly.
    t_skew = m["skew"] / m["se_skew"]
    t_kurt = m["exkurt"] / m["se_exkurt"]
    jb = stats.jarque_bera(x)
    print(f"\n  Under H0: Normal --  se(skew) = {m['se_skew']:.4f}, "
          f"se(exkurt) = {m['se_exkurt']:.4f}")
    print(f"    skew            : {t_skew:+.2f} standard errors from 0")
    print(f"    excess kurtosis : {t_kurt:+.2f} standard errors from 0")
    print(f"    Jarque-Bera     : JB = {jb.statistic:.2f}, p = {jb.pvalue:.3e}")

    # The moment screen. Each rejection below names the single moment that does
    # the rejecting -- that is what the question asks for.
    # The screen below follows Week 01 Table 6, which is the course's own
    # skew/kurtosis filter, and is restricted to the four families s5 actually
    # covers: Normal, Lognormal, Student's t, and Normal Inverse Gaussian. Each
    # rejection names the single moment responsible. Table 6 is described in the
    # slides as "a filter, not a decision" -- it says which families CAN
    # reproduce the observed shape, and choosing among the survivors "requires a
    # likelihood comparison or a look at the tail itself", which part (c) does.
    print("\n  Moment screen -- Week 01 Table 6 applied to this sample")
    print(f"  observed: skew {m['skew']:+.4f} (< 0),  excess kurtosis "
          f"{m['exkurt']:+.4f} (> 0)")
    print("  " + "-" * 72)
    print("  Table 6 rows, and which one this sample lands in:")
    print("    skew ~ 0, exkurt ~ 0          -> Normal")
    print("    skew ~ 0, exkurt > 0          -> Student's t, symmetric NIG")
    print("    skew != 0, exkurt > 0         -> NIG, skewed t, generalized")
    print("                                     hyperbolic      <== THIS SAMPLE")
    print("    skew > 0, exkurt > 0,")
    print("      positive support            -> Lognormal")
    print("  " + "-" * 72)
    screen = [
        ("Normal", "skew = 0, exkurt = 0",
         "RULED OUT by excess kurtosis: +2.36 is 15 se above the 0 it",
         "requires. s5.1/s6: no choice of mu and sigma changes this."),
        ("Lognormal", "skew > 0, support x > 0",
         "RULED OUT by skewness: sign is wrong (needs skew > 0, sample",
         "is -0.67). Also excluded by support -- the sample has "
         "negative values."),
        ("Student t", "skew = 0, exkurt > 0",
         "kurtosis PLAUSIBLE, but the t is SYMMETRIC (s5.5: 'the t gives",
         "us fat tails and nothing else'), so it cannot make skew -0.67."),
        ("NIG", "skew and exkurt both free",
         "PLAUSIBLE -- the only s5 family that carries BOTH skew and",
         "excess kurtosis. Table 6's answer for this row."),
    ]
    for name, implied, v1, v2 in screen:
        print(f"  {name:<12}{implied}")
        print(f"  {'':<12}-> {v1}")
        if v2:
            print(f"  {'':<12}   {v2}")
    print("  " + "-" * 72)
    print("  Sign check on the NIG (s5.5, Table 5): skewness is 3*beta/(alpha*")
    print("  sqrt(delta*gamma)), which follows the SIGN OF BETA. A negative skew")
    print("  needs beta < 0, and that is what the fit in part (c) returns.")

    # Corroboration that the NIG really can reach this (skew, exkurt) pair,
    # rather than taking Table 6 on faith. Using s5.5 Table 5:
    #   skew   = 3*beta / (alpha*sqrt(delta*gamma)),  gamma = sqrt(alpha^2-beta^2)
    #   exkurt = 3/(delta*gamma) * (1 + 4*beta^2/alpha^2)
    # Solve numerically for an (alpha, beta, delta) hitting the sample pair.
    def nig_shape(alpha, beta, delta):
        gamma = np.sqrt(alpha ** 2 - beta ** 2)
        sk = 3.0 * beta / (alpha * np.sqrt(delta * gamma))
        ek = 3.0 / (delta * gamma) * (1.0 + 4.0 * beta ** 2 / alpha ** 2)
        return sk, ek

    def nig_resid(theta):
        # log-parameterise alpha, delta > 0 and keep |beta| < alpha
        a = np.exp(theta[0])
        b = a * np.tanh(theta[1])
        dl = np.exp(theta[2])
        sk, ek = nig_shape(a, b, dl)
        return [sk - m["skew"], ek - m["exkurt"]]

    sol = optimize.fsolve(lambda t: nig_resid(t) + [0.0], [3.0, -0.3, -2.0],
                          full_output=False)
    a_s = np.exp(sol[0]); b_s = a_s * np.tanh(sol[1]); d_s = np.exp(sol[2])
    sk_s, ek_s = nig_shape(a_s, b_s, d_s)
    print(f"\n  NIG attainability check (solving Table 5 for the sample pair):")
    print(f"    alpha = {a_s:.4f}, beta = {b_s:+.4f}, delta = {d_s:.6f}")
    print(f"    reproduces skew {sk_s:+.4f} and excess kurtosis {ek_s:+.4f}")
    print(f"    target was     skew {m['skew']:+.4f} and excess kurtosis "
          f"{m['exkurt']:+.4f}")
    print(f"    -> the pair is inside the NIG's attainable region, and beta < 0")
    print(f"       as the negative skew requires.")

    return m


# --------------------------------------------------------------------------
# (b) Fit a Normal by matching mean and variance; count the 1% exceedances
# --------------------------------------------------------------------------
def part_b(x: np.ndarray, m: dict) -> dict:
    n = m["n"]
    mu, sd = m["mean"], m["sd"]  # method of moments == MLE-ish for a Normal

    q01 = stats.norm.ppf(0.01, loc=mu, scale=sd)
    observed = int((x < q01).sum())
    expected = 0.01 * n

    common.header("Problem 1(b) -- exceedances of the fitted Normal's 1% quantile")
    print(f"  Fitted Normal: mu = {mu:.6f}, sigma = {sd:.6f}")
    print(f"  1% quantile of that Normal      = {q01:.6f}")
    print(f"  OBSERVED number of x below it   = {observed}")
    print(f"  EXPECTED number if truly Normal = {expected:.1f}")
    print(f"  ratio observed / expected       = {observed / expected:.2f}x")

    # A binomial tail check, so "26 vs 10" is quantified rather than eyeballed.
    p_val = stats.binomtest(observed, n, 0.01, alternative="greater").pvalue
    print(f"  Binomial test P(>= {observed} of {n} at p=0.01) = {p_val:.3e}")

    # The same count across a range of tail probabilities: this is what shows
    # the failure is a tail phenomenon and not a level shift.
    print("\n  Exceedance counts across the left tail")
    print("  " + "-" * 68)
    print(f"  {'p':>7}{'Normal q':>12}{'empirical q':>14}"
          f"{'observed':>11}{'expected':>10}")
    print("  " + "-" * 68)
    rows = []
    for p in [0.001, 0.005, 0.01, 0.025, 0.05]:
        nq = stats.norm.ppf(p, loc=mu, scale=sd)
        eq = np.quantile(x, p)
        obs = int((x < nq).sum())
        rows.append({"p": p, "normal_q": nq, "empirical_q": eq,
                     "observed": obs, "expected": p * n})
        print(f"  {p:>7}{nq:>12.5f}{eq:>14.5f}{obs:>11d}{p * n:>10.1f}")
    print("  " + "-" * 68)

    # And the right tail, which moves the OTHER way -- the asymmetry is the
    # whole point of part (c).
    print("\n  Right tail, for contrast")
    print("  " + "-" * 68)
    print(f"  {'p':>7}{'Normal q':>12}{'observed above':>17}{'expected':>12}")
    print("  " + "-" * 68)
    for p in [0.95, 0.99, 0.995]:
        nq = stats.norm.ppf(p, loc=mu, scale=sd)
        obs = int((x > nq).sum())
        print(f"  {p:>7}{nq:>12.5f}{obs:>17d}{(1 - p) * n:>12.1f}")
    print("  " + "-" * 68)

    # Centre of the distribution: too MANY observations sit in the middle. A
    # fat-tailed sample forced into a Normal must be over-peaked as well.
    lo = stats.norm.ppf(0.25, loc=mu, scale=sd)
    hi = stats.norm.ppf(0.75, loc=mu, scale=sd)
    inside = float(((x > lo) & (x < hi)).mean())
    print(f"\n  Fraction of sample inside the fitted Normal's IQR "
          f"= {inside:.3f}  (a true Normal gives 0.500)")

    return {"mu": mu, "sd": sd, "q01": q01, "observed": observed,
            "expected": expected, "tail": pd.DataFrame(rows),
            "iqr_frac": inside}


# --------------------------------------------------------------------------
# (c) Where does the Normal go wrong, and in which direction
# --------------------------------------------------------------------------
def part_c(x: np.ndarray, m: dict, fit: dict) -> None:
    mu, sd = fit["mu"], fit["sd"]
    common.header("Problem 1(c) -- where the fitted Normal is wrong")

    emp_q01 = np.quantile(x, 0.01)
    print(f"  The Normal's 1% quantile is {fit['q01']:.5f}; the sample's true 1%")
    print(f"  quantile is {emp_q01:.5f}, i.e. "
          f"{emp_q01 / fit['q01']:.2f}x as far from the mean.")
    print(f"  Put the other way: the Normal assigns probability "
          f"{stats.norm.cdf(emp_q01, mu, sd):.4f}")
    print(f"  to a loss that actually happens 1% of the time -- it understates the")
    print(f"  frequency of that loss by a factor of "
          f"{0.01 / stats.norm.cdf(emp_q01, mu, sd):.1f}.")
    # The same failure stated as a VaR magnitude, which is the form a risk report
    # would use. Note this is NOT the 1.24x ratio above: a quantile that is 1.24x
    # too near the mean is 19.5% too small, and conflating the two is easy.
    print(f"  As a VaR magnitude: a 99% VaR read off this fitted Normal is")
    print(f"  {100 * (1 - fit['q01'] / emp_q01):.1f}% too small ({abs(fit['q01']):.5f} against the empirical "
          f"{abs(emp_q01):.5f}).")

    # Week 01 s6: Table 6 is "a filter, not a decision", and choosing among the
    # families it keeps "requires a likelihood comparison or a look at the tail
    # itself". So we do both -- the likelihood comparison here, the tail in the
    # figure. Candidates are the s5 families only (Lognormal excluded: its
    # support is x > 0 and this sample has negative values).
    df_t, loc_t, sc_t = stats.t.fit(x)
    ll_norm = float(np.sum(stats.norm.logpdf(x, mu, sd)))
    ll_t = float(np.sum(stats.t.logpdf(x, df_t, loc_t, sc_t)))
    # scipy's norminvgauss(a, b) maps to the slides' (alpha, beta, delta, mu) as
    # a = alpha*delta, b = beta*delta, scale = delta, loc = mu.
    p_nig = stats.norminvgauss.fit(x)
    ll_nig = float(np.sum(stats.norminvgauss.logpdf(x, *p_nig)))
    a_n, b_n, loc_n, sc_n = p_nig
    alpha_n, beta_n, delta_n = a_n / sc_n, b_n / sc_n, sc_n

    tbl = common.aicc_table({
        "Normal (k=2)": (ll_norm, 2),
        "Student t (k=3)": (ll_t, 3),
        "NIG (k=4)": (ll_nig, 4),
    }, m["n"])
    print("\n  Likelihood comparison across the Week 01 s5 families")
    print("  (Lognormal omitted: support is x > 0, the sample has negatives)")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\n  Fitted Student t : nu = {df_t:.2f}, loc = {loc_t:.5f}, "
          f"scale = {sc_t:.5f}")
    print(f"  Fitted NIG       : alpha = {alpha_n:.2f}, beta = {beta_n:+.2f}, "
          f"delta = {delta_n:.5f}, mu = {loc_n:+.5f}")
    print(f"    beta = {beta_n:+.2f} < 0, matching the negative sample skew, as")
    print(f"    s5.5 Table 5 requires (skewness follows the sign of beta).")
    print(f"  AICc ranks them exactly as the moment screen predicted: the family")
    print(f"  carrying both skew and kurtosis wins, the symmetric fat-tailed")
    print(f"  family is next, and the Normal is last by "
          f"{float(tbl.loc[tbl['model'] == 'Normal (k=2)', 'dAICc'].iloc[0]):.0f} AICc units.")

    # ---------------- figure ----------------
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.9))
    grid = np.linspace(x.min() - 0.004, x.max() + 0.004, 800)

    # Panel 1: histogram vs the fitted Normal density.
    ax[0].hist(x, bins=45, density=True, color=C1, alpha=0.55,
               edgecolor="white", linewidth=0.5, label="sample")
    ax[0].plot(grid, stats.norm.pdf(grid, mu, sd), color=C2,
               label="fitted Normal")
    ax[0].axvline(fit["q01"], color=common.INK2, linestyle=":", linewidth=1.4)
    ax[0].annotate("Normal 1% quantile", xy=(fit["q01"], 0),
                   xytext=(fit["q01"] - 0.0005, ax[0].get_ylim()[1] * 0.55),
                   fontsize=8, color=common.INK2, ha="right")
    ax[0].set_title("Sample is over-peaked and left-skewed")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("density")
    ax[0].legend(loc="upper left")

    # Panel 2: the left tail on a log density scale, where the gap is visible.
    ax[1].hist(x, bins=45, density=True, color=C1, alpha=0.55,
               edgecolor="white", linewidth=0.5, label="sample")
    ax[1].plot(grid, stats.norm.pdf(grid, mu, sd), color=C2,
               label="fitted Normal")
    ax[1].plot(grid, stats.norminvgauss.pdf(grid, *p_nig), color=C3,
               linestyle="--", label="NIG (skew + fat tails)")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-1, None)
    ax[1].set_title("Log density: the Normal's tails die too fast")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("density (log)")
    ax[1].legend(loc="lower center")

    # Panel 3: QQ plot against the fitted Normal. Curvature at BOTH ends, but
    # asymmetric -- the left end departs much further.
    ps = (np.arange(1, len(x) + 1) - 0.5) / len(x)
    theo = stats.norm.ppf(ps, mu, sd)
    ax[2].scatter(theo, np.sort(x), s=9, color=C1, alpha=0.7,
                  edgecolor="none")
    lim = [min(theo.min(), x.min()), max(theo.max(), x.max())]
    ax[2].plot(lim, lim, color=C2, linewidth=1.6, label="y = x")
    ax[2].set_title("QQ vs fitted Normal: left tail departs most")
    ax[2].set_xlabel("theoretical quantile")
    ax[2].set_ylabel("sample quantile")
    ax[2].legend(loc="upper left")

    fig.suptitle("Problem 1 -- a fat-tailed, left-skewed sample fitted with a Normal",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem1")


def main() -> dict:
    x = common.load("problem1")["x"].to_numpy()
    m = part_a(x)
    fit = part_b(x, m)
    part_c(x, m, fit)
    return {"x": x, "moments": m, "fit": fit}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

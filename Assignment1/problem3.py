"""
Problem 3 -- Pearson against Spearman.

problem3.csv holds x1..x4. We
  (a) plot every pair and predict which pair the two coefficients disagree on,
  (b) compute both correlation matrices and report the largest gap,
  (c) explain the mechanism and say which coefficient is honest about that pair.

Run:  python problem3.py
"""

from __future__ import annotations

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common
from common import C1, C2, C3


# --------------------------------------------------------------------------
# (a) Prediction: the pair plot, plus the per-series moments that give it away
# --------------------------------------------------------------------------
def part_a(d: pd.DataFrame) -> None:
    common.header("Problem 3(a) -- prediction stage: shape of each series")
    print(f"  {'series':<8}{'sd':>9}{'skew':>9}{'exkurt':>10}"
          f"{'min':>9}{'max':>9}")
    print("  " + "-" * 55)
    for c in d.columns:
        m = common.moments(d[c].to_numpy())
        print(f"  {c:<8}{m['sd']:>9.3f}{m['skew']:>9.3f}{m['exkurt']:>10.3f}"
              f"{d[c].min():>9.3f}{d[c].max():>9.3f}")
    print("  " + "-" * 55)
    # Computed, not hardcoded: this line previously carried a stale +21.8 from an
    # earlier estimator convention while the table above printed 21.99.
    _m2 = common.moments(d["x2"].to_numpy())
    _sdr = _m2["sd"] / np.mean([common.moments(d[c].to_numpy())["sd"]
                                for c in ("x1", "x3", "x4")])
    print(f"  x2 stands out: excess kurtosis ~ {_m2['exkurt']:+.1f} and an sd "
          f"roughly {_sdr:.0f}x the")
    print("  others. Any pair involving x2 is a candidate for disagreement,")
    print("  because Pearson is a moment statistic and x2's moments are")
    print("  dominated by a few extreme values.")

    # ---------------- figure: the full pair plot ----------------
    common.use_style()
    cols = list(d.columns)
    fig, axes = plt.subplots(4, 4, figsize=(10.5, 10.0))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                ax.hist(d[ci], bins=40, color=C1, alpha=0.6,
                        edgecolor="white", linewidth=0.4)
                ax.set_yticks([])
            else:
                ax.scatter(d[cj], d[ci], s=6, color=C1, alpha=0.45,
                           edgecolor="none")
                r_p = stats.pearsonr(d[cj], d[ci]).statistic
                r_s = stats.spearmanr(d[cj], d[ci]).statistic
                # Annotating both coefficients on every panel is what makes the
                # prediction checkable by eye rather than by faith.
                ax.text(0.04, 0.95, f"P {r_p:+.2f}\nS {r_s:+.2f}",
                        transform=ax.transAxes, fontsize=7.5, va="top",
                        color=common.INK,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor=common.GRID, linewidth=0.6))
            if i == 3:
                ax.set_xlabel(cj)
            if j == 0:
                ax.set_ylabel(ci)
            ax.tick_params(labelsize=7)
    fig.suptitle("Problem 3 -- all pairs, with Pearson (P) and Spearman (S)",
                 fontsize=11.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    common.finish(fig, "problem3_pairs")


# --------------------------------------------------------------------------
# (b) Both matrices, and the ranked gap table
# --------------------------------------------------------------------------
def part_b(d: pd.DataFrame) -> pd.DataFrame:
    P = d.corr(method="pearson")
    S = d.corr(method="spearman")

    common.header("Problem 3(b) -- the two correlation matrices")
    print("  PEARSON (linear association, uses the values)")
    print(P.round(4).to_string())
    print("\n  SPEARMAN (monotone association, uses only the ranks)")
    print(S.round(4).to_string())

    rows = []
    for a, b in itertools.combinations(d.columns, 2):
        rp, rs = P.loc[a, b], S.loc[a, b]
        rows.append({"pair": f"{a}-{b}", "pearson": rp, "spearman": rs,
                     "gap (S - P)": rs - rp, "abs gap": abs(rs - rp)})
    gaps = pd.DataFrame(rows).sort_values("abs gap", ascending=False)\
        .reset_index(drop=True)
    print("\n  All six pairs, ranked by |Spearman - Pearson|")
    print(gaps.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    top = gaps.iloc[0]
    print(f"\n  LARGEST GAP: {top['pair']} -- Pearson {top['pearson']:+.4f} vs")
    print(f"  Spearman {top['spearman']:+.4f}, a gap of {top['gap (S - P)']:+.4f}.")
    return gaps


# --------------------------------------------------------------------------
# (c) Mechanism
# --------------------------------------------------------------------------
def part_c(d: pd.DataFrame, gaps: pd.DataFrame) -> None:
    x1 = d["x1"].to_numpy()
    x2 = d["x2"].to_numpy()

    common.header("Problem 3(c) -- the mechanism behind the x1-x2 gap")

    # Hypothesis: x2 is a monotone but strongly CONVEX/CONCAVE function of x1,
    # i.e. a power transform. Test it directly by regressing x2 on x1**3.
    coef = np.polyfit(x1 ** 3, x2, 1)
    resid = x2 - np.polyval(coef, x1 ** 3)
    r2 = 1 - resid.var() / x2.var()
    print(f"  Test of a cubic link: regress x2 on x1^3")
    print(f"    x2 = {coef[1]:+.4f} + {coef[0]:.4f} * x1^3")
    print(f"    R^2 = {r2:.6f},  residual sd = {resid.std(ddof=1):.4f}")
    print(f"  So x2 is very nearly x1 cubed plus a little noise.")
    print()
    print(f"  This is Week 02 s1.4 Figure 2 almost exactly. The slide plots")
    print(f"  y = x^3 and reports 'Pearson is 0.74 but Spearman is 1.00 because")
    print(f"  the relationship is monotonic'. We get {stats.pearsonr(x1, x2).statistic:.2f} and "
          f"{stats.spearmanr(x1, x2).statistic:.2f} on the")
    print(f"  same functional form, the small differences coming from this")
    print(f"  sample's noise and its particular spread of x1 values.")
    print()
    print(f"  That single fact explains the whole gap:")
    print(f"    * the map x -> x^3 is strictly increasing, so RANKS are almost")
    print(f"      perfectly preserved -> Spearman "
          f"{stats.spearmanr(x1, x2).statistic:.4f}, near 1;")
    print(f"    * but it is badly non-linear, so a straight line through the")
    print(f"      cloud misses -> Pearson only "
          f"{stats.pearsonr(x1, x2).statistic:.4f}.")

    # Show the leverage explicitly: a handful of |z|>3 points on x2 hold the
    # Pearson coefficient down, which is the "moment statistic" story in numbers.
    z2 = (x2 - x2.mean()) / x2.std(ddof=1)
    keep = np.abs(z2) <= 3
    print(f"\n  Leverage check -- x2 has {int((np.abs(z2) > 3).sum())} points beyond")
    print(f"  3 sd (a Normal would give ~1 in 500). Dropping just those:")
    print(f"    Pearson  {stats.pearsonr(x1, x2).statistic:.4f} -> "
          f"{stats.pearsonr(x1[keep], x2[keep]).statistic:.4f}")
    print(f"    Spearman {stats.spearmanr(x1, x2).statistic:.4f} -> "
          f"{stats.spearmanr(x1[keep], x2[keep]).statistic:.4f}  (barely moves)")
    print(f"  Pearson is a function of the sample covariance, so it is moved by")
    print(f"  a few extreme values; Spearman only sees their ORDER.")

    # Contrast: the linear pair x1-x3, where the two agree, and the independent
    # pairs involving x4.
    # Why isn't Spearman exactly 1.0, given a near-deterministic monotone link?
    # Because x^3 is FLAT near zero: its derivative 3*x1^2 is 0.19 at x1 = 0.25
    # but 12.0 at x1 = 2. Where the local slope falls below the noise sd (0.05),
    # the noise, not x1, decides the ordering -- so ranks get shuffled in the
    # middle while the tails stay in perfect order.
    r1, r2 = stats.rankdata(x1), stats.rankdata(x2)
    rank_gap = np.abs(r1 - r2)
    inner = np.abs(x1) < 0.5
    print(f"\n  Why Spearman is {stats.spearmanr(x1, x2).statistic:.4f} and not exactly 1.0000:")
    print(f"    x^3 is FLAT near the origin -- slope 3*x1^2 is "
          f"{3 * 0.25 ** 2:.2f} at x1 = 0.25")
    print(f"    but {3 * 2.0 ** 2:.1f} at x1 = 2.0, against a noise sd of "
          f"{resid.std(ddof=1):.3f}. Where the local")
    print(f"    slope is smaller than the noise, the NOISE decides the ordering:")
    print(f"      mean |rank difference|, |x1| <  0.5 : "
          f"{rank_gap[inner].mean():6.2f}  (n = {int(inner.sum())})")
    print(f"      mean |rank difference|, |x1| >= 0.5 : "
          f"{rank_gap[~inner].mean():6.2f}  (n = {int((~inner).sum())})")
    print(f"    Restricting to |x1| >= 0.5, Spearman rises to "
          f"{stats.spearmanr(x1[~inner], x2[~inner]).statistic:.4f}.")
    print(f"    So the small rank imperfection is a local-flatness artefact, not")
    print(f"    evidence against the monotone story.")

    print(f"\n  For contrast:")
    print(f"    x1-x3 (linear, Gaussian-looking): Pearson "
          f"{stats.pearsonr(x1, d['x3']).statistic:+.4f}, Spearman "
          f"{stats.spearmanr(x1, d['x3']).statistic:+.4f} -- agree.")
    print(f"    Spearman actually sits slightly BELOW Pearson here, the usual")
    print(f"    ~2-3% efficiency loss from throwing away the values when the")
    print(f"    relationship really is linear and Normal.")
    print(f"    Pairs with x4: both coefficients ~0 -- x4 is independent of the")
    print(f"    rest, and neither statistic can manufacture association.")

    print(f"\n  WHICH ONE IS HONEST for x1-x2?")
    print(f"  Spearman ({stats.spearmanr(x1, x2).statistic:.4f}) is the honest")
    print(f"  description of the DEPENDENCE: x2 is an almost deterministic,")
    print(f"  strictly increasing function of x1, so knowing x1 tells you almost")
    print(f"  exactly where x2 sits. Pearson is not wrong, it is answering a")
    print(f"  different question -- 'how much of x2's variance does a STRAIGHT")
    print(f"  LINE in x1 explain?' -- and 0.80 is the correct answer to that")
    print(f"  question. It is the wrong question to ask of a cubic relationship.")
    print(f"  Week 02 s1.3 makes the same point with its own warning case: for")
    print(f"  y = x^2 the Pearson coefficient is exactly ZERO though y is a")
    print(f"  deterministic function of x. As the slide puts it, 'a correlation of")
    print(f"  zero means no linear relationship. It does not mean no relationship,")
    print(f"  and it does not mean independence.' Our x1-x2 pair is the milder")
    print(f"  version: monotone, so Pearson is large but still understated.")
    print(f"  s1.4 also notes the converse DOES hold for the multivariate normal,")
    print(f"  which is why zero correlation is safe to read as independence there")
    print(f"  and nowhere else.")
    print(f"  Practical note: for risk work this is exactly why copula methods")
    print(f"  are built on rank correlation -- the dependence survives any")
    print(f"  monotone re-scaling of the marginals, and Pearson does not. Week 02")
    print(f"  s1.4 flags this too: 'Spearman returns in Week 05 ... rank")
    print(f"  correlation is the natural way to describe dependence once the")
    print(f"  marginals have been stripped out.'")

    # ---------------- figure ----------------
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.9))

    # Panel 1: the raw pair with a straight line laid over it -- shows the miss.
    ax[0].scatter(x1, x2, s=11, color=C1, alpha=0.6, edgecolor="none")
    xs = np.linspace(x1.min(), x1.max(), 200)
    lin = np.polyfit(x1, x2, 1)
    ax[0].plot(xs, np.polyval(lin, xs), color=C2,
               label="OLS line (what Pearson measures)")
    ax[0].plot(xs, np.polyval(coef, xs ** 3), color=C3, linestyle="--",
               label="cubic fit (the real link)")
    ax[0].set_title(f"x1 vs x2:  Pearson {stats.pearsonr(x1, x2).statistic:.3f}, "
                    f"Spearman {stats.spearmanr(x1, x2).statistic:.3f}")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].legend(loc="upper left")

    # Panel 2: the same pair in RANK space -- the near-perfect line Spearman sees.
    rk1, rk2 = stats.rankdata(x1), stats.rankdata(x2)
    ax[1].scatter(rk1, rk2, s=11, color=C1, alpha=0.6, edgecolor="none")
    # Call out the middle, where the cubic is flat and the noise reorders points.
    mid = np.abs(x1) < 0.5
    ax[1].scatter(rk1[mid], rk2[mid], s=13, color=C2, alpha=0.7,
                  edgecolor="none",
                  label="|x1| < 0.5, where x^3 is flat")
    ax[1].set_title("Same pair in rank space: nearly a straight line")
    ax[1].set_xlabel("rank of x1")
    ax[1].set_ylabel("rank of x2")
    ax[1].legend(loc="upper left", fontsize=8)

    # Panel 3: the agreeing pair, for contrast.
    x3 = d["x3"].to_numpy()
    ax[2].scatter(x1, x3, s=11, color=C1, alpha=0.6, edgecolor="none")
    ax[2].plot(xs, np.polyval(np.polyfit(x1, x3, 1), xs), color=C2,
               label="OLS line")
    ax[2].set_title(f"x1 vs x3 (linear): P "
                    f"{stats.pearsonr(x1, x3).statistic:.3f}, S "
                    f"{stats.spearmanr(x1, x3).statistic:.3f}")
    ax[2].set_xlabel("x1")
    ax[2].set_ylabel("x3")
    ax[2].legend(loc="upper left")

    fig.suptitle("Problem 3 -- a monotone non-linearity splits Pearson from Spearman",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem3_mechanism")


def main() -> dict:
    d = common.load("problem3")
    part_a(d)
    gaps = part_b(d)
    part_c(d, gaps)
    return {"data": d, "gaps": gaps}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

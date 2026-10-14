"""
Problem 3 -- When diversification raises VaR.

problem3.csv holds 10,000 simulated one-year return scenarios for bonds A and B,
each bought at 90 and paying 100 unless it defaults. $2,000,000 to invest: all
in A, or $1,000,000 in each. We

  predict  : P&L distributions of both choices; the four moments of each bond;
             default counts (loss > 20%) for A, B, and "at least one"; a
             prediction for VaR and ES from the counts alone;
  fit      : historical VaR and ES at 5% for four positions, a Normal VaR for
             each from the sample mean and sd of its P&L, and the historical
             numbers again at 1%;
  reconcile: subadditivity of VaR and of ES at 5%, what the Normal VaR gets
             right and why it is still wrong, and why 1% flips the VaR result.

Run:  python problem3.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import common
from common import C1, C2, C3

UNIT = 1_000_000.0
DEFAULT_CUT = -0.20       # "loses more than 20%" is the problem's definition of a default


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def prepare() -> dict:
    d = common.load("problem3")
    ra, rb = d["A"].to_numpy(dtype=float), d["B"].to_numpy(dtype=float)
    # P&L in dollars for each position. Arithmetic returns, so P&L = holding * r
    # and the portfolio P&L is the sum (Week 04 s4.1.2).
    pnl = {
        "$1M in A": UNIT * ra,
        "$1M in B": UNIT * rb,
        "$2M in A": 2 * UNIT * ra,
        "$1M each": UNIT * ra + UNIT * rb,
    }
    return {"ra": ra, "rb": rb, "pnl": pnl, "n": ra.size}


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
def part_predict(D: dict) -> dict:
    ra, rb, n = D["ra"], D["rb"], D["n"]
    common.header("Problem 3 -- Predict: moments and default counts")
    print("  (a) Four moments of each bond's one-year RETURN:")
    print(f"  {'bond':<10}{'mean':>13}{'sd':>12}{'skew':>9}{'exkurt':>9}")
    ma, mb = common.moments(ra), common.moments(rb)
    common.print_moments("A", ma, pct=True)
    common.print_moments("B", mb, pct=True)
    print(f"\n      Return if no default = 100/90 - 1 = {100 / 90 - 1:+.4%} (the data scatter "
          f"around it:")
    nd = ra[ra > DEFAULT_CUT]
    print(f"      non-default A returns run {nd.min():+.3f} to {nd.max():+.3f}, "
          f"sd {nd.std(ddof=1):.4f}).")
    dd = ra[ra <= DEFAULT_CUT]
    print(f"      Default returns run {dd.min():+.3f} to {dd.max():+.3f}: recovery of "
          f"{90 * (1 + dd.min()):.0f}-{90 * (1 + dd.max()):.0f} on the 90 paid.")
    print(f"      Skew of {ma['skew']:.1f} and excess kurtosis of {ma['exkurt']:.0f} from a "
          f"two-lump distribution:")
    print(f"      a spike near +11% and a lump near -80%, with nothing in between. A Normal")
    print(f"      with this mean and sd puts its 5% quantile at mu - 1.645 sd = "
          f"{ma['mean'] - 1.645 * ma['sd']:+.3f},")
    print(f"      a {100 * abs(ma['mean'] - 1.645 * ma['sd']):.0f}% loss. Returns anywhere "
          f"between -30% and -5% occur in {int(((ra > -0.3) & (ra < -0.05)).sum())} of")
    print(f"      {n} scenarios: the Normal VaR describes a region of the distribution")
    print(f"      that is essentially empty.")

    # (b) counts
    da, db = ra <= DEFAULT_CUT, rb <= DEFAULT_CUT
    n_a, n_b = int(da.sum()), int(db.sum())
    n_either, n_both = int((da | db).sum()), int((da & db).sum())
    print(f"\n  (b) Scenarios losing more than 20%:")
    print(f"      A alone            : {n_a:>5}   ({100 * n_a / n:.2f}%)")
    print(f"      B alone            : {n_b:>5}   ({100 * n_b / n:.2f}%)")
    print(f"      at least one of two: {n_either:>5}   ({100 * n_either / n:.2f}%)")
    print(f"      both               : {n_both:>5}   ({100 * n_both / n:.2f}%; independence "
          f"predicts {n * (n_a / n) * (n_b / n):.1f})")
    print(f"      correlation of the two return series: {np.corrcoef(ra, rb)[0, 1]:+.4f}")
    # Is the 20% cut clean? Count scenarios in the gap between the default lump
    # and the coupon spike, where the cut could move a scenario across.
    for lab, x in (("A", ra), ("B", rb)):
        gap = x[(x > DEFAULT_CUT) & (x < 0.05)]
        if gap.size:
            print(f"      note: {lab} has {gap.size} scenario(s) between -20% and +5% "
                  f"({', '.join(f'{v:+.3f}' for v in gap)}): a high-recovery")
            print(f"      default the 20% cut leaves out. It changes no VaR or ES below.")
    print(f"\n      5% of {n} scenarios = {int(0.05 * n)} worst outcomes; 1% = {int(0.01 * n)}.")
    print(f"      Concentrated in A: {n_a} defaults < {int(0.05 * n)}, so the 500th-worst "
          f"scenario is a")
    print(f"      NON-default. Diversified: {n_either} scenarios hold a default > "
          f"{int(0.05 * n)}, so the")
    print(f"      500th-worst scenario IS a default (of one bond).")

    # Count-based prediction of the numbers themselves, using only the counts,
    # the AVERAGE loss on a default and the AVERAGE coupon-year gain.
    k5 = int(0.05 * n)
    loss_1 = -(UNIT * ra[da]).mean()                     # mean $ loss on a $1M default
    gain_1 = (UNIT * rb[~db]).mean()                     # mean $ gain on $1M, no default
    es_conc = (n_a * 2 * loss_1 - (k5 - n_a) * 2 * gain_1) / k5
    es_div = ((k5 - n_both) * (loss_1 - gain_1) + n_both * 2 * loss_1) / k5
    print(f"\n      Prediction from counts. ES is the average of the {k5} worst:")
    print(f"        concentrated: ~{n_a} defaults losing ~{common.money(2 * loss_1)} each, "
          f"{k5 - n_a} coupon years")
    print(f"          gaining ~{common.money(2 * gain_1)} each -> ES roughly "
          f"{common.money(es_conc)}")
    print(f"        diversified : ~{k5 - n_both} single defaults losing ~"
          f"{common.money(loss_1 - gain_1)} net of the")
    print(f"          other coupon, and {n_both} double defaults losing ~"
          f"{common.money(2 * loss_1)} -> ES roughly {common.money(es_div)}")
    print(f"      (both biased low: the tail holds the LOW-recovery defaults, not average")
    print(f"      ones). So: diversified VaR LARGER -- a ~{common.money(2 * UNIT * np.sort(ra)[k5 - 1])}"
          f" profit becomes a")
    print(f"      ~{common.money(loss_1 - gain_1)} loss -- and diversified ES SMALLER, by "
          f"roughly half.")
    return {"ma": ma, "mb": mb, "n_a": n_a, "n_b": n_b, "n_either": n_either,
            "n_both": n_both, "da": da, "db": db}


# --------------------------------------------------------------------------
# Fit
# --------------------------------------------------------------------------
def risk_table(D: dict, alpha: float) -> dict:
    out = {}
    print(f"  {'position':<12}{'VaR':>14}{'ES':>14}{'VaR/ES ratio':>14}")
    print("  " + "-" * 54)
    for name, x in D["pnl"].items():
        v, e = common.var_es(x, alpha)
        out[name] = (v, e)
        print(f"  {name:<12}{common.money(v):>14}{common.money(e):>14}{v / e:>14.2f}")
    print("  " + "-" * 54)
    return out


def part_c(D: dict) -> dict:
    common.header("Problem 3(c) -- historical VaR and ES at alpha = 5%")
    out = risk_table(D, 0.05)
    print("  A negative VaR is a PROFIT at the 5% quantile: the 500th-worst scenario")
    print("  for $2M in A is still a coupon-paying year.")
    return out


def part_d(D: dict) -> dict:
    common.header("Problem 3(d) -- Normal VaR from the sample mean and sd of each P&L")
    out = {}
    print(f"  {'position':<12}{'mean P&L':>13}{'sd P&L':>13}{'Normal VaR 5%':>15}"
          f"{'hist VaR 5%':>14}")
    print("  " + "-" * 67)
    for name, x in D["pnl"].items():
        mu, sd = x.mean(), x.std(ddof=1)
        v = common.var_normal(sd, 0.05, mu=mu)            # -(mu + sd * Phi^-1(0.05))
        out[name] = {"mu": mu, "sd": sd, "var": v}
        print(f"  {name:<12}{common.money(mu):>13}{common.money(sd):>13}{common.money(v):>15}"
              f"{common.money(common.var_es(x, 0.05)[0]):>14}")
    print("  " + "-" * 67)
    print("  Convention: VaR = -(mu + sigma Phi^-1(alpha)), the absolute P&L quantile of")
    print("  Week 04 s4.2.1, since the question asks for the mean to be used. Measured")
    print("  from the mean instead, each number would be larger by the mean P&L.")
    sd_c, sd_d = out["$2M in A"]["sd"], out["$1M each"]["sd"]
    print(f"\n  sd($1M each) / sd($2M in A) = {sd_d / sd_c:.3f}")
    print(f"  (two independent, equal-size positions give exactly 1/sqrt(2) = 0.707)")
    return out


def part_e(D: dict) -> dict:
    common.header("Problem 3(e) -- historical VaR and ES at alpha = 1%")
    return risk_table(D, 0.01)


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------
def part_f(D: dict, P: dict, R5: dict) -> dict:
    common.header("Problem 3(f) -- subadditivity at 5%")
    va, vb, vab = R5["$1M in A"][0], R5["$1M in B"][0], R5["$1M each"][0]
    ea, eb, eab = R5["$1M in A"][1], R5["$1M in B"][1], R5["$1M each"][1]
    print(f"  VaR: q(A) + q(B) = {common.money(va)} + {common.money(vb)} = "
          f"{common.money(va + vb)}   vs   q(A+B) = {common.money(vab)}")
    print(f"       subadditive (q(A)+q(B) >= q(A+B))?  "
          f"{'YES' if va + vb >= vab else 'NO -- VIOLATED'}")
    print(f"  ES : q(A) + q(B) = {common.money(ea)} + {common.money(eb)} = "
          f"{common.money(ea + eb)}   vs   q(A+B) = {common.money(eab)}")
    print(f"       subadditive?  {'YES' if ea + eb >= eab else 'NO -- VIOLATED'}")
    n = D["n"]
    print(f"\n  Mechanism (Week 04 s4.3.2). Each bond defaults in {100 * P['n_a'] / n:.2f}% / "
          f"{100 * P['n_b'] / n:.2f}% of")
    print(f"  scenarios, both below 5%, so each bond's 5% quantile is a coupon year and")
    print(f"  its VaR is a profit. Together, at least one defaults in "
          f"{100 * P['n_either'] / n:.2f}% of")
    print(f"  scenarios, above 5%, so the combined 5% quantile is a default. Splitting")
    print(f"  the money moved {100 * (P['n_either'] - max(P['n_a'], P['n_b'])) / n:.2f}% of "
          f"probability mass across the 5% line, and a")
    print(f"  measure that reads one quantile sees nothing on the far side of it.")
    print(f"  ES averages the whole tail: it already counted every default in A and in B")
    print(f"  separately, so combining cannot surprise it. It falls because a single")
    print(f"  default now costs $1M x loss instead of $2M x loss, and the {P['n_both']} double")
    print(f"  defaults (0.17%) are too few to offset that.")
    return {"var_sub": va + vb >= vab, "es_sub": ea + eb >= eab}


def part_g(R5: dict, N: dict) -> None:
    common.header("Problem 3(g) -- what the Normal VaR gets right, and why it is still wrong")
    nc, nd = N["$2M in A"]["var"], N["$1M each"]["var"]
    hc, hd = R5["$2M in A"][0], R5["$1M each"][0]
    ec, ed = R5["$2M in A"][1], R5["$1M each"][1]
    print(f"  {'measure':<14}{'$2M in A':>12}{'$1M each':>12}   favours")
    print("  " + "-" * 60)
    print(f"  {'Normal VaR 5%':<14}{common.money(nc):>12}{common.money(nd):>12}   "
          f"$1M each by {common.money(nc - nd)}")
    print(f"  {'Hist VaR 5%':<14}{common.money(hc):>12}{common.money(hd):>12}   "
          f"$2M in A by {common.money(hd - hc)}")
    print(f"  {'Hist ES 5%':<14}{common.money(ec):>12}{common.money(ed):>12}   "
          f"$1M each by {common.money(ec - ed)}")
    print("  " + "-" * 60)
    print(f"\n  The Normal favours diversification because sd falls by 1/sqrt(2) when two")
    print(f"  independent, equal positions are combined, and a Normal VaR is 1.645 sd.")
    print(f"  That is the only mechanism it has. It is right that diversification")
    print(f"  reduces the expected size of a bad year (ES agrees), but for the wrong")
    print(f"  reason: the real benefit is that one default now costs half as much, and")
    print(f"  the real cost -- which the 5% VaR reports -- is that defaults become twice")
    print(f"  as frequent in the portfolio. A Normal has no defaults in it at all; its")
    print(f"  numbers describe a bell curve around +$170k that no scenario follows.")


def part_h(D: dict, P: dict, R5: dict, R1: dict) -> None:
    common.header("Problem 3(h) -- why 1% flips the VaR result")
    n = D["n"]
    print(f"  {'':<12}{'VaR 5%':>14}{'VaR 1%':>14}{'ES 5%':>14}{'ES 1%':>14}")
    for name in ("$2M in A", "$1M each"):
        print(f"  {name:<12}{common.money(R5[name][0]):>14}{common.money(R1[name][0]):>14}"
              f"{common.money(R5[name][1]):>14}{common.money(R1[name][1]):>14}")
    print(f"\n  At 1% the threshold sits BELOW every single-default probability:")
    print(f"    P(A defaults)          = {100 * P['n_a'] / n:.2f}%  > 1%  -> concentrated VaR now"
          f" sees a $2M default")
    print(f"    P(at least one)        = {100 * P['n_either'] / n:.2f}%  > 1%  -> diversified VaR"
          f" still sees one $1M default")
    print(f"    P(both default)        = {100 * P['n_both'] / n:.2f}%  < 1%  -> and never the"
          f" double default")
    print(f"  so the diversified VaR is about half the concentrated one and VaR is")
    print(f"  subadditive again. VaR fails when alpha falls BETWEEN the probability of")
    print(f"  the bad event in one position and in the combination -- here between "
          f"{100 * max(P['n_a'], P['n_b']) / n:.1f}%")
    print(f"  and {100 * P['n_either'] / n:.1f}%. 5% is inside that window; 1% is not. "
          f"Whether VaR 'works' is")
    print(f"  not a property of the portfolio; it depends on where alpha lands relative")
    print(f"  to the lumps of probability in the loss distribution, which is exactly")
    print(f"  what a risk measure should not depend on. ES has no such window.")


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def figure(D: dict, R5: dict | None = None, R1: dict | None = None,
           name: str = "problem3") -> None:
    """P&L histograms of both choices (log count scale, so the default lumps
    are visible next to the coupon spike) and the empirical CDFs in the loss
    region. Called once in the Predict stage without risk measures, and again
    after the fit with the VaR/ES lines drawn in."""
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    bins = np.linspace(-1.75e6, 0.45e6, 89)
    for k, (pname, col) in enumerate((("$2M in A", C2), ("$1M each", C1))):
        x = D["pnl"][pname]
        ax[k].hist(x / 1e6, bins=bins / 1e6, color=col, alpha=0.6, edgecolor="white",
                   linewidth=0.3)
        ax[k].set_yscale("log")
        ax[k].set_ylim(0.7, 3e4)
        for a, R, ls in ((0.05, R5, "-"), (0.01, R1, "--")):
            if R is None:
                continue
            v, e = R[pname]
            ax[k].axvline(-v / 1e6, color=common.INK, linestyle=ls, linewidth=1.1)
            ax[k].axvline(-e / 1e6, color=C3, linestyle=ls, linewidth=1.1)
            ax[k].text(-v / 1e6, 1.2e4, f"VaR {int(100 * a)}%", rotation=90, fontsize=7,
                       ha="right", va="top", color=common.INK)
            ax[k].text(-e / 1e6, 1.2e4, f"ES {int(100 * a)}%", rotation=90, fontsize=7,
                       ha="right", va="top", color=C3)
        ax[k].set_title(f"P&L, {pname}")
        ax[k].set_xlabel("P&L ($ millions)")
        ax[k].set_ylabel("scenarios (log)")

    # Empirical CDFs of the loss region with the two alpha levels.
    for pname, col in (("$2M in A", C2), ("$1M each", C1)):
        x = np.sort(D["pnl"][pname]) / 1e6
        p = np.arange(1, x.size + 1) / x.size
        ax[2].step(x, p, where="post", color=col, label=pname)
    for a in (0.05, 0.01):
        ax[2].axhline(a, color=common.INK2, linestyle=":", linewidth=1)
        ax[2].text(0.25, a * 1.1, f"alpha = {int(100 * a)}%", fontsize=8, color=common.INK2,
                   ha="right")
    ax[2].set_yscale("log")
    ax[2].set_ylim(5e-4, 0.3)
    ax[2].set_xlim(-1.8, 0.3)
    ax[2].set_title("Empirical CDF of P&L: where alpha cuts each curve")
    ax[2].set_xlabel("P&L ($ millions)")
    ax[2].set_ylabel("P(P&L <= x)  (log)")
    ax[2].legend(loc="upper left")
    title = ("Problem 3 -- two bonds: diversification lowers ES and raises the 5% VaR"
             if R5 is not None else
             "Problem 3 -- Predict: P&L of the two choices, before any risk measure")
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, name)


def main() -> dict:
    D = prepare()
    P = part_predict(D)
    figure(D, name="problem3_predict")          # the plot the Predict stage asks for
    R5 = part_c(D)
    N = part_d(D)
    R1 = part_e(D)
    S = part_f(D, P, R5)
    part_g(R5, N)
    part_h(D, P, R5, R1)
    figure(D, R5, R1)                           # same plot with VaR / ES drawn in
    return {"D": D, "P": P, "R5": R5, "N": N, "R1": R1, "S": S}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

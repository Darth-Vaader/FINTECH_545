"""
Problem 4 -- Conditional distributions.

problem4.csv holds two correlated series x1 and x2. We
  (a) write the conditional variance of x2 | x1 in blocks of Sigma and give the
      variance-reduction factor as a formula and a number,
  (b) argue from the formula whether that factor depends on the OBSERVED x1,
  (c) write the conditional mean, identify the regression coefficient, and plot
      E[x2|x1] with a 95% band over the scatter,
  (d) report overall coverage of the band,
  (e) report coverage split by |x1 - mean| / sd in three buckets,
  (f) explain the non-flat coverage and name the failed assumption.

Run:  python problem4.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats

import common
from common import C1, C2, C3


# --------------------------------------------------------------------------
# (a) + (b) prediction: only the covariance matrix is allowed to be looked at
# --------------------------------------------------------------------------
def part_ab(x1: np.ndarray, x2: np.ndarray) -> dict:
    """Parts (a) and (b): the conditional variance, in the blocks of Sigma.

    BLOCK CONVENTION -- this follows Week 02 section 2.4 exactly, and it is worth
    being careful because the slides' subscripts are the opposite way round from
    the "obvious" reading of this problem.

    Week 02 s2.4 partitions X = [x1 ; x2], CONDITIONS ON THE SECOND BLOCK
    (given X2 = a) and describes the distribution of the FIRST:

        Xbar_1 ~ N(mubar, Sigmabar)
        mubar    = mu_1 + Sigma_12 Sigma_22^{-1} (a - mu_2)
        Sigmabar = Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21

    Here the assignment asks for x2 GIVEN x1, so the variable we learn must sit in
    the slides' second block. Mapping this data onto the slide notation:

        slide block 1  <-  the CSV's x2   (the target: what we want the
                                           distribution of)
        slide block 2  <-  the CSV's x1   (the conditioning variable: what we
                                           learn)

    i.e. the partition is [x2 ; x1], NOT [x1 ; x2]. So, in slide subscripts,

        Sigma_11 = Var(x2)        the TARGET block
        Sigma_22 = Var(x1)        the CONDITIONING block
        Sigma_12 = Cov(x2, x1)    the cross block

    and the conditional variance Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21 reads
    Var(x2) - Cov(x2,x1)^2 / Var(x1). Getting this backwards would still produce
    a number, and on a 2x2 problem it would be a plausible-looking one, so the
    mapping is stated explicitly rather than left for the reader to infer.
    """
    # Sample covariance matrix. Built in the CSV's own order [x1, x2] because
    # that is what "compute the sample covariance matrix" means; the block
    # relabelling above is applied when the partitioned formula is used.
    S = np.cov(np.column_stack([x1, x2]), rowvar=False)
    v_x1, cov_12, v_x2 = S[0, 0], S[0, 1], S[1, 1]
    rho = cov_12 / np.sqrt(v_x1 * v_x2)

    # In SLIDE subscripts, with the [x2 ; x1] partition:
    Sig11 = v_x2      # Var(x2)     -- target
    Sig22 = v_x1      # Var(x1)     -- conditioning
    Sig12 = cov_12    # Cov(x2, x1) -- cross

    cond_var = Sig11 - Sig12 * Sig12 / Sig22   # Sigma_11 - Sigma_12 Sig22^-1 Sig21
    factor = cond_var / Sig11                  # = 1 - rho^2
    beta = Sig12 / Sig22                       # Sigma_12 Sigma_22^{-1}

    common.header("Problem 4(a) -- conditional variance of x2 given x1")
    print("  Sample covariance matrix, in the CSV's own [x1, x2] order:")
    print(f"    [[{v_x1:8.5f}, {cov_12:8.5f}],")
    print(f"     [{cov_12:8.5f}, {v_x2:8.5f}]]")
    print(f"    implied correlation rho = {rho:.5f}")
    print()
    print("  WHICH SERIES IS WHICH BLOCK.")
    print("  Week 02 s2.4 conditions on the SECOND block and returns the")
    print("  distribution of the FIRST. We want x2 GIVEN x1, so x1 has to be the")
    print("  second block. The partition is therefore [x2 ; x1]:")
    print(f"    Sigma_11 = Var(x2)     = {Sig11:.5f}   <- TARGET (block 1)")
    print(f"    Sigma_22 = Var(x1)     = {Sig22:.5f}   <- CONDITIONING (block 2)")
    print(f"    Sigma_12 = Cov(x2, x1) = {Sig12:.5f}   <- cross block")
    print()
    print("  Week 02 s2.4 partitioned result, as written on the slide:")
    print("    Var(x2 | x1) = Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21")
    print(f"                 = {Sig11:.5f} - ({Sig12:.5f})^2 / {Sig22:.5f}")
    print(f"                 = {cond_var:.5f}      (conditional sd "
          f"= {np.sqrt(cond_var):.5f})")
    print()
    print("  VARIANCE-REDUCTION FACTOR, as a formula in the same blocks:")
    print("    Var(x2|x1) / Var(x2)")
    print("        = 1 - Sigma_12 Sigma_22^{-1} Sigma_21 / Sigma_11")
    print("        = 1 - rho^2")
    print(f"    numerically = 1 - {rho:.5f}^2 = {factor:.5f}")
    print(f"  So learning x1 removes {100 * (1 - factor):.1f}% of the VARIANCE of x2.")
    print(f"  In standard-deviation terms the uncertainty falls by a factor of")
    print(f"  sqrt({factor:.5f}) = {np.sqrt(factor):.5f}, i.e. sd drops from "
          f"{np.sqrt(Sig11):.4f} to {np.sqrt(cond_var):.4f}")
    print(f"  -- a {100 * (1 - np.sqrt(factor)):.1f}% reduction in sd, which is the")
    print(f"  honest way to quote it since sd is what a risk band is built from.")
    print()
    print(f"  Sanity check from s2.4: Sigma_12 Sigma_22^-1 Sigma_21 is positive")
    print(f"  semi-definite, so conditioning can never INCREASE variance. Here it")
    print(f"  is {Sig12 * Sig12 / Sig22:.5f} >= 0, and the variance falls. OK.")

    common.header("Problem 4(b) -- does the factor depend on the value of x1?")
    print("  No. Answering from the formula alone, before looking at the data:")
    print("    Var(x2 | x1 = a) = Sigma_11 - Sigma_12 Sigma_22^{-1} Sigma_21")
    print("  The right-hand side is built ONLY from blocks of Sigma. The observed")
    print("  value a appears nowhere in it -- it appears only in the conditional")
    print("  MEAN, mu_1 + Sigma_12 Sigma_22^{-1} (a - mu_2).")
    print("  THE TERM THAT SETTLES IT: Sigma_12 Sigma_22^{-1} Sigma_21 is a")
    print("  product of covariance blocks with no a in it.")
    print("  Week 02 s2.4 makes the same point in words: the conditional variance")
    print("  'does not depend on a at all, so learning the value of X2 reduces the")
    print("  uncertainty in X1 by the same amount regardless of what value showed")
    print("  up.'")
    print("  Consequence: under the multivariate Normal the conditional variance")
    print("  is HOMOSKEDASTIC in x1, so the 95% band has constant width. That is")
    print("  a property of the Normal, not a general fact -- and part (f) is")
    print("  where this data declines to cooperate.")

    return {"S": S, "Sig11": Sig11, "Sig22": Sig22, "Sig12": Sig12,
            "v_x1": v_x1, "v_x2": v_x2, "cov_12": cov_12, "rho": rho,
            "cond_var": cond_var, "factor": factor, "beta": beta}


# --------------------------------------------------------------------------
# (c) conditional mean + band, (d) overall coverage, (e) bucketed coverage
# --------------------------------------------------------------------------
def part_cde(x1: np.ndarray, x2: np.ndarray, B: dict) -> dict:
    # Slide subscripts throughout, per the [x2 ; x1] partition fixed in part_ab:
    # Sigma_11 = Var(x2) (target), Sigma_22 = Var(x1) (conditioning),
    # Sigma_12 = Cov(x2, x1).
    Sig11, Sig12, Sig22 = B["Sig11"], B["Sig12"], B["Sig22"]
    cond_var, beta = B["cond_var"], B["beta"]
    mu1, mu2 = x1.mean(), x2.mean()
    cond_sd = np.sqrt(cond_var)

    common.header("Problem 4(c) -- conditional mean of x2 given x1")
    print("  Week 02 s2.4, in the same [x2 ; x1] blocks as part (a):")
    print("    E[x2 | x1 = a] = mu_2 + Sigma_12 Sigma_22^{-1} (a - mu_1)")
    print("  (slide block 1 is x2, so the slide's mu_1 is this data's mean of x2;")
    print("   renaming to match the CSV to avoid confusion:)")
    print(f"    E[x2 | x1 = a] = mean(x2) + Sigma_12 Sigma_22^{{-1}} (a - mean(x1))")
    print(f"                   = {mu2:.5f} + ({Sig12:.5f} / {Sig22:.5f}) "
          f"(a - {mu1:.5f})")
    print(f"                   = {mu2:.5f} + {beta:.5f} (a - {mu1:.5f})")
    print()
    print(f"  THE COEFFICIENT ON x1 is Sigma_12 Sigma_22^{{-1}} = {beta:.5f}.")
    print(f"  In regression terms this is exactly the OLS slope of x2 on x1:")
    ols = sm.OLS(x2, sm.add_constant(x1)).fit()
    print(f"    OLS slope     = {ols.params[1]:.5f}   (identical, as it must be)")
    print(f"    OLS intercept = {ols.params[0]:.5f}  = mu_2 - beta*mu_1 = "
          f"{mu2 - beta * mu1:.5f}")
    print(f"  So 'conditioning a bivariate Normal' and 'running a regression'")
    print(f"  are the same computation; the partitioned formula is where the")
    print(f"  regression coefficient comes from. Week 02 s2.4 says so directly:")
    print(f"  'The expression for mubar is a regression. Sigma_12 Sigma_22^-1 is")
    print(f"  the matrix of coefficients from regressing X1 on X2', and s3 then")
    print(f"  derives the same quantity from least squares instead.")
    print()
    # Print the multiplier at full precision rather than as "1.96": the band used
    # below is built from norm.ppf(0.975) = 1.95996, and quoting a rounded 1.96
    # beside an exactly-computed half-width would not reconcile.
    z_ = stats.norm.ppf(0.975)
    print(f"  The 95% band is E[x2|x1] +/- z * sqrt(Var(x2|x1)), where")
    print(f"  z = norm.ppf(0.975) = {z_:.5f}")
    print(f"                = E[x2|x1] +/- {z_:.5f} * {cond_sd:.5f}")
    print(f"                = E[x2|x1] +/- {z_ * cond_sd:.5f}   (CONSTANT width)")
    print(f"  NOTE this is a PREDICTION band for a new observation, not a")
    print(f"  confidence band for the fitted line -- the question asks what")
    print(f"  fraction of OBSERVATIONS fall inside, so the band has to be the")
    print(f"  one observations are supposed to fall into.")

    z = stats.norm.ppf(0.975)   # 1.959964, not 2
    pred = mu2 + beta * (x1 - mu1)
    lo, hi = pred - z * cond_sd, pred + z * cond_sd
    inside = (x2 >= lo) & (x2 <= hi)

    common.header("Problem 4(d) -- overall coverage of the 95% band")
    print(f"  inside: {int(inside.sum())} of {x2.size}  -> "
          f"coverage = {inside.mean():.4f}")
    print(f"  nominal 95%; a Monte-Carlo se at n={x2.size} is about "
          f"{np.sqrt(0.95 * 0.05 / x2.size):.4f},")
    print(f"  so {inside.mean():.4f} is "
          f"{(inside.mean() - 0.95) / np.sqrt(0.95 * 0.05 / x2.size):+.1f} se from"
          f" nominal -- close overall.")
    print(f"  Overall coverage being close is exactly what hides the problem")
    print(f"  that part (e) exposes.")

    # (e) Bucket by distance of x1 from its mean, in sd units.
    sd1 = x1.std(ddof=1)
    dev = np.abs(x1 - mu1) / sd1
    buckets = [("|x1 - mu1| <= 1 sd", dev <= 1),
               ("1 sd < ... <= 2 sd", (dev > 1) & (dev <= 2)),
               ("        ... >  2 sd", dev > 2)]

    common.header("Problem 4(e) -- coverage within each |x1| bucket")
    print(f"  {'bucket':<22}{'n':>6}{'coverage':>11}{'resid sd':>11}")
    print("  " + "-" * 50)
    resid = x2 - pred
    out = []
    for name, mask in buckets:
        cov = float(inside[mask].mean())
        rsd = float(resid[mask].std(ddof=1))
        out.append({"bucket": name, "n": int(mask.sum()), "coverage": cov,
                    "resid_sd": rsd})
        print(f"  {name:<22}{int(mask.sum()):>6}{cov:>11.4f}{rsd:>11.4f}")
    print("  " + "-" * 50)
    print(f"  {'pooled':<22}{x2.size:>6}{inside.mean():>11.4f}"
          f"{resid.std(ddof=1):>11.4f}")

    # (f) Diagnose. The residual sd column above already tells the story; back it
    # with formal tests so the claim is not just a pattern in three numbers.
    common.header("Problem 4(f) -- why coverage is not flat")
    bp = sm.stats.diagnostic.het_breuschpagan(ols.resid, sm.add_constant(x1))
    wh = sm.stats.diagnostic.het_white(
        ols.resid, sm.add_constant(np.column_stack([x1, x1 ** 2])))
    print(f"  The residual sd is not constant across the buckets:")
    print(f"    {out[0]['resid_sd']:.4f} (centre) -> {out[1]['resid_sd']:.4f} -> "
          f"{out[2]['resid_sd']:.4f} (tails)")
    print(f"  Formal tests of that:")
    print(f"    Breusch-Pagan (linear in x1): LM = {bp[0]:.3f}, p = {bp[1]:.4f}")
    print(f"    White (allows x1^2, so it can see a U shape):")
    print(f"        LM = {wh[0]:.3f}, p = {wh[1]:.3e}")
    print(f"  White is overwhelming while Breusch-Pagan is marginal, and that")
    print(f"  contrast is itself informative: the variance is not monotone in x1,")
    print(f"  it grows with |x1|, which a test linear in x1 largely misses.")
    print()
    print(f"  THE FAILED ASSUMPTION: homoskedasticity of the conditional")
    print(f"  distribution -- equivalently, joint normality of (x1, x2). A")
    print(f"  bivariate Normal must have Var(x2|x1) free of x1 (part b). Here it")
    print(f"  clearly is not, so the pair is not jointly Normal even though each")
    print(f"  margin may look fine. The constant-width band inherits that error:")
    print(f"  it is too WIDE near the centre (over-covers at "
          f"{out[0]['coverage']:.3f}) and too NARROW")
    print(f"  out in the wings (under-covers at "
          f"{min(out[1]['coverage'], out[2]['coverage']):.3f}).")
    print()
    print(f"  WHAT SURVIVES from part (c), and what does not:")
    print(f"    SURVIVES -- the conditional MEAN. beta = Sigma_12 Sigma_22^-1 =")
    print(f"      {beta:.5f} is the best linear predictor of x2 given x1, and that")
    print(f"      is a statement about second moments only. It needs no")
    print(f"      normality and no homoskedasticity, so the LINE is still the")
    print(f"      right line and remains unbiased.")
    print(f"    DOES NOT SURVIVE -- everything about the WIDTH. The single number")
    print(f"      Var(x2|x1) = {cond_var:.5f} is only an average of a quantity that")
    print(f"      actually varies with |x1|; the constant-width band built from")
    print(f"      it is mis-calibrated pointwise; and the reported se(beta) from")
    print(f"      OLS is not trustworthy either.")
    print(f"  Fixes, in increasing order of commitment: HC3 robust standard")
    print(f"  errors for inference on beta; a variance function / WLS or a GARCH-")
    print(f"  style model for the band; or a copula if the goal is the joint law.")
    rob = sm.OLS(x2, sm.add_constant(x1)).fit(cov_type="HC3")
    print(f"    e.g. se(beta): {ols.bse[1]:.5f} (OLS) vs {rob.bse[1]:.5f} (HC3)")

    return {"pred": pred, "inside": inside, "cond_sd": cond_sd, "z": z,
            "mu1": mu1, "mu2": mu2, "sd1": sd1, "buckets": out,
            "resid": resid, "coverage": float(inside.mean())}


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def figures(x1: np.ndarray, x2: np.ndarray, B: dict, R: dict) -> None:
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 4.0))

    # Panel 1: scatter + conditional mean + constant-width 95% band, with the
    # points that fall OUTSIDE ringed so the failure is visible, not asserted.
    grid = np.linspace(x1.min(), x1.max(), 200)
    gpred = R["mu2"] + B["beta"] * (grid - R["mu1"])
    ax[0].fill_between(grid, gpred - R["z"] * R["cond_sd"],
                       gpred + R["z"] * R["cond_sd"], color=C1, alpha=0.13,
                       linewidth=0, label="95% band (constant width)")
    ax[0].scatter(x1[R["inside"]], x2[R["inside"]], s=9, color=C1, alpha=0.5,
                  edgecolor="none", label="inside band")
    ax[0].scatter(x1[~R["inside"]], x2[~R["inside"]], s=22, color=C2,
                  alpha=0.9, edgecolor="white", linewidth=0.6,
                  label=f"outside (n={int((~R['inside']).sum())})")
    ax[0].plot(grid, gpred, color=common.INK, linewidth=2.0,
               label=f"E[x2|x1], slope {B['beta']:.3f}")
    # Mark the bucket boundaries, since they are what part (e) splits on.
    for k in (1, 2):
        for s in (-1, 1):
            ax[0].axvline(R["mu1"] + s * k * R["sd1"], color=common.INK2,
                          linestyle=":", linewidth=0.9)
    ax[0].set_title(f"Conditional mean with 95% band "
                    f"(coverage {R['coverage']:.3f})")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].legend(loc="upper left", fontsize=7.5)

    # Panel 2: coverage by bucket. A bar chart of magnitudes against a nominal
    # reference line; bars are direct-labelled so no colour-only reading.
    # A DOT plot, not a bar chart: the interesting range is 0.85-0.96, and a bar
    # chart may not have a truncated baseline (the bar length would stop
    # encoding the value). Dots carry position only, so clipping the axis is
    # legitimate. Each dot is direct-labelled with its value and bucket n, and a
    # binomial 95% interval shows which departures are bigger than sampling noise.
    names = ["<= 1 sd", "1-2 sd", "> 2 sd"]
    covs = [b["coverage"] for b in R["buckets"]]
    ns = [b["n"] for b in R["buckets"]]
    xs_ = np.arange(3)
    ax[1].axhline(0.95, color=C2, linewidth=1.6, linestyle="--", zorder=1)
    ax[1].annotate("nominal 95%", xy=(2.42, 0.95), xytext=(0, 5),
                   textcoords="offset points", fontsize=8, color=C2,
                   ha="right")
    for xi, c, n_ in zip(xs_, covs, ns):
        # Wilson interval, which behaves properly for proportions near 1.
        lo_, hi_ = stats.binomtest(int(round(c * n_)), n_).proportion_ci(
            confidence_level=0.95, method="wilson")
        ax[1].vlines(xi, lo_, hi_, color=C1, linewidth=2.0, alpha=0.45)
    ax[1].scatter(xs_, covs, s=110, color=C1, zorder=5, edgecolor="white",
                  linewidth=1.6)
    for xi, c, n_ in zip(xs_, covs, ns):
        ax[1].annotate(f"{c:.3f}  (n={n_})", xy=(xi, c), xytext=(0, 13),
                       textcoords="offset points", ha="center", fontsize=8.5,
                       color=common.INK)
    ax[1].set_xticks(xs_)
    ax[1].set_xticklabels(names)
    ax[1].set_xlim(-0.5, 2.5)
    ax[1].set_ylim(0.80, 1.0)
    ax[1].set_title("Coverage is not flat across |x1| buckets")
    ax[1].set_xlabel("distance of x1 from its mean")
    ax[1].set_ylabel("empirical coverage  (bars: 95% Wilson CI)")

    # Panel 3: the mechanism. Binned against |x1 - mu1| rather than signed x1,
    # because the claim being tested is that the spread grows with DISTANCE from
    # the centre; folding the two sides together also doubles the points per bin,
    # which a signed-x1 version does not have enough of in the far tails to
    # estimate an sd stably.
    dev = np.abs(x1 - R["mu1"])
    ax[2].scatter(dev, np.abs(R["resid"]), s=8, color=C1, alpha=0.30,
                  edgecolor="none", label="|residual| (one point per obs)")
    # Equal-count (quantile) bins on |x1 - mu1|, so every plotted sd rests on the
    # same number of observations and the marker sizes are comparable.
    edges = np.quantile(dev, np.linspace(0, 1, 11))
    centres, sds, counts = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (dev >= a) & (dev <= b)
        if m.sum() >= 20:
            centres.append(dev[m].mean())
            sds.append(R["resid"][m].std(ddof=1))
            counts.append(int(m.sum()))
    ax[2].plot(centres, sds, color=C2, marker="o", markersize=6,
               markeredgecolor="white", markeredgewidth=1.3,
               label="binned residual sd (deciles of |x1 - mu1|)")
    ax[2].axhline(R["cond_sd"], color=C3, linestyle="--", linewidth=1.6,
                  label=f"sd the band assumes = {R['cond_sd']:.3f}")
    # Direct-label the two ends, which is the whole comparison.
    for i in (0, len(centres) - 1):
        ax[2].annotate(f"{sds[i]:.2f}", xy=(centres[i], sds[i]),
                       xytext=(0, 10), textcoords="offset points",
                       ha="center", fontsize=8.5, color=common.INK)
    ax[2].set_title("Residual spread grows with distance from the centre")
    ax[2].set_xlabel("|x1 - mean(x1)|")
    ax[2].set_ylabel("|residual|  /  binned sd")
    ax[2].legend(loc="upper left", fontsize=7.5)

    fig.suptitle("Problem 4 -- the conditional mean survives, the constant-width band does not",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem4")


def main() -> dict:
    d = common.load("problem4")
    x1, x2 = d["x1"].to_numpy(), d["x2"].to_numpy()
    B = part_ab(x1, x2)
    R = part_cde(x1, x2, B)
    figures(x1, x2, B, R)
    return {"blocks": B, "results": R}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

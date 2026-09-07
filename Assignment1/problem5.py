"""
Problem 5 -- Identifying an AR or MA order.

problem5.csv holds a single series x. We
  (a) plot the series, its ACF and its PACF, and commit to an order/family,
  (b) state the identification rule and the significance band used,
  (c) fit AR(1..3) and MA(1..3) and report AICc for each,
  (d) say which AICc selects and whether it matched,
  (e) compare AR(2) vs AR(3) and explain the AICc trade-off, and why R^2 would
      not have made the same choice.

Run:  python problem5.py
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf

import common
from common import C1, C2, C3

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# (a) + (b) prediction from the ACF / PACF
# --------------------------------------------------------------------------
def part_ab(x: np.ndarray) -> dict:
    n = x.size
    NL = 15  # lags to inspect

    # CONVENTION on the PACF estimator: Yule-Walker ('ywm'). The default in some
    # libraries is OLS-based, which can return values outside [-1,1] at longer
    # lags. Nothing here hinges on the choice at lags 1-3, where the estimators
    # agree closely, but it is a real choice so it is named.
    a, a_ci = acf(x, nlags=NL, alpha=0.05)
    p, p_ci = pacf(x, nlags=NL, alpha=0.05, method="ywm")

    # SIGNIFICANCE BAND. Two bands exist and they answer different questions:
    #   * +/- 1.96/sqrt(n)  -- Bartlett band under the null of WHITE NOISE. This
    #     is the correct band for the "cuts off" judgement, because "cuts off
    #     after lag q" means "lags beyond q are indistinguishable from zero".
    #   * the widening Bartlett band returned in a_ci, which conditions on the
    #     lower-order autocorrelations being real, is appropriate for reading an
    #     ACF decay, and is plotted alongside for honesty.
    band = 1.96 / np.sqrt(n)

    common.header("Problem 5(a,b) -- reading the ACF and PACF")
    adf = adfuller(x, autolag="AIC")
    print(f"  n = {n}, mean = {x.mean():.4f}, sd = {x.std(ddof=1):.4f}")
    print(f"  ADF test for a unit root: stat = {adf[0]:.3f}, p = {adf[1]:.2e}")
    print(f"  -> stationary, so no differencing; identify on the levels.")
    print()
    print(f"  Significance band used for the 'cuts off' call:")
    print(f"    +/- 1.96/sqrt(n) = +/- {band:.4f}   (Bartlett, H0 = white noise)")
    print()
    print(f"  {'lag':>4}{'ACF':>10}{'|ACF|>band':>12}{'PACF':>10}"
          f"{'|PACF|>band':>13}")
    print("  " + "-" * 50)
    for k in range(1, NL + 1):
        print(f"  {k:>4}{a[k]:>10.4f}{'yes' if abs(a[k]) > band else '.':>12}"
              f"{p[k]:>10.4f}{'yes' if abs(p[k]) > band else '.':>13}")
    print("  " + "-" * 50)

    sig_acf = [k for k in range(1, NL + 1) if abs(a[k]) > band]
    sig_pacf = [k for k in range(1, NL + 1) if abs(p[k]) > band]
    print(f"  ACF  outside the band at lags:  {sig_acf}")
    print(f"  PACF outside the band at lags:  {sig_pacf}")
    print()
    print("  THE RULE (Week 02 s6.1, Table 3 -- the ACF/PACF identification")
    print("  table; the time-series material is in the Week 02 deck):")
    print("    ACF cuts off after lag q, PACF decays      -> MA(q)")
    print("    PACF cuts off after lag p, ACF decays      -> AR(p)")
    print("    neither cuts off, both decay               -> ARMA(p,q)")
    print("  s6.1 states why both are always reported together: 'Table 3 is the")
    print("  reason both functions are reported together. Either one alone is")
    print("  ambiguous.' The PACF is defined there as 'the autocorrelation of the")
    print("  residual of the series regressed on the n-1 lags', which is what")
    print("  makes the cut-off argument below work.")
    print("  The reasoning behind it: an AR(p) has exactly p non-zero partial")
    print("  autocorrelations by construction, since the PACF at lag k is the")
    print("  coefficient on x_{t-k} in a regression that already contains lags")
    print("  1..k-1 -- past lag p there is nothing left to add. An MA(q) is the")
    print("  mirror image: its ACF is exactly zero past lag q.")
    print()
    print("  PREDICTION, committed before fitting:")
    print(f"    The PACF CUTS OFF: it is large at lags 1 ({p[1]:+.3f}) and 2")
    print(f"    ({p[2]:+.3f}), then drops inside the band and stays there through")
    print(f"    lag 8 ({p[3]:+.3f}, {p[4]:+.3f}, {p[5]:+.3f}, ...).")
    print(f"    The ACF DECAYS: {a[1]:+.3f}, {a[2]:+.3f}, {a[3]:+.3f}, "
          f"{a[4]:+.3f}, {a[5]:+.3f} -- it")
    print(f"    changes sign and tapers rather than stopping dead, which is the")
    print(f"    signature of an AR with a negative second coefficient (a damped")
    print(f"    oscillation), not of an MA. Week 02 s6.3 derives rho(k) = beta^k")
    print(f"    for the AR(1), which is monotone when beta > 0; the sign flips")
    print(f"    here require the second-order term.")
    print(f"    => AR(2). Order 2, autoregressive.")
    print()
    print(f"  Caveat recorded up front: PACF lag 9 ({p[9]:+.3f}) and ACF lag 10")
    print(f"  ({a[10]:+.3f}) poke just outside the band. With 15 lags tested at")
    print(f"  the 5% level we expect about {0.05 * NL:.2f} false positives, so one")
    print(f"  or two marginal spikes at long lags with no mechanism behind them")
    print(f"  is what noise looks like. I am not reading them as structure.")

    # ---------------- figure ----------------
    common.use_style()
    fig = plt.figure(figsize=(13.0, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.42,
                          wspace=0.22)

    # Top: the series itself.
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(np.arange(n), x, color=C1, linewidth=1.0)
    ax0.axhline(x.mean(), color=C2, linestyle="--", linewidth=1.4,
                label=f"mean {x.mean():.3f}")
    ax0.set_title("The series: mean-reverting, no trend, no variance drift")
    ax0.set_xlabel("t")
    ax0.set_ylabel("x")
    ax0.legend(loc="upper right", bbox_to_anchor=(0.995, 1.02))

    lags = np.arange(1, NL + 1)

    # Bottom-left: ACF. Stems + the white-noise band.
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.vlines(lags, 0, a[1:], color=C1, linewidth=2.0)
    ax1.scatter(lags, a[1:], s=30, color=C1, zorder=5, edgecolor="white",
                linewidth=1.2)
    ax1.axhline(0, color=common.INK2, linewidth=0.9)
    ax1.fill_between([0.4, NL + 0.6], -band, band, color=C2, alpha=0.12,
                     linewidth=0, label=f"+/- 1.96/sqrt(n) = {band:.3f}")
    ax1.set_xlim(0.4, NL + 0.6)
    # Headroom so the lag-1 marker is not clipped by the axes frame.
    apad = 0.10 * (a[1:].max() - a[1:].min())
    ax1.set_ylim(a[1:].min() - apad, a[1:].max() + 2.2 * apad)
    ax1.set_title("ACF: decays and changes sign (does not cut off)")
    ax1.set_xlabel("lag")
    ax1.set_ylabel("autocorrelation")
    ax1.legend(loc="upper right", fontsize=8)

    # Bottom-right: PACF, with lags 1-2 called out as the ones that matter.
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.vlines(lags, 0, p[1:], color=C1, linewidth=2.0)
    ax2.scatter(lags, p[1:], s=30, color=C1, zorder=5, edgecolor="white",
                linewidth=1.2)
    ax2.scatter([1, 2], [p[1], p[2]], s=70, facecolor="none", edgecolor=C2,
                linewidth=1.8, zorder=6)
    ax2.axhline(0, color=common.INK2, linewidth=0.9)
    ax2.fill_between([0.4, NL + 0.6], -band, band, color=C2, alpha=0.12,
                     linewidth=0, label=f"+/- 1.96/sqrt(n) = {band:.3f}")
    # Label the two significant lags to the RIGHT of their stems rather than
    # above/below: lag 1 and lag 2 are the largest values in the panel, so a
    # vertical offset pushes the text outside the axes or onto the tick labels.
    for k in (1, 2):
        ax2.annotate(f"lag {k}: {p[k]:+.3f}", xy=(k, p[k]), xytext=(9, 0),
                     textcoords="offset points", ha="left", va="center",
                     fontsize=8.5, color=common.INK)
    ax2.set_xlim(0.4, NL + 0.6)
    # Headroom so the lag-1 label and the legend do not fight for the top-right.
    pad = 0.10 * (p[1:].max() - p[1:].min())
    ax2.set_ylim(p[1:].min() - pad, p[1:].max() + 2.2 * pad)
    ax2.set_title("PACF: cuts off after lag 2  ->  AR(2)")
    ax2.set_xlabel("lag")
    ax2.set_ylabel("partial autocorrelation")
    ax2.legend(loc="upper right", fontsize=8)

    fig.suptitle("Problem 5 -- prediction stage: PACF cuts off, ACF decays",
                 fontsize=11.5, fontweight="bold", y=0.98)
    common.finish(fig, "problem5_identify")

    return {"acf": a, "pacf": p, "band": band, "n": n}


# --------------------------------------------------------------------------
# (c) + (d) fit the six models, compare AICc
# --------------------------------------------------------------------------
def part_cd(x: np.ndarray) -> dict:
    n = x.size
    specs = [("AR(1)", (1, 0, 0)), ("AR(2)", (2, 0, 0)), ("AR(3)", (3, 0, 0)),
             ("MA(1)", (0, 0, 1)), ("MA(2)", (0, 0, 2)), ("MA(3)", (0, 0, 3))]

    common.header("Problem 5(c) -- AICc for the six candidate models")
    print("  All six are fitted with an intercept ('c' trend), by exact MLE on")
    print("  the levels. k counts every estimated parameter: the intercept, the")
    print("  AR/MA coefficients, and sigma^2. So AR(2) has k = 4, not 2.")
    print()

    fits, entries = {}, {}
    for name, order in specs:
        res = ARIMA(x, order=order, trend="c").fit()
        k = len(res.params)          # includes const and sigma2
        fits[name] = res
        entries[name] = (float(res.llf), k)

    tbl = common.aicc_table(entries, n)
    # Attach R^2 for part (e). R^2 here = 1 - SSR/SST on the in-sample one-step
    # residuals, which is the natural analogue for a time-series fit.
    sst = float(np.sum((x - x.mean()) ** 2))
    tbl["R2"] = [1 - float(np.sum(fits[m].resid ** 2)) / sst
                 for m in tbl["model"]]
    tbl["sigma2"] = [float(fits[m].params[-1]) for m in tbl["model"]]

    # R^2 needs more decimals than everything else in this table: the whole point
    # of part (e) is that AR(2) -> AR(3) moves it by ~2e-4, which rounds away
    # entirely at 4dp and would make the "R^2 always rises" claim unverifiable
    # from the printed output.
    print(tbl.to_string(index=False, formatters={
        "loglik": lambda v: f"{v:.4f}",
        "AICc": lambda v: f"{v:.4f}",
        "dAICc": lambda v: f"{v:.4f}",
        "R2": lambda v: f"{v:.6f}",
        "sigma2": lambda v: f"{v:.6f}",
    }))
    print()
    print("  Fitted coefficients of each model:")
    for name, _ in specs:
        res = fits[name]
        ps = ", ".join(f"{nm}={v:+.4f}"
                       for nm, v in zip(res.param_names, res.params))
        print(f"    {name}: {ps}")

    best = tbl.loc[0, "model"]
    common.header("Problem 5(d) -- what AICc selects")
    print(f"  AICc selects {best} (AICc = {tbl.loc[0, 'AICc']:.3f}).")
    print(f"  Runner-up: {tbl.loc[1, 'model']} at dAICc = "
          f"{tbl.loc[1, 'dAICc']:.3f}.")
    print(f"  The prediction from part (a) was AR(2). It MATCHED.")
    print()
    print(f"  Reading the margins, since 'selected' alone is not informative:")
    print(f"    * every AR beats the MA of the same order, which is the ACF/PACF")
    print(f"      diagnosis showing up again in the likelihood;")
    print(f"    * MA(3) gets close (dAICc = "
          f"{float(tbl.loc[tbl['model'] == 'MA(3)', 'dAICc'].iloc[0]):.2f}) -- expected, since a")
    print(f"      stationary AR(2) equals an MA(infinity), and three MA terms is")
    print(f"      enough to approximate it. It needs one more parameter than")
    print(f"      AR(2) to do a slightly worse job, which is exactly the kind of")
    print(f"      trade AICc is built to price.")
    print(f"    * AR(1) is far behind (dAICc = "
          f"{float(tbl.loc[tbl['model'] == 'AR(1)', 'dAICc'].iloc[0]):.1f}): it cannot produce the")
    print(f"      negative lag-2 partial autocorrelation at all.")

    return {"fits": fits, "table": tbl}


# --------------------------------------------------------------------------
# (e) AR(2) vs AR(3): what AICc is pricing, and why R^2 would not
# --------------------------------------------------------------------------
def part_e(x: np.ndarray, F: dict) -> None:
    n = x.size
    ar2, ar3 = F["fits"]["AR(2)"], F["fits"]["AR(3)"]
    tbl = F["table"].set_index("model")

    common.header("Problem 5(e) -- AR(2) against AR(3)")
    print(f"  {'':<14}{'AR(2)':>14}{'AR(3)':>14}")
    print("  " + "-" * 42)
    print(f"  {'log-lik':<14}{ar2.llf:>14.4f}{ar3.llf:>14.4f}")
    print(f"  {'k':<14}{len(ar2.params):>14d}{len(ar3.params):>14d}")
    print(f"  {'AICc':<14}{tbl.loc['AR(2)', 'AICc']:>14.4f}"
          f"{tbl.loc['AR(3)', 'AICc']:>14.4f}")
    print(f"  {'R^2':<14}{tbl.loc['AR(2)', 'R2']:>14.6f}"
          f"{tbl.loc['AR(3)', 'R2']:>14.6f}")
    print(f"  {'sigma^2':<14}{tbl.loc['AR(2)', 'sigma2']:>14.6f}"
          f"{tbl.loc['AR(3)', 'sigma2']:>14.6f}")
    print("  " + "-" * 42)

    i3 = list(ar3.param_names).index("ar.L3")
    phi3 = ar3.params[i3]
    se3 = ar3.bse[i3]
    print(f"  The third coefficient: phi3 = {phi3:+.4f} with se {se3:.4f}")
    print(f"    t = {phi3 / se3:+.3f}, p = "
          f"{2 * (1 - stats.norm.cdf(abs(phi3 / se3))):.3f} -- indistinguishable")
    print(f"    from zero. Its 95% CI, [{phi3 - 1.96 * se3:+.3f}, "
          f"{phi3 + 1.96 * se3:+.3f}], covers 0 comfortably.")
    print()
    dll = ar3.llf - ar2.llf
    print(f"  WHAT AICc IS TRADING OFF:")
    print(f"    AICc = -2*loglik + 2k + 2k(k+1)/(n-k-1).")
    print(f"    Going AR(2) -> AR(3) buys dloglik = {dll:.4f}, so the fit term")
    print(f"    -2*loglik improves by only {2 * dll:.4f}.")
    print(f"    The penalty term rises by "
          f"{(2 * 5 + 2 * 5 * 6 / (n - 5 - 1)) - (2 * 4 + 2 * 4 * 5 / (n - 4 - 1)):.4f}")
    print(f"    (k goes 4 -> 5). The penalty wins, so AICc RISES by "
          f"{tbl.loc['AR(3)', 'AICc'] - tbl.loc['AR(2)', 'AICc']:.4f}")
    print(f"    and the larger model is rejected.")
    print(f"    The rule of thumb behind the arithmetic: one extra parameter has")
    print(f"    to earn about 1 unit of log-likelihood to pay for itself. This")
    print(f"    one earned {dll:.4f}. It is not close.")
    print()
    print(f"  WHY R^2 WOULD NOT MAKE THE SAME CHOICE (Week 02 s5.1-5.3):")
    print(f"    R^2 rises from {tbl.loc['AR(2)', 'R2']:.6f} to "
          f"{tbl.loc['AR(3)', 'R2']:.6f} -- a gain of")
    print(f"    {tbl.loc['AR(3)', 'R2'] - tbl.loc['AR(2)', 'R2']:.2e}. Tiny, but POSITIVE, and that is the point:")
    print(f"    R^2 is a monotone function of the residual sum of squares, and")
    print(f"    adding a regressor to a least-squares fit can never increase SSR.")
    print(f"    So R^2 is weakly increasing in model size BY CONSTRUCTION. It has")
    print(f"    no term that charges for a parameter, so it cannot express")
    print(f"    'this improvement is too small to be worth a parameter' -- it")
    print(f"    would pick AR(3) over AR(2), and AR(50) over AR(3).")
    print(f"    Week 02 s5.1 states the defect directly: 'The biggest problem")
    print(f"    with R^2 is that as variables are added, it continuously")
    print(f"    increases, regardless if they are explanatory', and 'Adding a")
    print(f"    column of random numbers to any regression raises the R^2.' An")
    print(f"    R^2 of 1.0 reached that way 'describes the sample perfectly and")
    print(f"    forecasts nothing at all'.")
    print(f"    R^2 measures in-sample fit; AICc estimates out-of-sample")
    print(f"    predictive loss. Those are different objectives, and only the")
    print(f"    second one can decline a free improvement in fit.")
    print(f"    Adjusted R^2 (s5.2) does add a penalty -- the (n-1)/(n-p-1)")
    print(f"    factor -- but a much weaker one, and it is defined for nested")
    print(f"    least-squares models rather than for comparing an AR against an")
    print(f"    MA. AICc is the criterion s5.3 offers for that job, and s5.3 also")
    print(f"    supplies the reason to prefer it over plain AIC here: 'In small")
    print(f"    sample sizes, AIC can tend to select models with higher numbers")
    print(f"    of parameters and overfitting of the model, in the same way R^2")
    print(f"    does.'")
    print(f"    One caution from s5.3 on reading the table above: 'The absolute")
    print(f"    value of AIC or BIC means nothing. Only differences between")
    print(f"    models fit to the same data are interpretable.' Hence dAICc.")

    # Residual diagnostics on the selected model: if AR(2) is right, its
    # residuals must be white. Reporting this is what makes "selected" credible.
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(ar2.resid, lags=[5, 10, 15], model_df=2)
    ra10 = float(acf(np.asarray(ar2.resid), nlags=10)[10])
    print()
    print(f"  Residual check on the selected AR(2) -- Ljung-Box")
    print(f"  (model_df=2, so the dof are corrected for the two AR terms):")
    print(lb.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"  Lags 5 and 15 are comfortably insignificant. Lag 10 comes in at")
    print(f"  p = {float(lb['lb_pvalue'].iloc[1]):.3f}, marginally below 0.05, and I am reporting that")
    print(f"  rather than rounding it away. It traces to the lag-10 ACF spike")
    print(f"  noted in part (a) ({ra10:+.3f}, just outside the band). Two reasons not to")
    print(f"  chase it: testing three lag-windows at 5% makes one marginal")
    print(f"  rejection unsurprising, and no AR/MA order in the candidate set")
    print(f"  fixes a lone lag-10 spike without adding many unpriced parameters.")
    print(f"  Read as: AR(2) is adequate for lags 1-5, where the structure is,")
    print(f"  with a marginal residual at lag 10 that I am not modelling.")

    # ---------------- figure ----------------
    # Quantities for the AR(2)->AR(3) callout in panel 2.
    dll_e = ar3.llf - ar2.llf
    pen_e = ((2 * 5 + 2 * 5 * 6 / (n - 5 - 1))
             - (2 * 4 + 2 * 4 * 5 / (n - 4 - 1)))
    net_e = tbl.loc["AR(3)", "AICc"] - tbl.loc["AR(2)", "AICc"]
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.9))
    order = ["AR(1)", "AR(2)", "AR(3)", "MA(1)", "MA(2)", "MA(3)"]
    t2 = F["table"].set_index("model").loc[order]

    # Panel 1: dAICc by model. Lower is better; the winner is called out.
    cols = [C2 if m == "AR(2)" else C1 for m in order]
    bars = ax[0].bar(order, t2["dAICc"], color=cols, width=0.62)
    for b, v, m in zip(bars, t2["dAICc"], order):
        # AR(2) is the reference model, so its bar has zero height and no label
        # would be visible at the bar top. Its marker is drawn well above the
        # dAICc = 2 rule with a leader down to the axis, which keeps it clear of
        # both that rule and the neighbouring AR(3) label.
        if m == "AR(2)":
            ax[0].annotate("AR(2): dAICc 0.0\nSELECTED",
                           xy=(b.get_x() + b.get_width() / 2, 0),
                           xytext=(0, 34), textcoords="offset points",
                           ha="center", fontsize=8.5, color=C2,
                           fontweight="bold",
                           arrowprops=dict(arrowstyle="-", color=C2,
                                           linewidth=1.0, shrinkA=2,
                                           shrinkB=1))
        else:
            ax[0].annotate(f"{v:.1f}",
                           xy=(b.get_x() + b.get_width() / 2, v),
                           xytext=(0, 5), textcoords="offset points",
                           ha="center", fontsize=8.5, color=common.INK)
    ax[0].axhline(2, color=common.INK2, linestyle=":", linewidth=1.0)
    # Park this label high and left, where no bar reaches, instead of on the
    # reference line itself -- at y=2 it lands on the AR(3)/MA(3) bar labels.
    ax[0].annotate("dotted line: dAICc = 2, the rough\n'not distinguishable' threshold",
                   xy=(0.30, 0.80), xycoords="axes fraction", fontsize=7.5,
                   color=common.INK2, ha="left")
    ax[0].set_ylim(0, max(t2["dAICc"]) * 1.12)
    ax[0].set_title("dAICc from the best model")
    ax[0].set_xlabel("model")
    ax[0].set_ylabel("dAICc  (lower is better)")

    # Panel 2: the AICc trade-off, fit term vs penalty term. Two series, so a
    # legend is present and both are direct-labelled.
    ks = t2["k"].to_numpy()
    fit_term = -2 * t2["loglik"].to_numpy()
    pen_term = 2 * ks + 2 * ks * (ks + 1) / (n - ks - 1)
    # Markers only, NOT a line: the x axis is six distinct models, not a
    # continuum, so a connecting segment from AR(3) to MA(1) would assert a
    # relationship that does not exist. A light vertical rule separates the
    # two families instead.
    xs2 = np.arange(6)
    ax[1].scatter(xs2, fit_term - fit_term.min(), s=70, color=C1, zorder=5,
                  edgecolor="white", linewidth=1.3,
                  label="fit term  -2*loglik  (rebased)")
    ax[1].scatter(xs2, pen_term - pen_term.min(), s=70, color=C2, marker="s",
                  zorder=5, edgecolor="white", linewidth=1.3,
                  label="penalty term  (rebased)")
    # Drop a thin stem to the axis so each model's pair reads as one column.
    for xi, fv, pv in zip(xs2, fit_term - fit_term.min(),
                          pen_term - pen_term.min()):
        ax[1].vlines(xi, 0, max(fv, pv), color=common.GRID, linewidth=1.0,
                     zorder=1)
    ax[1].axvline(2.5, color=common.INK2, linewidth=0.8, linestyle=":")
    ax[1].annotate("AR family", xy=(1.0, 0.96), xycoords=("data", "axes fraction"),
                   fontsize=8, color=common.INK2, ha="center")
    ax[1].annotate("MA family", xy=(4.0, 0.96), xycoords=("data", "axes fraction"),
                   fontsize=8, color=common.INK2, ha="center")
    # Call out the one comparison the panel exists to make.
    ax[1].annotate(f"AR(2) -> AR(3): fit improves {2 * dll_e:.2f},\n"
                   f"penalty costs {pen_e:.2f}  ->  net +{net_e:.2f}",
                   xy=(0.5, 0.55), xycoords="axes fraction", fontsize=7.5,
                   color=common.INK, ha="center",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                             edgecolor=common.GRID, linewidth=0.7))
    ax[1].set_xticks(xs2)
    ax[1].set_xticklabels(order)
    ax[1].set_xlim(-0.5, 5.5)
    ax[1].set_title("AICc = fit term + penalty term")
    ax[1].set_xlabel("model")
    ax[1].set_ylabel("units of AICc, rebased to min")
    ax[1].legend(loc="upper center", fontsize=8,
                 bbox_to_anchor=(0.5, 0.92))

    # Panel 3: ACF of the AR(2) residuals -- the adequacy check.
    r = np.asarray(ar2.resid)
    ra = acf(r, nlags=15)
    band = 1.96 / np.sqrt(n)
    lags = np.arange(1, 16)
    ax[2].vlines(lags, 0, ra[1:], color=C1, linewidth=2.0)
    ax[2].scatter(lags, ra[1:], s=28, color=C1, zorder=5, edgecolor="white",
                  linewidth=1.2)
    ax[2].axhline(0, color=common.INK2, linewidth=0.9)
    ax[2].fill_between([0.4, 15.6], -band, band, color=C2, alpha=0.12,
                       linewidth=0, label=f"+/- {band:.3f}")
    ax[2].set_xlim(0.4, 15.6)
    ax[2].scatter([10], [ra[10]], s=70, facecolor="none", edgecolor=C2,
                  linewidth=1.7, zorder=6)
    ax[2].annotate(f"lag 10: {ra[10]:+.3f}", xy=(10, ra[10]), xytext=(-9, 7),
                   textcoords="offset points", fontsize=7.5, color=C2,
                   ha="right")
    # Headroom so the lag-1 stem and the lag-10 callout both clear the frame.
    rpad = 0.12 * (ra[1:].max() - ra[1:].min())
    ax[2].set_ylim(ra[1:].min() - rpad, ra[1:].max() + 2.2 * rpad)
    ax[2].set_title("AR(2) residual ACF")
    ax[2].set_xlabel("lag")
    ax[2].set_ylabel("autocorrelation")
    ax[2].legend(loc="lower right", fontsize=8)

    fig.suptitle("Problem 5 -- AICc prices the third coefficient and declines it",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem5_fit")


def main() -> dict:
    x = common.load("problem5")["x"].to_numpy()
    ident = part_ab(x)
    F = part_cd(x)
    part_e(x, F)
    return {"ident": ident, "fits": F}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

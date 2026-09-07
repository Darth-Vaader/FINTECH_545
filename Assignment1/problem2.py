"""
Problem 2 -- A regression whose errors are not Normal.

problem2.csv holds x and y. We
  (a) plot y against x and predict the error distribution,
  (b) name the violated OLS assumption and predict its effect on beta and on
      se(beta),
  (fit) estimate OLS, MLE-Normal, and MLE-t, choose by AICc,
  (c) compare the three betas,
  (d) say what the rejected model gets wrong if not the slope,
  (e) compare the 95% and 99.5% quantiles of the two fitted error laws.

Run:  python problem2.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import optimize, stats

import common
from common import C1, C2, C3


# --------------------------------------------------------------------------
# (a) + (b) prediction: look at the scatter, then commit
# --------------------------------------------------------------------------
def part_ab(x: np.ndarray, y: np.ndarray) -> None:
    common.header("Problem 2(a,b) -- what the scatter shows before fitting")

    # A cheap, fit-free read on the error law: take out a ROBUST line (Theil-Sen
    # is not pulled around by the very outliers we are trying to detect) and look
    # at the moments of what is left. Using OLS residuals here would be mildly
    # circular, since OLS is the estimator whose assumptions we are testing.
    ts = stats.theilslopes(y, x)
    rob_resid = y - (ts.intercept + ts.slope * x)
    m = common.moments(rob_resid)
    print(f"  Robust (Theil-Sen) line, used only as a fit-free probe:")
    print(f"    slope = {ts.slope:.4f}, intercept = {ts.intercept:.4f}")
    print(f"  Moments of the residuals about that robust line:")
    print(f"    sd              = {m['sd']:.4f}")
    print(f"    skewness        = {m['skew']:+.4f}  "
          f"({m['skew'] / m['se_skew']:+.1f} se from 0)")
    print(f"    excess kurtosis = {m['exkurt']:+.4f}  "
          f"({m['exkurt'] / m['se_exkurt']:+.1f} se from 0)")
    print(f"  Roughly symmetric, strongly leptokurtic -> a Student t error is the")
    print(f"  natural guess. The kurtosis is what rules out the Normal.")

    # Turn that kurtosis into an actual PREDICTION for nu, rather than gesturing
    # at "low degrees of freedom". A t_nu has excess kurtosis 6/(nu-4) for nu>4,
    # so matching moments gives:
    nu_moment = 6.0 / m["exkurt"] + 4.0
    print(f"\n  Moment-matched prediction for nu: a t_nu has excess kurtosis")
    print(f"  6/(nu-4) for nu > 4, so setting 6/(nu-4) = {m['exkurt']:.4f} gives")
    print(f"    nu ~= {nu_moment:.2f}")
    print(f"  CAVEAT worth recording before the fit: that formula requires nu > 4")
    print(f"  for the population 4th moment to exist at all. If the true nu is")
    print(f"  below 4, the sample kurtosis is not estimating a finite quantity")
    print(f"  and this estimate is unreliable -- biased DOWNWARD in tail weight,")
    print(f"  i.e. it will over-state nu. So {nu_moment:.1f} should be read as an upper")
    print(f"  bound on nu rather than a point prediction.")

    # Serial correlation, so the claim that the OTHER assumptions hold is backed
    # by a number rather than asserted.
    dw = float(sm.stats.stattools.durbin_watson(
        sm.OLS(y, sm.add_constant(x)).fit().resid))
    print(f"\n  Durbin-Watson on the OLS residuals = {dw:.4f} (2.0 = no serial")
    print(f"  correlation), so the independence assumption looks intact and")
    print(f"  normality is the assumption in question.")
    print()
    print(f"  In Week 02 s3.1's numbering, the violated assumption is #7, 'the")
    print(f"  error term is normally distributed'. The slide adds the exact")
    print(f"  caveat that matters here: assumption 7 'can be broken, however, if")
    print(f"  assumed, then hypothesis testing along with confidence and")
    print(f"  prediction intervals can be calculated.' So breaking 7 costs us the")
    print(f"  INTERVALS, not the point estimate -- which is the prediction below.")
    print(f"  Checking the others: #1 linearity holds (no curvature in the")
    print(f"  scatter); #4 is the DW statistic above; #5 constant variance shows")
    print(f"  no fanning; #6 cannot fail with a single regressor.")

    # ---------------- figure ----------------
    common.use_style()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax[0].scatter(x, y, s=16, color=C1, alpha=0.75, edgecolor="none")
    xs = np.linspace(x.min(), x.max(), 100)
    ax[0].plot(xs, ts.intercept + ts.slope * xs, color=C2,
               label=f"robust line (slope {ts.slope:.2f})")
    # Flag the points that a Normal error would essentially forbid: |resid| > 3sd.
    far = np.abs(rob_resid) > 3 * m["sd"]
    ax[0].scatter(x[far], y[far], s=52, facecolor="none", edgecolor=C2,
                  linewidth=1.6, label=f"|resid| > 3 sd  (n={far.sum()})")
    ax[0].set_title("Strong linear trend with a handful of far-out points")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(loc="upper left")

    ax[1].hist(rob_resid, bins=35, density=True, color=C1, alpha=0.55,
               edgecolor="white", linewidth=0.5, label="residuals")
    grid = np.linspace(rob_resid.min() - 1, rob_resid.max() + 1, 500)
    ax[1].plot(grid, stats.norm.pdf(grid, 0, m["sd"]), color=C2,
               label="Normal, matched sd")
    dfe, loce, sce = stats.t.fit(rob_resid)
    ax[1].plot(grid, stats.t.pdf(grid, dfe, loce, sce), color=C3,
               linestyle="--", label=f"Student t (nu={dfe:.1f})")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-3, None)
    ax[1].set_title(f"Residual density (log): exkurt = {m['exkurt']:+.2f}")
    ax[1].set_xlabel("residual")
    ax[1].set_ylabel("density (log)")
    ax[1].legend(loc="lower center")

    fig.suptitle("Problem 2 -- prediction stage: the error term is fat-tailed",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem2_predict")


# --------------------------------------------------------------------------
# Numerical Hessian, for MLE standard errors
# --------------------------------------------------------------------------
def _num_hessian(f, p, h: float = 1e-5) -> np.ndarray:
    """Central-difference Hessian of the NEGATIVE log-likelihood at p.

    Inverting this gives the observed-information covariance matrix, which is
    how the MLE standard errors below are produced. We do it by hand rather than
    leaning on a library so the source of every reported se is explicit.
    """
    p = np.asarray(p, dtype=float)
    k = p.size
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ei = np.zeros(k); ei[i] = h
            ej = np.zeros(k); ej[j] = h
            H[i, j] = (f(p + ei + ej) - f(p + ei - ej)
                       - f(p - ei + ej) + f(p - ei - ej)) / (4 * h * h)
    return H


# --------------------------------------------------------------------------
# Fit: OLS, MLE-Normal, MLE-t
# --------------------------------------------------------------------------
def fit_models(x: np.ndarray, y: np.ndarray) -> dict:
    n = y.size
    X = sm.add_constant(x)

    # ---- Model 1: OLS -----------------------------------------------------
    ols = sm.OLS(y, X).fit()
    a_ols, b_ols = ols.params
    # NOTE the convention: sigma_hat here is the UNBIASED regression estimate,
    # s^2 = SSR/(n-2), which is what statsmodels reports and what the OLS
    # standard errors are built from. The MLE below divides by n instead, which
    # is why the two sigmas differ slightly while the slopes are identical.
    sigma_ols = np.sqrt(ols.mse_resid)
    ll_ols = float(ols.llf)

    common.header("Problem 2 -- Model 1: OLS")
    print(f"  alpha = {a_ols:.4f}   (se {ols.bse[0]:.4f})")
    print(f"  beta  = {b_ols:.4f}   (se {ols.bse[1]:.4f})")
    print(f"  sigma = {sigma_ols:.4f}   [s^2 = SSR/(n-2), the unbiased form]")
    print(f"  R^2   = {ols.rsquared:.4f}")
    print(f"  95% CI for beta: [{ols.conf_int()[1][0]:.4f}, "
          f"{ols.conf_int()[1][1]:.4f}]")
    # Sandwich (heteroskedasticity-robust) standard errors. Reported because they
    # are the reflex response to "my standard errors look wrong", and it is worth
    # being precise about what they do and do not repair here.
    rob = sm.OLS(y, X).fit(cov_type="HC3")
    bp = sm.stats.diagnostic.het_breuschpagan(ols.resid, X)
    print(f"\n  HC3 robust standard errors, for comparison:")
    print(f"    se(beta): {ols.bse[1]:.4f} (conventional) -> {rob.bse[1]:.4f} (HC3),")
    print(f"    a {100 * (1 - rob.bse[1] / ols.bse[1]):.0f}% reduction.")
    print(f"  That is a substantial change, and it lands close to the t-MLE's")
    print(f"  se(beta) reported below. But the REASON is not heteroskedasticity:")
    print(f"    Breusch-Pagan on this data: LM = {bp[0]:.3f}, p = {bp[1]:.4f}")
    print(f"    -> assumption #5 (constant variance) is NOT rejected.")
    e_ = ols.resid
    big = np.abs(e_) > 3 * e_.std(ddof=2)
    print(f"  The mechanism is WHERE the outliers sit. A sandwich estimator")
    print(f"  weights squared residuals by leverage, and this sample's "
          f"{int(big.sum())} largest")
    print(f"  residuals fall at CENTRAL x values (mean |x - xbar| = "
          f"{np.abs(x[big] - x.mean()).mean():.3f} against")
    print(f"  {np.abs(x - x.mean()).mean():.3f} overall), i.e. at LOW leverage. So HC3 sees little")
    print(f"  error variance where the x-leverage is, and returns a smaller")
    print(f"  se(beta). That is a fact about this sample's geometry, not a")
    print(f"  correction of a variance trend.")
    print(f"  CRUCIALLY, HC3 leaves sigma-hat -- and therefore the fitted ERROR")
    print(f"  DISTRIBUTION -- untouched; it re-weights the covariance of beta-hat")
    print(f"  only. The 99.5% error quantile stays the Normal one, so the tail")
    print(f"  understatement quantified in part (e) survives HC3 entirely.")
    print(f"  Robust standard errors fix inference on beta. They do not fix the")
    print(f"  density, which is where the risk number comes from.")

    # ---- Model 2: MLE under a Normal error -------------------------------
    def nll_normal(p):
        a, b, s = p
        if s <= 0:
            return 1e10
        return -np.sum(stats.norm.logpdf(y, a + b * x, s))

    r_n = optimize.minimize(nll_normal, [a_ols, b_ols, sigma_ols],
                            method="Nelder-Mead",
                            options=dict(xatol=1e-11, fatol=1e-11,
                                         maxiter=200_000, maxfev=200_000))
    a_n, b_n, s_n = r_n.x
    ll_n = -r_n.fun
    se_n = np.sqrt(np.diag(np.linalg.inv(_num_hessian(nll_normal, r_n.x))))

    common.header("Problem 2 -- Model 2: MLE, Normal error")
    print(f"  alpha = {a_n:.4f}   (se {se_n[0]:.4f})")
    print(f"  beta  = {b_n:.4f}   (se {se_n[1]:.4f})")
    print(f"  sigma = {s_n:.4f}   (se {se_n[2]:.4f})   "
          f"[MLE: divides by n, so slightly below the OLS s]")
    print(f"  log-likelihood = {ll_n:.4f}")
    print(f"  Note alpha and beta match OLS to 4 decimals. That is not luck:")
    print(f"  maximising a Normal likelihood IS minimising the sum of squares.")

    # ---- Model 3: MLE under a Student t error ----------------------------
    # Parameterisation: e = y - a - b*x, and e/scale ~ t(nu). So `scale` is NOT
    # the error sd; the implied sd is scale*sqrt(nu/(nu-2)), reported below.
    # nu is bounded below at 2.01 so that the error variance stays finite -- a
    # deliberate choice, since an infinite-variance error would make the
    # comparison against the Normal's sigma meaningless.
    def nll_t(p):
        a, b, s, nu = p
        if s <= 0 or nu <= 2.01:
            return 1e10
        z = (y - a - b * x) / s
        return -np.sum(stats.t.logpdf(z, nu) - np.log(s))

    r_t = optimize.minimize(nll_t, [a_ols, b_ols, sigma_ols * 0.7, 5.0],
                            method="Nelder-Mead",
                            options=dict(xatol=1e-11, fatol=1e-11,
                                         maxiter=400_000, maxfev=400_000))
    a_t, b_t, s_t, nu_t = r_t.x
    ll_t = -r_t.fun
    se_t = np.sqrt(np.diag(np.linalg.inv(_num_hessian(nll_t, r_t.x))))
    sd_implied = s_t * np.sqrt(nu_t / (nu_t - 2))

    common.header("Problem 2 -- Model 3: MLE, Student t error")
    print(f"  alpha = {a_t:.4f}   (se {se_t[0]:.4f})")
    print(f"  beta  = {b_t:.4f}   (se {se_t[1]:.4f})")
    print(f"  scale = {s_t:.4f}   (se {se_t[2]:.4f})")
    print(f"  nu    = {nu_t:.4f}   (se {se_t[3]:.4f})")
    print(f"  implied error sd = scale*sqrt(nu/(nu-2)) = {sd_implied:.4f}")
    print(f"  log-likelihood = {ll_t:.4f}")

    # ---- AICc comparison --------------------------------------------------
    # k counts every estimated parameter including the scale: Normal k=3,
    # t k=4. OLS and MLE-Normal are the same model, so OLS is listed with the
    # MLE-Normal likelihood and k=3 rather than being double-counted.
    tbl = common.aicc_table({
        "MLE Normal (= OLS), k=3": (ll_n, 3),
        "MLE Student t, k=4": (ll_t, 4),
    }, n)
    common.header("Problem 2 -- model choice by AICc")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  AICc prefers: {tbl.loc[0, 'model']}, by "
          f"{tbl.loc[1, 'dAICc']:.2f} AICc units.")
    print(f"  The t buys {ll_t - ll_n:.2f} log-likelihood points for one extra")
    print(f"  parameter. A likelihood-ratio test of nu -> infinity is on the")
    print(f"  boundary, but 2*dLL = {2 * (ll_t - ll_n):.1f} against a chi2(1)")
    print(f"  critical value of 3.84 is not a close call.")

    return {
        "n": n, "ols": ols,
        "normal": {"alpha": a_n, "beta": b_n, "sigma": s_n, "se": se_n,
                   "ll": ll_n},
        "t": {"alpha": a_t, "beta": b_t, "scale": s_t, "nu": nu_t,
              "se": se_t, "ll": ll_t, "sd_implied": sd_implied},
        "aicc": tbl,
    }


# --------------------------------------------------------------------------
# (c) + (d) + (e) reconcile
# --------------------------------------------------------------------------
def reconcile(x: np.ndarray, y: np.ndarray, F: dict) -> None:
    ols, N, T = F["ols"], F["normal"], F["t"]

    common.header("Problem 2(c) -- the three slope estimates")
    print(f"  {'model':<26}{'beta':>10}{'se(beta)':>11}{'width of 95% CI':>18}")
    print("  " + "-" * 65)
    for nm, b, se in [("OLS", ols.params[1], ols.bse[1]),
                      ("MLE Normal", N["beta"], N["se"][1]),
                      ("MLE Student t", T["beta"], T["se"][1])]:
        print(f"  {nm:<26}{b:>10.4f}{se:>11.4f}{2 * 1.96 * se:>18.4f}")
    print("  " + "-" * 65)
    spread = max(ols.params[1], N["beta"], T["beta"]) - \
        min(ols.params[1], N["beta"], T["beta"])
    print(f"  Spread across the three betas = {spread:.4f}, which is "
          f"{spread / ols.bse[1]:.2f} OLS standard errors.")
    print(f"  se(beta) falls from {ols.bse[1]:.4f} (OLS) to {T['se'][1]:.4f} (t),")
    print(f"  a {100 * (1 - T['se'][1] / ols.bse[1]):.1f}% reduction: the t")
    print(f"  down-weights the outliers instead of letting them inflate s^2.")

    common.header("Problem 2(d) -- what the rejected model gets wrong")
    print(f"  Not the slope. The Normal recovers beta = {N['beta']:.4f} against")
    print(f"  the t's {T['beta']:.4f}. What it gets wrong is the DENSITY:")
    print(f"    * it must set sigma = {N['sigma']:.4f} to cover the outliers,")
    print(f"      which inflates the assumed spread of every ordinary point;")
    print(f"    * per-observation log-likelihood is {N['ll'] / F['n']:.4f} under")
    print(f"      the Normal vs {T['ll'] / F['n']:.4f} under the t.")
    # Where does the likelihood gain actually come from? Almost entirely from
    # the handful of large residuals -- worth showing, since it is the answer.
    e = y - T["alpha"] - T["beta"] * x
    ll_i_n = stats.norm.logpdf(e, 0, N["sigma"])
    ll_i_t = stats.t.logpdf(e / T["scale"], T["nu"]) - np.log(T["scale"])
    gain = ll_i_t - ll_i_n
    order = np.argsort(-np.abs(e))
    top10 = gain[order[:10]].sum()
    print(f"\n  Decomposing the {T['ll'] - N['ll']:.2f} log-likelihood gain by")
    print(f"  observation: the 10 largest-|residual| points contribute "
          f"{top10:.2f}")
    print(f"  of it ({100 * top10 / (T['ll'] - N['ll']):.0f}%). The Normal's error")
    print(f"  is concentrated exactly where risk management cares.")

    common.header("Problem 2(e) -- quantiles of the two fitted error laws")
    print(f"  {'p':>8}{'Normal':>12}{'Student t':>12}{'t / Normal':>13}"
          f"{'  wider'}")
    print("  " + "-" * 60)
    rows = []
    for p in [0.90, 0.95, 0.975, 0.99, 0.995]:
        qn = stats.norm.ppf(p, 0, N["sigma"])
        qt = T["scale"] * stats.t.ppf(p, T["nu"])
        rows.append((p, qn, qt))
        print(f"  {p:>8.3f}{qn:>12.4f}{qt:>12.4f}{qt / qn:>13.4f}"
              f"  {'t' if qt > qn else 'Normal'}")
    print("  " + "-" * 60)
    cross = optimize.brentq(
        lambda p: T["scale"] * stats.t.ppf(p, T["nu"])
        - stats.norm.ppf(p, 0, N["sigma"]), 0.90, 0.9999)
    print(f"  The two curves cross at p = {cross:.4f}.")
    print(f"  Below it the Normal is wider; above it the t is wider.")
    print(f"  Reason: both fits must explain the SAME total dispersion. The t")
    print(f"  puts its mass in the far tail, so to keep the variance right it")
    print(f"  must be TIGHTER through the body -- hence narrower at 95%. The")
    print(f"  Normal has no far tail available, so it pays for the outliers by")
    print(f"  widening the middle. For a capital buffer you want the t: buffers")
    print(f"  are set at 99%+ where the t is {rows[-1][2] / rows[-1][1]:.2f}x wider,")
    print(f"  and it is exactly the loss beyond that point that breaks a firm.")

    # ---------------- figure ----------------
    common.use_style()
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.9))

    # Panel 1: data with the two nearly-identical fitted lines.
    ax[0].scatter(x, y, s=16, color=C1, alpha=0.7, edgecolor="none",
                  label="data")
    xs = np.linspace(x.min(), x.max(), 100)
    ax[0].plot(xs, N["alpha"] + N["beta"] * xs, color=C2,
               label=f"Normal / OLS  (b={N['beta']:.3f})")
    ax[0].plot(xs, T["alpha"] + T["beta"] * xs, color=C3, linestyle="--",
               label=f"Student t     (b={T['beta']:.3f})")
    ax[0].set_title("The two fitted lines nearly coincide")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].legend(loc="upper left")

    # Panel 2: the fitted error densities, log scale -- this is the real gap.
    grid = np.linspace(-6.5, 6.5, 700)
    ax[1].hist(e, bins=35, density=True, color=C1, alpha=0.5,
               edgecolor="white", linewidth=0.5, label="residuals")
    ax[1].plot(grid, stats.norm.pdf(grid, 0, N["sigma"]), color=C2,
               label=f"Normal (sd={N['sigma']:.2f})")
    ax[1].plot(grid, stats.t.pdf(grid / T["scale"], T["nu"]) / T["scale"],
               color=C3, linestyle="--",
               label=f"t (nu={T['nu']:.2f})")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-4, 1.0)
    ax[1].set_title("Where the models differ: the error density")
    ax[1].set_xlabel("error")
    ax[1].set_ylabel("density (log)")
    ax[1].legend(loc="lower center")

    # Panel 3: the quantile ratio, with the crossover marked. One series, so no
    # legend box is needed -- the title names it.
    ps = np.linspace(0.80, 0.999, 400)
    qn = stats.norm.ppf(ps, 0, N["sigma"])
    qt = T["scale"] * stats.t.ppf(ps, T["nu"])
    ax[2].plot(ps, qt / qn, color=C1)
    ax[2].axhline(1.0, color=common.INK2, linewidth=0.9, linestyle=":")
    ax[2].axvline(cross, color=C2, linewidth=1.2, linestyle="--")
    ax[2].annotate(f"crossover p = {cross:.3f}", xy=(cross, 1.0),
                   xytext=(cross - 0.005, 1.22), fontsize=8.5,
                   color=C2, ha="right")
    for p in (0.95, 0.995):
        r = (T["scale"] * stats.t.ppf(p, T["nu"])) / \
            stats.norm.ppf(p, 0, N["sigma"])
        ax[2].scatter([p], [r], s=42, color=C1, zorder=5,
                      edgecolor="white", linewidth=1.5)
        # Label below the marker when the ratio is under 1, so the text does not
        # collide with the ratio = 1 reference line.
        ax[2].annotate(f"p={p}: {r:.2f}x", xy=(p, r),
                       xytext=(-6, 9 if r > 1 else -17),
                       textcoords="offset points", fontsize=8.5,
                       color=common.INK, ha="right" if r < 1 else "center")
    ax[2].set_title("t quantile / Normal quantile")
    ax[2].set_xlabel("quantile level p")
    ax[2].set_ylabel("ratio")

    fig.suptitle("Problem 2 -- same slope, different tails",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem2_fit")


def main() -> dict:
    d = common.load("problem2")
    x, y = d["x"].to_numpy(), d["y"].to_numpy()
    part_ab(x, y)
    F = fit_models(x, y)
    reconcile(x, y, F)
    return F


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

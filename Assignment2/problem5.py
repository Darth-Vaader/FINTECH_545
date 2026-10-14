"""
Problem 5 -- Model-based simulation and residual correlation.

problem5.csv holds 501 daily prices for a market index MKT and two stocks A and
B. Zero expected returns are assumed. Portfolios:
    P1 = long $1M A, long  $1M B
    P2 = long $1M A, short $1M B
Each stock follows r_i = alpha_i + beta_i r_MKT + eps_i with Normal market and
Normal errors. We

  predict  : with Fact 7 of Week 02, the sign of the error made by treating
             eps_A and eps_B as independent, for P1 and for P2;
  fit      : the two OLS regressions; 100,000 simulated days with the full
             residual covariance and with its off-diagonal zeroed; VaR of both
             portfolios both ways; a delta-normal VaR straight from the sample
             covariance of the two stocks;
  reconcile: direction and size of the shortcut's error; why the full-residual
             simulation and the delta-normal number agree; when they would not.

Conventions:
  * arithmetic returns P_t/P_{t-1} - 1 (Week 04 s4.1.2), so a portfolio's P&L
    is the holdings-weighted sum of returns;
  * "zero expected returns" means the simulation drops alpha and gives the
    market a zero mean; alpha is still reported from the fit;
  * every variance and covariance divides by n-1 (Week 01 s4.2), INCLUDING the
    residual covariance, so the OLS identity Sigma_stocks = beta beta' sigma_M^2
    + Sigma_eps holds exactly in sample. The OLS residual sd with n-2 is printed
    alongside for reference; it differs in the third decimal.

Run:  python problem5.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import common
from common import C1, C2, C3

UNIT = 1_000_000.0
ALPHA = 0.05
NSIM = 100_000
HOLD = {"P1 (long A, long B)": np.array([UNIT, UNIT]),
        "P2 (long A, short B)": np.array([UNIT, -UNIT])}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def prepare() -> dict:
    d = common.load("problem5")
    p = d[["MKT", "A", "B"]].to_numpy(dtype=float)
    r = pd.DataFrame(p[1:] / p[:-1] - 1.0, columns=["MKT", "A", "B"])
    return {"r": r, "n": len(r)}


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------
def part_a(D: dict) -> dict:
    r = D["r"]
    common.header("Problem 5(a) -- Predict: what the independent-errors shortcut does")
    print("  Fact 7 (Week 02 s2.1.1) for h_A r_A + h_B r_B:")
    print("    Var = h_A^2 Var(r_A) + h_B^2 Var(r_B) + 2 h_A h_B Cov(r_A, r_B)")
    print("  and under the model, Cov(r_A, r_B) = beta_A beta_B Var(r_M) + Cov(eps_A, eps_B).")
    print("  The shortcut sets Cov(eps_A, eps_B) = 0. If the stocks share an industry the")
    print("  market misses, the true Cov(eps_A, eps_B) > 0, so the shortcut drops a")
    print("  POSITIVE term from the cross-covariance:")
    print("    P1: h_A h_B = +$1M^2  -> variance UNDERSTATED -> VaR too LOW")
    print("    P2: h_A h_B = -$1M^2  -> variance OVERSTATED  -> VaR too HIGH")
    print("  Variances are untouched, so the whole error is the one cross term.")
    # What the data say before any regression: a one-factor model with
    # independent errors implies corr(A, B) = corr(A, M) corr(B, M).
    c = r.corr()
    implied = c.loc["A", "MKT"] * c.loc["B", "MKT"]
    print(f"\n  A look at the raw correlations (no model fitted):")
    print(f"    corr(A, MKT) = {c.loc['A', 'MKT']:.4f}, corr(B, MKT) = {c.loc['B', 'MKT']:.4f}, "
          f"corr(A, B) = {c.loc['A', 'B']:.4f}")
    print(f"    With independent errors a one-factor model forces corr(A, B) =")
    print(f"    corr(A, M) x corr(B, M) = {implied:.4f}. The data show {c.loc['A', 'B']:.4f}: "
          f"the gap of {c.loc['A', 'B'] - implied:+.3f} is")
    print(f"    co-movement the market does not explain, so Cov(eps_A, eps_B) is positive")
    print(f"    and large. Expect the shortcut to miss by a lot, and P2 to suffer most:")
    print(f"    with beta_A ~ beta_B the market leg of P2 nearly cancels and what remains")
    print(f"    is almost entirely residual, where the dropped term lives.")
    return {"corr": c, "implied_corr": implied}


# --------------------------------------------------------------------------
# Fit (b): regressions
# --------------------------------------------------------------------------
def ols(y: np.ndarray, x: np.ndarray) -> dict:
    """y = alpha + beta x + eps by OLS (Week 02 s2.3), with conventional standard
    errors. Residual sd is reported two ways: n-1 (used everywhere else here so
    the covariance identity is exact) and the OLS n-2."""
    n = y.size
    X = np.column_stack([np.ones(n), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    s2 = resid @ resid / (n - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return {"alpha": coef[0], "beta": coef[1], "se_alpha": np.sqrt(cov[0, 0]),
            "se_beta": np.sqrt(cov[1, 1]), "resid": resid,
            "sd_resid": resid.std(ddof=1), "sd_resid_ols": np.sqrt(s2),
            "r2": 1 - resid.var(ddof=0) / y.var(ddof=0)}


def part_b(D: dict) -> dict:
    r = D["r"]
    common.header("Problem 5(b) -- the two regressions on the market")
    fits = {s: ols(r[s].to_numpy(), r["MKT"].to_numpy()) for s in ("A", "B")}
    print(f"  {'stock':<7}{'alpha':>11}{'(se)':>10}{'beta':>9}{'(se)':>9}{'resid sd':>11}"
          f"{'(n-2)':>10}{'R^2':>7}")
    print("  " + "-" * 74)
    for s, f in fits.items():
        print(f"  {s:<7}{f['alpha']:>+11.6f}{f['se_alpha']:>10.6f}{f['beta']:>9.4f}"
              f"{f['se_beta']:>9.4f}{f['sd_resid']:>11.6f}{f['sd_resid_ols']:>10.6f}"
              f"{f['r2']:>7.3f}")
    print("  " + "-" * 74)
    E = np.column_stack([fits["A"]["resid"], fits["B"]["resid"]])
    sig_eps = np.cov(E, rowvar=False, ddof=1)
    rho_eps = sig_eps[0, 1] / np.sqrt(sig_eps[0, 0] * sig_eps[1, 1])
    sd_m = r["MKT"].std(ddof=1)
    print(f"  market sd = {sd_m:.6f}   (mean {r['MKT'].mean():+.6f}, set to 0 in the "
          f"simulation)")
    print(f"  residual correlation corr(eps_A, eps_B) = {rho_eps:+.4f}")
    print(f"  residual covariance Sigma_eps (x 1e-4):")
    common.print_matrix(sig_eps * 1e4, ["eps_A", "eps_B"], fmt="{:>9.4f}")
    print(f"  Both alphas are within one standard error of zero, consistent with the")
    print(f"  zero-expected-return assumption. R^2 of ~0.42 leaves 58% of each stock's")
    print(f"  variance in the residual, and the residuals are {rho_eps:.2f} correlated: the")
    print(f"  shared industry the market does not capture, in one number.")
    return {"fits": fits, "sig_eps": sig_eps, "rho_eps": rho_eps, "sd_m": sd_m, "E": E}


# --------------------------------------------------------------------------
# Fit (c): simulate with and without the residual correlation
# --------------------------------------------------------------------------
def model_cov(F: dict, diagonal: bool) -> np.ndarray:
    """Implied covariance of (r_A, r_B): beta beta' sigma_M^2 + Sigma_eps, with
    Sigma_eps either full or with its off-diagonal zeroed (the shortcut)."""
    beta = np.array([F["fits"]["A"]["beta"], F["fits"]["B"]["beta"]])
    s_eps = np.diag(np.diag(F["sig_eps"])) if diagonal else F["sig_eps"]
    return np.outer(beta, beta) * F["sd_m"] ** 2 + s_eps


def part_c(F: dict) -> dict:
    common.header(f"Problem 5(c) -- {NSIM:,} simulated days, full vs diagonal residual "
                  f"covariance")
    beta = np.array([F["fits"]["A"]["beta"], F["fits"]["B"]["beta"]])
    rng = np.random.default_rng(common.SEED)
    z = rng.standard_normal((3, NSIM))          # shared draws: [market, eps_A, eps_B]
    out = {}
    print("  Joint vector [r_M, eps_A, eps_B] ~ N(0, blockdiag(sigma_M^2, Sigma_eps))")
    print("  (Week 05 s5.2.2); r_i = beta_i r_M + eps_i with alpha dropped. Same standard")
    print("  normal draws feed both runs, so the difference is the off-diagonal alone.")
    print(f"\n  {'portfolio':<24}{'Sigma_eps':>11}{'VaR 5%':>12}{'ES 5%':>12}"
          f"{'closed form':>13}{'MC se':>9}")
    print("  " + "-" * 81)
    for label, diagonal in (("full", False), ("diagonal", True)):
        s_eps = np.diag(np.diag(F["sig_eps"])) if diagonal else F["sig_eps"]
        cov = np.zeros((3, 3))
        cov[0, 0] = F["sd_m"] ** 2
        cov[1:, 1:] = s_eps
        L = np.linalg.cholesky(cov)
        draws = (L @ z).T                                   # NSIM x 3
        r_m, eps = draws[:, 0], draws[:, 1:]
        stocks = r_m[:, None] * beta[None, :] + eps         # NSIM x 2: r_A, r_B
        sig_model = model_cov(F, diagonal)
        for pname, h in HOLD.items():
            pnl = stocks @ h
            v, e = common.var_es(pnl, ALPHA)
            closed = common.var_normal(np.sqrt(h @ sig_model @ h), ALPHA)
            se = np.sqrt(ALPHA * (1 - ALPHA) / NSIM) / stats.norm.pdf(
                stats.norm.ppf(ALPHA)) * np.sqrt(h @ sig_model @ h)
            out[(pname, label)] = {"var": v, "es": e, "closed": closed, "se": se}
            print(f"  {pname:<24}{label:>11}{common.money(v):>12}{common.money(e):>12}"
                  f"{common.money(closed):>13}{common.money(se):>9}")
    print("  " + "-" * 81)
    print("  'closed form' is 1.645 sqrt(h' Sigma_model h) with the same Sigma_model the")
    print("  simulation was drawn from; 'MC se' is the asymptotic standard error of the")
    print("  simulated 5% quantile, sqrt(a(1-a)/N) / f(q). Every simulated VaR is within")
    print("  about one se of its closed form, so the Monte Carlo is doing what it should.")
    return out


# --------------------------------------------------------------------------
# Fit (d): delta normal from the sample covariance of the stocks
# --------------------------------------------------------------------------
def part_d(D: dict) -> dict:
    common.header("Problem 5(d) -- delta-normal VaR from the sample covariance of A and B")
    sig = D["r"][["A", "B"]].cov(ddof=1).to_numpy()
    out = {}
    print("  Sigma_sample (x 1e-4):")
    common.print_matrix(sig * 1e4, ["A", "B"], fmt="{:>9.4f}")
    print(f"\n  VaR = -Phi^-1(0.05) sqrt(h' Sigma h) with h in dollars (Week 04 s4.4.1; a stock")
    print(f"  has delta 1, so the gradient is just the holdings):")
    for pname, h in HOLD.items():
        sd = np.sqrt(h @ sig @ h)
        out[pname] = {"sd": sd, "var": common.var_normal(sd, ALPHA)}
        print(f"    {pname:<24} sd of P&L {common.money(sd):>10}   VaR {common.money(out[pname]['var']):>10}")
    return {"sig": sig, "var": out}


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------
def part_e(F: dict, S: dict) -> None:
    common.header("Problem 5(e) -- did the shortcut move each portfolio as predicted?")
    cov_full, cov_diag = model_cov(F, False), model_cov(F, True)
    print(f"  {'portfolio':<24}{'full VaR':>10}{'diag VaR':>10}{'VaR error':>11}"
          f"{'dropped/var':>13}{'var ratio':>11}")
    print("  " + "-" * 79)
    for pname, h in HOLD.items():
        vf, vd = S[(pname, "full")]["var"], S[(pname, "diagonal")]["var"]
        var_full, var_diag = h @ cov_full @ h, h @ cov_diag @ h
        cross = 2 * h[0] * h[1] * F["sig_eps"][0, 1]          # the dropped term
        print(f"  {pname:<24}{common.money(vf):>10}{common.money(vd):>10}"
              f"{100 * (vd - vf) / vf:>+10.1f}%{100 * cross / var_full:>+12.1f}%"
              f"{var_diag / var_full:>11.3f}")
    print("  " + "-" * 79)
    print("  'dropped/var' is the term the shortcut drops, 2 h_A h_B Cov(eps_A, eps_B), as")
    print("  a fraction of the portfolio's true variance; 'var ratio' is the shortcut's")
    print("  variance over the true one. P1: UNDERSTATED, as predicted. P2: OVERSTATED,")
    print("  as predicted. P2 is far more exposed: its")
    beta = np.array([F["fits"]["A"]["beta"], F["fits"]["B"]["beta"]])
    h2 = HOLD["P2 (long A, short B)"]
    mkt_part = (h2 @ beta) ** 2 * F["sd_m"] ** 2
    print(f"  market leg is (beta_A - beta_B)^2 sigma_M^2 = ({beta[0] - beta[1]:+.4f})^2 "
          f"sigma_M^2, only {100 * mkt_part / (h2 @ cov_full @ h2):.1f}% of its")
    print(f"  variance, so P2 is essentially a bet on the residual spread -- and the")
    print(f"  residual correlation of {F['rho_eps']:.2f} is what makes that spread narrow. Drop")
    print(f"  it and the model thinks the hedge does nothing. P1 is dominated by the")
    print(f"  market leg, which both runs share, so the missing term is a smaller slice.")


def part_f(D: dict, F: dict, S: dict, DN: dict) -> None:
    common.header("Problem 5(f) -- full-residual simulation vs delta normal")
    cov_full = model_cov(F, False)
    diff = np.abs(cov_full - DN["sig"]).max()
    print(f"  {'portfolio':<24}{'MC full':>11}{'closed form':>13}{'delta normal':>14}")
    print("  " + "-" * 62)
    for pname in HOLD:
        print(f"  {pname:<24}{common.money(S[(pname, 'full')]['var']):>11}"
              f"{common.money(S[(pname, 'full')]['closed']):>13}"
              f"{common.money(DN['var'][pname]['var']):>14}")
    print("  " + "-" * 62)
    print(f"  max |beta beta' sigma_M^2 + Sigma_eps - Sigma_sample| = {diff:.1e}")
    print(f"  The model covariance IS the sample covariance, to machine precision. OLS")
    print(f"  makes the residual orthogonal to the regressor in sample, so Fact 7 splits")
    print(f"  Cov(r_A, r_B) exactly into beta_A beta_B Var(r_M) + Cov(eps_A, eps_B) when")
    print(f"  every piece is estimated on the same rows with the same n-1. Week 05 s5.2.1:")
    print(f"  with everything normal, OLS 'is the same thing as just using a joint,")
    print(f"  multivariate distribution on Y and X' -- the regression was already inside")
    print(f"  the joint normal, so simulating its pieces and recombining them returns the")
    print(f"  joint normal we could have written down. The only gap left is Monte Carlo")
    print(f"  noise, about one standard error (part c).")
    print(f"\n  They stop agreeing when any of those conditions goes:")
    print(f"    * a non-normal piece -- t errors, a t market -- the sum is no longer")
    print(f"      normal and 1.645 sd is no longer its 5% quantile (this is where the")
    print(f"      Week 05 copula machinery comes in);")
    print(f"    * a non-linear position -- options on A and B -- delta normal linearises,")
    print(f"      the simulation reprices;")
    print(f"    * pieces estimated differently -- an EW market variance with equal-weight")
    print(f"      residuals, or regressions on different windows -- so the identity")
    print(f"      above no longer holds in sample;")
    print(f"    * the shortcut itself: a diagonal Sigma_eps, which is the whole of this")
    print(f"      problem;")
    print(f"    * means that matter -- at a one-year horizon dropping alpha is no longer")
    print(f"      harmless, and the absolute and relative VaR conventions separate.")


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def figure(D: dict, F: dict, S: dict, DN: dict) -> None:
    common.use_style()
    r = D["r"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    xs = np.linspace(r["MKT"].min(), r["MKT"].max(), 2)
    for s, col in (("A", C1), ("B", C2)):
        f = F["fits"][s]
        ax[0].scatter(r["MKT"], r[s], s=8, color=col, alpha=0.45, edgecolor="none",
                      label=f"{s}: beta {f['beta']:.3f}, R2 {f['r2']:.2f}")
        ax[0].plot(xs, f["alpha"] + f["beta"] * xs, color=col, linewidth=1.6)
    ax[0].set_xlabel("market return")
    ax[0].set_ylabel("stock return")
    ax[0].set_title("Each stock on the market")
    ax[0].legend(loc="upper left")

    ax[1].scatter(F["E"][:, 0], F["E"][:, 1], s=8, color=C3, alpha=0.5, edgecolor="none")
    lim = np.abs(F["E"]).max() * 1.05
    ax[1].plot([-lim, lim], [-lim, lim], color=common.INK2, linewidth=0.8, linestyle=":")
    ax[1].set_xlim(-lim, lim); ax[1].set_ylim(-lim, lim)
    ax[1].set_aspect("equal")
    ax[1].set_xlabel("eps_A")
    ax[1].set_ylabel("eps_B")
    ax[1].set_title(f"Residuals: corr = {F['rho_eps']:.3f}, the term the shortcut drops")

    labels = list(HOLD)
    xpos = np.arange(len(labels))
    w = 0.26
    full = [S[(p, "full")]["var"] / 1e3 for p in labels]
    diag = [S[(p, "diagonal")]["var"] / 1e3 for p in labels]
    dn = [DN["var"][p]["var"] / 1e3 for p in labels]
    ax[2].bar(xpos - w, full, w, color=C1, label="MC, full Sigma_eps")
    ax[2].bar(xpos, diag, w, color=C2, label="MC, diagonal Sigma_eps (shortcut)")
    ax[2].bar(xpos + w, dn, w, color=C3, label="delta normal, sample cov")
    for k in range(len(labels)):
        for off, val in ((-w, full[k]), (0, diag[k]), (w, dn[k])):
            ax[2].text(xpos[k] + off, val + 0.6, f"{val:.1f}", ha="center", fontsize=7.5)
    ax[2].set_xticks(xpos)
    ax[2].set_xticklabels(["P1: long A, long B", "P2: long A, short B"])
    ax[2].set_ylim(0, max(full + diag + dn) * 1.4)
    ax[2].set_ylabel("5% VaR ($ thousands)")
    ax[2].set_title("The shortcut lowers P1 and raises P2")
    ax[2].legend(loc="upper right", fontsize=7.5)
    fig.suptitle("Problem 5 -- two stocks on one market: what the residual correlation is worth",
                 fontsize=11.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    common.finish(fig, "problem5")


def main() -> dict:
    D = prepare()
    A = part_a(D)
    F = part_b(D)
    S = part_c(F)
    DN = part_d(D)
    part_e(F, S)
    part_f(D, F, S, DN)
    figure(D, F, S, DN)
    return {"D": D, "A": A, "F": F, "S": S, "DN": DN}


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    main()

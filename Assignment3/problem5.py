"""
Problem 5 -- Alpha or factor exposure?

problem5.csv: five years (1996-10-01 .. 2001-09-28) of daily decimal returns for
FUND1 and FUND2. The Fama-French daily factor files give Mkt-RF, SMB, HML, RF
and Mom in PERCENT -> divided by 100.

  predict  : (a) mean excess return, vol, Sharpe and CAPM for each fund;
             (b) correlation of each CAPM residual with SMB, HML and Mom, and a
             prediction of what adding them does to alpha.
  fit      : (c) CAPM, FF3 and Carhart-4 for each fund: coefficients with
             standard errors, alpha t, R^2, AICc; (d) expected annual return from
             the 4-factor betas and the most recent 10 years of factor means,
             with and without alpha; (e) the worst daily momentum return in the
             file applied to $10,000,000 through the momentum beta, as a multiple
             of each fund's daily sd.
  reconcile: (f) the alpha change explained by the factor premia in sample;
             (g) APT and which alpha is worth a fee; (h) (d) against the realised
             return and what it means for an optimizer's inputs.

Conventions:
  * excess return = fund return - RF on the same day (inner join on dates);
  * OLS with classical standard errors; AICc with k = number of coefficients
    + 1 for the error variance (common.ols);
  * annualisation: 255 trading days, the course convention (Week 03 onward;
    the Week 07 homework uses the same 255 for this calculation). Expected
    annual return = 255 x (alpha + sum_k beta_k E[f_k]) + annual RF, where E[f]
    and RF are the daily means over the most recent 10 years in the files;
  * "the most recent 10 years in the factor files" = the 10 years ending on the
    last date in the files (2014-10-01 .. 2024-09-30).

Run:  python problem5.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
from common import C1, C2, C3, C4, INK2

DPY = 255.0
FUNDS = ["FUND1", "FUND2"]
MODELS = {"CAPM": ["Mkt-RF"], "FF3": ["Mkt-RF", "SMB", "HML"],
          "Carhart4": ["Mkt-RF", "SMB", "HML", "Mom"]}
NOTIONAL = 10_000_000.0


def load_factors() -> pd.DataFrame:
    ff = common.load("F-F_Research_Data_Factors_daily.CSV")
    mom = common.load("F-F_Momentum_Factor_daily.CSV")
    mom.columns = [c.strip() for c in mom.columns]          # header is "Mom   "
    f = ff.merge(mom, on="Date", how="inner")
    f["Date"] = pd.to_datetime(f.Date.astype(str), format="%Y%m%d")
    for c in ["Mkt-RF", "SMB", "HML", "RF", "Mom"]:
        f[c] = f[c] / 100.0                                 # percent -> decimal
    return f


def prepare() -> dict:
    fac = load_factors()
    fu = common.load("problem5")
    fu["Date"] = pd.to_datetime(fu.Date)
    d = fu.merge(fac, on="Date", how="inner")
    for f in FUNDS:
        d[f + "_ex"] = d[f] - d.RF
    return {"fac": fac, "d": d, "n_fund": len(fu)}


def fit(d: pd.DataFrame, fund: str, model: str) -> dict:
    cols = MODELS[model]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy() for c in cols])
    return common.ols(d[fund + "_ex"].to_numpy(), X, ["alpha"] + cols)


# --------------------------------------------------------------------------
# (a) summary and CAPM
# --------------------------------------------------------------------------
def part_a(D: dict) -> pd.DataFrame:
    d = D["d"]
    rows = []
    for f in FUNDS:
        ex = d[f + "_ex"]
        c = fit(d, f, "CAPM")
        rows.append({"fund": f, "mean ex/day": ex.mean(), "ann mean ex": DPY * ex.mean(),
                     "sd/day": d[f].std(ddof=1), "ann vol": np.sqrt(DPY) * d[f].std(ddof=1),
                     "Sharpe (ann)": np.sqrt(DPY) * ex.mean() / ex.std(ddof=1),
                     "CAPM alpha/day": c["beta"][0], "alpha ann": DPY * c["beta"][0],
                     "alpha t": c["t"][0], "beta": c["beta"][1], "R^2": c["r2"]})
    m = d["Mkt-RF"]
    rows.append({"fund": "Mkt-RF", "mean ex/day": m.mean(), "ann mean ex": DPY * m.mean(),
                 "sd/day": m.std(ddof=1), "ann vol": np.sqrt(DPY) * m.std(ddof=1),
                 "Sharpe (ann)": np.sqrt(DPY) * m.mean() / m.std(ddof=1)})
    out = pd.DataFrame(rows).set_index("fund")
    common.header("Problem 5(a) - excess return, vol, Sharpe and CAPM")
    print(f"{len(d)} matched days {d.Date.min().date()} .. {d.Date.max().date()} "
          f"(fund file has {D['n_fund']})\n")
    common.show(out, {"mean ex/day": "{:.6f}", "ann mean ex": "{:+.4f}", "sd/day": "{:.5f}",
                      "ann vol": "{:.4f}", "Sharpe (ann)": "{:.3f}", "CAPM alpha/day": "{:+.6f}",
                      "alpha ann": "{:+.4f}", "alpha t": "{:+.2f}", "beta": "{:.3f}",
                      "R^2": "{:.3f}"})
    return out


# --------------------------------------------------------------------------
# (b) residual correlations
# --------------------------------------------------------------------------
def part_b(D: dict) -> pd.DataFrame:
    d = D["d"]
    rows = []
    for f in FUNDS:
        e = fit(d, f, "CAPM")["resid"]
        rows.append({"fund": f, **{k: np.corrcoef(e, d[k])[0, 1] for k in ("SMB", "HML", "Mom")}})
    out = pd.DataFrame(rows).set_index("fund")
    prem = pd.DataFrame({k: [d[k].mean(), DPY * d[k].mean(),
                             d[k].mean() / d[k].std(ddof=1) * np.sqrt(len(d))]
                         for k in ("Mkt-RF", "SMB", "HML", "Mom")},
                        index=["mean/day", "ann mean", "t (mean=0)"])
    common.header("Problem 5(b) - correlation of each fund's CAPM residual with the factors")
    common.show(out, "{:+.3f}")
    print(f"\n|rho| > 2/sqrt(n) = {2 / np.sqrt(len(d)):.3f} is significant at ~5%.")
    print("\nFactor premia in the fund sample (1996-10 .. 2001-09):")
    common.show(prem, "{:+.5f}")
    D["prem_sample"] = prem
    return out


# --------------------------------------------------------------------------
# (c) the three models
# --------------------------------------------------------------------------
def part_c(D: dict) -> dict:
    d = D["d"]
    fits = {(f, m): fit(d, f, m) for f in FUNDS for m in MODELS}
    common.header("Problem 5(c) - CAPM, FF3 and Carhart-4: coefficient (se)")
    for f in FUNDS:
        rows = []
        for m in MODELS:
            r = fits[(f, m)]
            rec = {"model": m}
            for nm in ["alpha", "Mkt-RF", "SMB", "HML", "Mom"]:
                if nm in r["names"]:
                    i = r["names"].index(nm)
                    scale = DPY if nm == "alpha" else 1.0
                    dp = 4 if nm == "alpha" else 3
                    rec[nm] = f"{scale * r['beta'][i]:+.{dp}f}({scale * r['se'][i]:.{dp}f})"
                else:
                    rec[nm] = ""
            rec["alpha t"] = f"{r['t'][0]:+.2f}"
            rec["R^2"] = f"{r['r2']:.4f}"
            rec["AICc"] = f"{r['aicc']:.1f}"
            rows.append(rec)
        print(f"\n{f}   (alpha annualised: x 255)")
        print(pd.DataFrame(rows).set_index("model").to_string())
        a = {m: fits[(f, m)]["aicc"] for m in MODELS}
        best = min(a, key=a.get)
        print(f"  AICc differences vs best ({best}): "
              + ", ".join(f"{m} {a[m] - a[best]:+.1f}" for m in MODELS))
    D["fits"] = fits
    return fits


# --------------------------------------------------------------------------
# (d) expected annual return from the 4-factor betas
# --------------------------------------------------------------------------
def part_d(D: dict) -> pd.DataFrame:
    fac, d, fits = D["fac"], D["d"], D["fits"]
    end = fac.Date.max()
    start = end - pd.DateOffset(years=10)
    recent = fac[fac.Date > start]
    cols = MODELS["Carhart4"]
    Ef = recent[cols].mean().to_numpy()
    rf_d = recent.RF.mean()
    rows = []
    for f in FUNDS:
        r = fits[(f, "Carhart4")]
        a, b = r["beta"][0], r["beta"][1:]
        ex_no_a = DPY * float(b @ Ef)
        rows.append({"fund": f, "E[excess] no a": ex_no_a,
                     "E[total] no a": ex_no_a + DPY * rf_d,
                     "alpha (ann)": DPY * a,
                     "E[total] with a": ex_no_a + DPY * (a + rf_d),
                     "realised total": DPY * d[f].mean(),
                     "realised excess": DPY * d[f + "_ex"].mean()})
    out = pd.DataFrame(rows).set_index("fund")
    fm = pd.DataFrame({"recent 10y mean/day": recent[cols + ["RF"]].mean(),
                       "recent 10y ann": DPY * recent[cols + ["RF"]].mean(),
                       "fund-sample ann": DPY * d[cols + ["RF"]].mean()})
    common.header("Problem 5(d) - expected annual return from the 4-factor betas")
    print(f"Recent window: {recent.Date.min().date()} .. {recent.Date.max().date()} "
          f"({len(recent)} days)\n")
    common.show(fm, {"recent 10y mean/day": "{:+.6f}", "recent 10y ann": "{:+.4f}",
                     "fund-sample ann": "{:+.4f}"})
    print()
    common.show(out, "{:+.4f}")
    # Contribution of each factor
    contrib = pd.DataFrame({f: DPY * fits[(f, "Carhart4")]["beta"][1:] * Ef for f in FUNDS},
                           index=cols)
    contrib.loc["RF"] = DPY * rf_d
    print("\nContribution to the expected annual return (beta x 255 x recent mean):")
    common.show(contrib, "{:+.4f}")
    D["recent"] = recent
    return out


# --------------------------------------------------------------------------
# (e) worst momentum day
# --------------------------------------------------------------------------
def part_e(D: dict) -> pd.DataFrame:
    fac, d, fits = D["fac"], D["d"], D["fits"]
    i = fac.Mom.idxmin()
    mom_min, when = fac.Mom[i], fac.Date[i]
    rows = []
    for f in FUNDS:
        b = fits[(f, "Carhart4")]["beta"][4]
        ret = b * mom_min
        sd = d[f].std(ddof=1)
        rows.append({"fund": f, "beta_Mom": b, "fund return": ret,
                     "P&L on $10m": ret * NOTIONAL, "daily sd": sd,
                     "loss / daily sd": -ret / sd,
                     "worst day in sample": d[f].min(),
                     "worst / sd": -d[f].min() / sd})
    out = pd.DataFrame(rows).set_index("fund")
    common.header("Problem 5(e) - the worst daily momentum return, through the 4-factor beta")
    print(f"Worst Mom day in the file: {when.date()}  Mom = {mom_min:+.4f} "
          f"({mom_min / fac.Mom.std(ddof=1):+.1f} full-history sd)\n")
    common.show(out, {"beta_Mom": "{:+.4f}", "fund return": "{:+.4%}", "P&L on $10m": "{:+,.0f}",
                      "daily sd": "{:.5f}", "loss / daily sd": "{:.2f}",
                      "worst day in sample": "{:+.4%}", "worst / sd": "{:.2f}"})
    return out


# --------------------------------------------------------------------------
# (f) the alpha change, decomposed
# --------------------------------------------------------------------------
def part_f(D: dict) -> pd.DataFrame:
    """OLS passes through the means, so alpha = ybar - sum beta_k fbar_k
    for every model. Hence, exactly,
        alpha_CAPM - alpha_4F = (beta_mkt,4F - beta_mkt,CAPM) fbar_mkt
                                + beta_SMB fbar_SMB + beta_HML fbar_HML
                                + beta_Mom fbar_Mom ."""
    d, fits = D["d"], D["fits"]
    cols = MODELS["Carhart4"]
    fbar = d[cols].mean()
    rows = []
    for f in FUNDS:
        c, k = fits[(f, "CAPM")], fits[(f, "Carhart4")]
        bk = dict(zip(k["names"], k["beta"]))
        parts = {"d beta_mkt x mkt": (bk["Mkt-RF"] - c["beta"][1]) * fbar["Mkt-RF"],
                 "beta_SMB x SMB": bk["SMB"] * fbar["SMB"],
                 "beta_HML x HML": bk["HML"] * fbar["HML"],
                 "beta_Mom x Mom": bk["Mom"] * fbar["Mom"]}
        rows.append({"fund": f, "alpha CAPM": DPY * c["beta"][0], "alpha 4F": DPY * k["beta"][0],
                     "drop": DPY * (c["beta"][0] - k["beta"][0]),
                     **{p: DPY * v for p, v in parts.items()},
                     "sum of parts": DPY * sum(parts.values())})
    out = pd.DataFrame(rows).set_index("fund")
    common.header("Problem 5(f) - CAPM alpha minus 4-factor alpha, by factor (annualised)")
    common.show(out.T, "{:+.4f}")
    return out


def figure(D: dict) -> None:
    d, fits = D["d"], D["fits"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    ax = axes[0]
    for f, c in zip(FUNDS, (C1, C2)):
        ax.plot(d.Date, np.cumprod(1 + d[f]), color=c, lw=1.2, label=f)
    ax.plot(d.Date, np.cumprod(1 + d["Mkt-RF"] + d.RF), color=INK2, lw=1.0, ls="--", label="market")
    ax.set_title("growth of $1")
    ax.legend()
    ax = axes[1]
    x = np.arange(3)
    w = 0.36
    for j, (f, c) in enumerate(zip(FUNDS, (C1, C2))):
        a = [DPY * fits[(f, m)]["beta"][0] for m in MODELS]
        se = [DPY * fits[(f, m)]["se"][0] for m in MODELS]
        ax.bar(x + (j - 0.5) * w, a, w, yerr=1.96 * np.array(se), color=c, alpha=0.8,
               capsize=3, label=f)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, list(MODELS))
    ax.set_ylabel("annualised alpha (+/- 1.96 se)")
    ax.set_title("alpha by model")
    ax.legend()
    fig.tight_layout()
    common.finish(fig, "problem5")


def run_all() -> None:
    D = prepare()
    part_a(D)
    part_b(D)
    part_c(D)
    part_d(D)
    part_e(D)
    part_f(D)
    figure(D)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    common.use_style()
    run_all()

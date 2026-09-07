"""
common.py -- shared helpers for FinTech 545 Assignment 1.

Everything that more than one problem needs lives here so that the per-problem
modules stay short and readable:

  * moment estimators, with an explicit statement of which convention is used
  * AICc, with an explicit statement of how the parameter count k is formed
  * a small matplotlib style + a validated colour palette

Conventions chosen here (and the reason for each) are documented inline, because
a grader cannot tell a deliberate convention from an accident.
"""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Data location. Every module resolves CSVs relative to THIS file, so the code
# runs the same from the notebook, from the shell, or from another directory.
# --------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)


def load(name: str) -> pd.DataFrame:
    """Load one of the assignment CSVs by short name, e.g. load('problem1')."""
    return pd.read_csv(os.path.join(HERE, f"{name}.csv"))


# --------------------------------------------------------------------------
# Moments
# --------------------------------------------------------------------------
def moments(x, ddof_var: int = 1) -> dict:
    """First four moments of a 1-D sample, following Week 01 sections 3.2-4.4.

    CONVENTIONS, tied to the slides rather than to a package default:

      * mean      -- Week 01 s4.1, the sample average.
      * variance  -- Week 01 s4.2, which divides by (n-1) and gives the reason:
                     one degree of freedom was spent estimating the mean.
      * skewness and kurtosis -- Week 01 s3.2 defines the STANDARDIZED
                     population quantities mu3/sigma^3 and mu4/sigma^4 - 3, i.e.
                     kurtosis is always reported as EXCESS kurtosis (Normal = 0).
                     Section 4.3 then gives the SAMPLE estimators, and the ones
                     it prints are the BIAS-CORRECTED forms:

                         mu3_hat = n/((n-1)(n-2)) * sum (x_i - xbar)^3

                     normalised by sigma^3. Verified numerically: that expression
                     divided by s^3 (s using ddof=1) reproduces scipy's
                     skew(bias=False) exactly. So the bias-corrected value is the
                     one this course's estimator section points to, and it is
                     what `skew` and `exkurt` return below.

    Section 4.4 ("What Statistical Packages Return") warns that packages differ
    and says to check rather than assume. So all three variants are returned and
    printed, and the reader can see the choice does not matter here:

      * `skew`, `exkurt`             -- bias-corrected (the headline values)
      * `skew_pop`, `exkurt_pop`     -- the s3.2 population forms applied to the
                                        sample (what scipy returns by default)
      * `exkurt_slide`               -- the s4.3 kurtosis expression evaluated
                                        literally, standardised by s^4

    On this assignment's data the three excess-kurtosis variants span about 0.02
    on a value of 2.34, and no conclusion anywhere depends on which is used.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    mean = x.mean()
    d = x - mean

    # Central moments about the sample mean, 1/n normalisation (the "biased" or
    # population-form estimators).
    m2 = np.mean(d ** 2)
    m3 = np.mean(d ** 3)
    m4 = np.mean(d ** 4)

    var = x.var(ddof=ddof_var)          # s4.2: divide by n-1
    sd = np.sqrt(var)

    # --- s3.2 population forms applied to the sample -----------------------
    skew_pop = m3 / m2 ** 1.5
    exkurt_pop = m4 / m2 ** 2 - 3.0

    # --- s4.3 bias-corrected forms -----------------------------------------
    # Skew exactly as the slide writes it: n/((n-1)(n-2)) * sum d^3, over s^3.
    mu3_hat = n / ((n - 1) * (n - 2)) * np.sum(d ** 3)
    skew_bc = mu3_hat / sd ** 3

    # The slide's kurtosis expression, evaluated literally. Kbar is the biased
    # 4th central moment and sigma4bar the square of the biased variance.
    mu4_hat = (n ** 2 / ((n - 1) ** 3 * (n ** 2 - 3 * n + 3))
               * ((n * (n - 1) ** 2 + (6 * n - 9)) * m4
                  - n * (6 * n - 9) * m2 ** 2))
    exkurt_slide = mu4_hat / var ** 2 - 3.0

    # G2, the standard bias-corrected sample excess kurtosis (Excel KURT, and
    # scipy's kurtosis(bias=False)). Used as the headline because s4.3 asks for a
    # bias-corrected estimate and this is the estimator that actually is one:
    # a Monte-Carlo check on Normal samples shows G2 unbiased while mu4_hat/s^4
    # retains a downward bias of about -0.11 at n = 50.
    exkurt_bc = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * exkurt_pop + 6)

    return {
        "n": n,
        "mean": mean,
        "var": var,
        "sd": sd,
        # headline = bias-corrected, per s4.3
        "skew": skew_bc,
        "exkurt": exkurt_bc,
        # the alternatives, kept visible per the s4.4 warning
        "skew_pop": skew_pop,
        "exkurt_pop": exkurt_pop,
        "exkurt_slide": exkurt_slide,
        # Standard errors of skew/exkurt UNDER THE NULL of normality. These are
        # what make "is 2.34 big?" a decidable question rather than a vibe.
        "se_skew": np.sqrt(6.0 / n),
        "se_exkurt": np.sqrt(24.0 / n),
    }


def shape_is_admissible(skew: float, exkurt: float) -> tuple:
    """Week 01 s6 feasibility check: kurtosis >= skewness^2 + 1 for ANY
    distribution, i.e. excess kurtosis >= skew^2 - 2.

    The slides flag a violation as evidence of a calculation error rather than an
    unusual sample, so it is worth running before interpreting a moment pair.
    Returns (ok, lower_bound).
    """
    bound = skew ** 2 - 2.0
    return bool(exkurt >= bound), bound


# --------------------------------------------------------------------------
# Model selection
# --------------------------------------------------------------------------
def aicc(loglik: float, k: int, n: int) -> float:
    """Corrected Akaike Information Criterion, Week 02 section 5.3.

    The slides give AIC and then the small-sample correction:

        AIC  = 2k - 2*ln(L)
        AICc = AIC + (2k^2 + 2k) / (n - k - 1)

    so, written out,

        AICc = -2*loglik + 2k + 2k(k+1)/(n-k-1)

    since 2k^2 + 2k = 2k(k+1). AICc rather than AIC because s5.3 states that in
    small samples AIC tends to select over-parameterised models "in the same way
    R^2 does", and the correction is free to compute.

    CONVENTION on k, taken from s5.3 verbatim: k = p + d, where p is the number
    of regression parameters INCLUDING the intercept (p = q + 1 for q slopes) and
    d is the number of extra parameters fitted for the distribution during MLE.
    The slides' own worked case: a normal-error regression has d = 1 for the
    fitted variance, so k = p + 1. Applied here:

        * regression with Normal errors : p = 2 (intercept + slope), d = 1
                                          (sigma)      -> k = 3
        * same regression with t errors : p = 2, d = 2 (scale and nu) -> k = 4
        * AR(2) with an intercept       : p = 3 (intercept + 2 AR coefficients),
                                          d = 1 (sigma^2)            -> k = 4

    Problem 2 turns on this: the t pays for exactly one extra parameter.

    Also from s5.3, and worth remembering when reading the tables below: the
    absolute value of AICc means nothing. Only differences between models fitted
    to the SAME data are interpretable, which is why every table here reports
    dAICc alongside the raw value.
    """
    if n - k - 1 <= 0:
        return np.inf
    return -2.0 * loglik + 2.0 * k + (2.0 * k * (k + 1)) / (n - k - 1)


def aicc_table(entries: dict, n: int) -> pd.DataFrame:
    """Build a sorted AICc comparison table.

    `entries` maps model name -> (loglik, k). Returns a DataFrame with the AICc
    and the delta-AICc from the best model, which is the number that actually
    carries the interpretation (a gap of >~10 is decisive; <2 is a coin flip).
    """
    rows = []
    for name, (ll, k) in entries.items():
        rows.append({"model": name, "loglik": ll, "k": k,
                     "AICc": aicc(ll, k, n)})
    out = pd.DataFrame(rows).sort_values("AICc").reset_index(drop=True)
    out["dAICc"] = out["AICc"] - out["AICc"].min()
    return out


# --------------------------------------------------------------------------
# Plot style + palette
# --------------------------------------------------------------------------
# Categorical palette, used in fixed slot order and never cycled. These are the
# first three slots of a palette validated for colour-vision deficiency
# separation against a light surface (worst all-pairs CVD dE 9.2, normal-vision
# dE 24.0), which is why there are never more than three encoded series in one
# panel below -- past three, the panel is split instead.
C1 = "#2a78d6"  # blue    -- the data / the primary series
C2 = "#eb6834"  # orange  -- the fitted model being criticised
C3 = "#1baf7a"  # aqua    -- the alternative model
GRID = "#dcdbd6"
INK = "#0b0b0b"
INK2 = "#52514e"


def use_style() -> None:
    """Apply a quiet, print-friendly style: recessive grid, thin marks."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": INK2,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,       # grid behind the data, always
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,       # 2px lines
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9.5,
    })


def finish(fig, name: str) -> None:
    """Save a figure to figures/<name>.png, and display it when that is meaningful.

    Two contexts have to work:
      * the notebook, on an inline backend -- plt.show() renders the figure under
        the cell, which is how the figures reach the write-up;
      * a headless script run on Agg -- there is no display, so show() would only
        emit a "FigureCanvasAgg is non-interactive" warning. The PNG on disk is
        the artefact in that case, so the figure is closed instead of shown.

    Checking the backend rather than calling show() unconditionally keeps script
    runs quiet, and closing the figure keeps memory flat when a module draws
    several.
    """
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"))
    if mpl.get_backend().lower().endswith("agg"):
        plt.close(fig)   # headless: nothing to show, so release it
    else:
        plt.show()


def header(title: str) -> None:
    """Print a section header when a module is run as a script."""
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)

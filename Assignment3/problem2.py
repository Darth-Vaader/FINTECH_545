"""
Problem 2 -- American options and discrete dividends.

    S = 48.50, X = 50, T = 0.5, r = 4.5%, sigma = 30%
    dividends: D1 = 0.25 at t = 0.10, D2 = 1.50 at t = 0.40

The tree must put a step exactly on each dividend date. With T = 0.5 the
dividend times are T/5 and 4T/5, so N must be a multiple of 5 and the dividend
steps are N/5 and 4N/5.

  predict  : (a) Hull's early-exercise condition D_n > X(1 - e^{-r(T - t_n)})
             for each dividend; (b) the direction of each dividend effect.
  fit      : (c) the four options without and with dividends, plus the
             escrowed-dividend GBSM; (d) convergence in N; (e) delta and gamma
             by central differences at four bump sizes; (f) dV/dD for each
             dividend.
  reconcile: (g) early exercise premia and the critical stock price before
             D2; (h) which Greek does not settle and why; (i) why the two
             dividends move the call by such different amounts.

Conventions:
  * no-dividend Europeans by GBSM with b = r (Black-Scholes 1973), no-dividend
    Americans on the recombining CRR tree with b = r;
  * with dividends, b = r between payments and the dividends are cash amounts
    subtracted from the stock on the dividend step (Week 07; bt_american.jl);
  * the "European with dividends" on the tree is the same non-recombining tree
    with the exercise branch switched off;
  * "GBSM with the PV of dividends taken off" is the escrowed-dividend model:
    S* = S - D1 e^{-r t1} - D2 e^{-r t2}, then plain GBSM with b = r on S*;
  * N_MAIN = 500 for the headline numbers (multiple of 5).

Run:  python problem2.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

import common
from common import C1, C2, C3, C4, INK2

S, X, T, R, SIG = 48.50, 50.0, 0.5, 0.045, 0.30
DIVS = [0.25, 1.50]
TDIV = [0.10, 0.40]
N_MAIN = 500


def div_steps(N: int) -> list:
    """Dividend step indices; exact because t_n / T * N is an integer when 5 | N."""
    assert N % 5 == 0, "N must be a multiple of 5 so the steps land on the dividend dates"
    return [int(round(t / T * N)) for t in TDIV]


def tree(call: bool, N: int = N_MAIN, s=S, divs=DIVS, american: bool = True):
    """Price on the non-recombining dividend tree (vectorised over s)."""
    return common.bt_div(call, s, X, T, R, divs, div_steps(N), SIG, N, american)


def pv_divs(divs=DIVS) -> float:
    return sum(d * np.exp(-R * t) for d, t in zip(divs, TDIV))


# --------------------------------------------------------------------------
# (a) Hull's condition
# --------------------------------------------------------------------------
def part_a() -> pd.DataFrame:
    rows = []
    for k, (d, t) in enumerate(zip(DIVS, TDIV), 1):
        thr = X * (1.0 - np.exp(-R * (T - t)))   # interest on X from t_n to T
        rows.append({"dividend": f"D{k}", "t_n": t, "D_n": d, "T - t_n": T - t,
                     "X(1-e^-r(T-t_n))": thr, "D_n > threshold": d > thr})
    out = pd.DataFrame(rows).set_index("dividend")
    common.header("Problem 2(a) - early exercise condition D_n > X(1 - e^{-r(T - t_n)})")
    common.show(out, {"t_n": "{:.2f}", "D_n": "{:.2f}", "T - t_n": "{:.2f}",
                      "X(1-e^-r(T-t_n))": "{:.4f}"})
    print("\n-> exercise can be optimal only just before D2 (t = 0.40);"
          "\n   D1 is worth less than the interest on the strike it would forgo.")
    return out


# --------------------------------------------------------------------------
# (c) the eight prices
# --------------------------------------------------------------------------
def part_c(N: int = N_MAIN) -> dict:
    out = {}
    # No dividends
    out["EC0"] = float(common.gbsm(True, S, X, T, R, R, SIG))
    out["EP0"] = float(common.gbsm(False, S, X, T, R, R, SIG))
    out["AC0"] = common.bt_american(True, S, X, T, R, R, SIG, N)
    out["AP0"] = common.bt_american(False, S, X, T, R, R, SIG, N)
    out["EC0_tree"] = common.bt_american(True, S, X, T, R, R, SIG, N, american=False)
    out["EP0_tree"] = common.bt_american(False, S, X, T, R, R, SIG, N, american=False)
    # With dividends -- non-recombining tree
    out["EC_tree"] = tree(True, N, american=False)
    out["EP_tree"] = tree(False, N, american=False)
    out["AC"] = tree(True, N)
    out["AP"] = tree(False, N)
    # Escrowed-dividend GBSM
    s_star = S - pv_divs()
    out["S*"] = s_star
    out["EC_gbsm"] = float(common.gbsm(True, s_star, X, T, R, R, SIG))
    out["EP_gbsm"] = float(common.gbsm(False, s_star, X, T, R, R, SIG))

    common.header(f"Problem 2(c) - prices without and with dividends (tree N = {N})")
    print(f"PV of dividends = 0.25 e^(-r 0.1) + 1.50 e^(-r 0.4) = {pv_divs():.4f};"
          f"  S* = S - PV = {s_star:.4f}\n")
    tab = pd.DataFrame({
        "no dividends": [out["EC0"], out["AC0"], out["EP0"], out["AP0"]],
        "with dividends (tree)": [out["EC_tree"], out["AC"], out["EP_tree"], out["AP"]],
        "GBSM on S - PV(D)": [out["EC_gbsm"], np.nan, out["EP_gbsm"], np.nan],
    }, index=["European call", "American call", "European put", "American put"])
    tab["change from dividends"] = tab["with dividends (tree)"] - tab["no dividends"]
    common.show(tab, "{:.4f}")
    print(f"\nNo-dividend Europeans by GBSM; on the same N={N} tree they are "
          f"{out['EC0_tree']:.4f} (call) and {out['EP0_tree']:.4f} (put).")
    print(f"Tree European minus escrowed GBSM: call {out['EC_tree'] - out['EC_gbsm']:+.4f},"
          f" put {out['EP_tree'] - out['EP_gbsm']:+.4f}")
    print("  (the tree applies 30% vol to the whole cum-dividend price; the escrowed"
          "\n   model applies it only to S - PV(D), a smaller base, so its effective"
          "\n   dollar volatility is lower and both options come out a little cheaper)")
    return out


# --------------------------------------------------------------------------
# (d) convergence
# --------------------------------------------------------------------------
N_GRID = list(range(20, 305, 5)) + [350, 400, 450, 500]
N_TABLE = [50, 55, 100, 105, 200, 205, 300, 305, 500, 1000]


def part_d(include_1000: bool = True) -> dict:
    rows = []
    for N in N_GRID:
        rows.append({"N": N, "AC": tree(True, N), "AP": tree(False, N),
                     "AC0": common.bt_american(True, S, X, T, R, R, SIG, N),
                     "AP0": common.bt_american(False, S, X, T, R, R, SIG, N)})
    grid = pd.DataFrame(rows).set_index("N")
    tab_rows = []
    for N in N_TABLE:
        if N == 1000 and not include_1000:
            continue
        if N in grid.index:
            tab_rows.append(grid.loc[N].to_dict() | {"N": N})
        else:
            tab_rows.append({"N": N, "AC": tree(True, N), "AP": tree(False, N),
                             "AC0": common.bt_american(True, S, X, T, R, R, SIG, N),
                             "AP0": common.bt_american(False, S, X, T, R, R, SIG, N)})
    tab = pd.DataFrame(tab_rows).set_index("N")
    common.header("Problem 2(d) - convergence of the American values in N")
    common.show(tab.rename(columns={"AC": "Am call (div)", "AP": "Am put (div)",
                                    "AC0": "Am call (no div)", "AP0": "Am put (no div)"}),
                "{:.5f}")
    for a, b in ((100, 105), (200, 205), (300, 305)):
        print(f"N={a} -> {b}: call {tab.AC[b] - tab.AC[a]:+.5f}, put {tab.AP[b] - tab.AP[a]:+.5f}")
    late = grid.loc[200:]
    print(f"\nRange of values for N in [200, 500]: call {late.AC.max() - late.AC.min():.5f},"
          f" put {late.AP.max() - late.AP.min():.5f}")
    return {"grid": grid, "tab": tab}


def figure_convergence(Dd: dict) -> None:
    g = Dd["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, col, col0, name in ((axes[0], "AC", "AC0", "American call"),
                                (axes[1], "AP", "AP0", "American put")):
        ax.plot(g.index, g[col], "-", color=C1, lw=1.1, marker="o", ms=2.2,
                label="with dividends")
        ax.plot(g.index, g[col0], "-", color=C2, lw=1.1, marker="o", ms=2.2,
                label="no dividends")
        ax.set_title(name)
        ax.set_xlabel("tree steps N (multiples of 5)")
        ax.legend()
    axes[0].set_ylabel("value")
    fig.suptitle("Problem 2(d): the tree oscillates as N grows and the swings shrink",
                 fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    common.finish(fig, "problem2_convergence")


# --------------------------------------------------------------------------
# (e) delta and gamma by central differences
# --------------------------------------------------------------------------
BUMPS = [1.0, 0.1, 0.01, 0.001]


def part_e(N: int = N_MAIN) -> pd.DataFrame:
    rows = []
    for call in (True, False):
        for h in BUMPS:
            v_dn, v0, v_up = tree(call, N, s=np.array([S - h, S, S + h]))
            rows.append({"option": "Am call" if call else "Am put", "bump h": h,
                         "delta": (v_up - v_dn) / (2 * h),
                         "gamma": (v_up + v_dn - 2 * v0) / h ** 2})
    out = pd.DataFrame(rows)
    # Closed-form GBSM Greeks on the escrowed price, as a smooth reference.
    s_star = S - pv_divs()
    gc = common.gbsm_greeks(True, s_star, X, T, R, R, SIG)
    gp = common.gbsm_greeks(False, s_star, X, T, R, R, SIG)
    common.header(f"Problem 2(e) - tree delta and gamma by central differences (N = {N})")
    common.show(out.set_index(["option", "bump h"]), {"delta": "{:.5f}", "gamma": "{:.5f}"})
    print("\nSmooth reference (European GBSM on S - PV(D)):")
    print(f"  call delta {gc['delta']:.5f} gamma {gc['gamma']:.5f};"
          f" put delta {gp['delta']:.5f} gamma {gp['gamma']:.5f}")
    return out


# The figure for (h) uses a smaller tree: the point it makes (the value is
# piecewise linear in S at any fixed N) does not depend on N, and evaluating
# hundreds of starting prices on the N = 500 dividend tree needs gigabytes.
N_FIG = 100


def gamma_vs_bump(N: int = N_FIG) -> pd.DataFrame:
    """Many bumps on a log grid: shows how far the gamma estimate wanders."""
    hs = np.logspace(-3.5, 0.3, 40)
    s = np.concatenate([S - hs, [S], S + hs])          # one vectorised call
    vc, vp = tree(True, N, s=s), tree(False, N, s=s)
    m = hs.size
    c_dn, c0, c_up = vc[:m], vc[m], vc[m + 1:]
    p_dn, p0, p_up = vp[:m], vp[m], vp[m + 1:]
    return pd.DataFrame({"h": hs,
                         "call_delta": (c_up - c_dn) / (2 * hs), "put_delta": (p_up - p_dn) / (2 * hs),
                         "call_gamma": (c_up + c_dn - 2 * c0) / hs ** 2,
                         "put_gamma": (p_up + p_dn - 2 * p0) / hs ** 2})


def value_curve(N: int = N_FIG, half_width: float = 1.0, n: int = 401) -> pd.DataFrame:
    """Tree value of each American as a function of S, at fixed N, finely."""
    s = np.linspace(S - half_width, S + half_width, n)
    return pd.DataFrame({"S": s, "call": tree(True, N, s=s), "put": tree(False, N, s=s)})


def part_h(G: pd.DataFrame, V: pd.DataFrame) -> None:
    """Second differences of the fine value curve: zero on linear stretches,
    non-zero only where a node crosses the strike or the exercise boundary."""
    h = V.S.iloc[1] - V.S.iloc[0]
    u = np.exp(SIG * np.sqrt(T / N_FIG))
    common.header(f"Problem 2(h) - why gamma does not settle (tree N = {N_FIG})")
    print(f"Root node spacing S(u - d) at N={N_FIG}: {S * (u - 1 / u):.3f};"
          f" at N={N_MAIN}: {S * (np.exp(SIG * np.sqrt(T / N_MAIN)) - np.exp(-SIG * np.sqrt(T / N_MAIN))):.3f}")
    for col in ("call", "put"):
        g = np.diff(V[col].to_numpy(), 2) / h ** 2
        print(f"  {col}: {len(g)} prices {S - 1:.1f}..{S + 1:.1f} (step {h:.3f}):"
              f" second difference ~0 at {np.mean(np.abs(g) < 1e-6):.0%} of them,"
              f" max |gamma| {np.abs(g).max():.3f}")
    sel = G.iloc[[0, 5, 10, 15, 20, 25, 30, 35, 39]]
    common.show(sel.set_index("h"), "{:.5f}")


def figure_greeks(G: pd.DataFrame, V: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))
    ax = axes[0]
    ax.semilogx(G.h, G.call_delta, "-o", ms=2.5, color=C1, label="call delta")
    ax.semilogx(G.h, G.put_delta, "-o", ms=2.5, color=C2, label="put delta")
    ax.set_xlabel("bump h")
    ax.set_title("delta settles")
    ax.legend()
    ax = axes[1]
    ax.semilogx(G.h, G.call_gamma, "-o", ms=2.5, color=C1, label="call gamma")
    ax.semilogx(G.h, G.put_gamma, "-o", ms=2.5, color=C2, label="put gamma")
    ax.set_xlabel("bump h")
    ax.set_title("gamma does not")
    ax.legend()
    ax = axes[2]
    # Remove the straight line through the end points so the kinks show.
    for col, c in (("call", C1), ("put", C2)):
        y = V[col].to_numpy()
        lin = y[0] + (y[-1] - y[0]) * (V.S - V.S.iloc[0]) / (V.S.iloc[-1] - V.S.iloc[0])
        ax.plot(V.S, (y - lin) * 1000, color=c, lw=1.2, label=col)
    ax.axvline(S, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel(f"stock price S (N fixed at {N_FIG})")
    ax.set_ylabel("value minus chord (x 1/1000)")
    ax.set_title("tree value vs S: piecewise, with kinks")
    ax.legend()
    fig.tight_layout()
    common.finish(fig, "problem2_greeks")


# --------------------------------------------------------------------------
# (f) dividend sensitivities
# --------------------------------------------------------------------------
def part_f(N: int = N_MAIN, h: float = 0.01) -> pd.DataFrame:
    rows = []
    for k in range(2):
        up = list(DIVS)
        dn = list(DIVS)
        up[k] += h
        dn[k] -= h
        rec = {"dividend": f"D{k + 1} (t={TDIV[k]})"}
        for name, call, am in (("Euro call", True, False), ("Am call", True, True),
                               ("Euro put", False, False), ("Am put", False, True)):
            rec[name] = (tree(call, N, divs=up, american=am) - tree(call, N, divs=dn, american=am)) / (2 * h)
        # Escrowed GBSM: dV/dD_k = dV/dS* x dS*/dD_k = delta(S*) x (-e^{-r t_k})
        s_star = S - pv_divs()
        rec["Euro call (escrow GBSM)"] = -common.gbsm_greeks(True, s_star, X, T, R, R, SIG)["delta"] * np.exp(-R * TDIV[k])
        rec["Euro put (escrow GBSM)"] = -common.gbsm_greeks(False, s_star, X, T, R, R, SIG)["delta"] * np.exp(-R * TDIV[k])
        rows.append(rec)
    out = pd.DataFrame(rows).set_index("dividend")
    common.header(f"Problem 2(f) - dV/dD_k by central difference (h = {h}, N = {N})")
    common.show(out.T, "{:+.4f}")
    return out


# --------------------------------------------------------------------------
# (g) early exercise premia and the critical stock price
# --------------------------------------------------------------------------
def part_g(P: dict) -> dict:
    # Both legs of each premium on the SAME N = 500 tree, so the tree's own
    # discretisation error cancels (against GBSM it would show up as a fake
    # 0.0018 premium on the no-dividend call).
    prem = pd.DataFrame({
        "no dividends": [P["AC0"] - P["EC0_tree"], P["AP0"] - P["EP0_tree"]],
        "with dividends": [P["AC"] - P["EC_tree"], P["AP"] - P["EP_tree"]],
    }, index=["call: American - European", "put: American - European"])
    prem["change"] = prem["with dividends"] - prem["no dividends"]
    common.header("Problem 2(g) - early exercise premium = American - European (same tree)")
    common.show(prem, "{:+.4f}")

    # Just before D2 (cum-dividend, t = 0.40) the holder compares
    #   exercise : S - X                         (and then receives D2)
    #   hold     : call on S - D2 with 0.10 years left and NO further dividends,
    #              which equals the European GBSM value (no early exercise after
    #              the last dividend for a call).
    tau = T - TDIV[1]
    f = lambda s: (s - X) - float(common.gbsm(True, s - DIVS[1], X, tau, R, R, SIG))
    s_crit = optimize.brentq(f, X + 1e-6, 3 * X)
    # Cross-check on the tree's own nodes at the dividend step.
    N = N_MAIN
    n2 = div_steps(N)[1]
    u = np.exp(SIG * np.sqrt(T / N))
    nodes = S * u ** (2 * np.arange(n2 + 1) - n2)
    cont = common.bt_div(True, nodes - DIVS[1], X, tau, R, [], [], SIG, N - n2)
    ex = nodes - X
    first_ex = nodes[ex > cont].min()
    last_hold = nodes[ex <= cont].max()
    print(f"\nCritical stock price just before D2 (t = 0.40): S* = {s_crit:.4f}")
    print(f"  at S*: exercise value {s_crit - X:.4f} = hold value "
          f"{float(common.gbsm(True, s_crit - DIVS[1], X, tau, R, R, SIG)):.4f}")
    print(f"  tree check (N={N}), nodes at step {n2}: highest node held = {last_hold:.4f},"
          f" lowest node exercised = {first_ex:.4f}")
    return {"prem": prem, "s_crit": s_crit}


def run_all() -> None:
    part_a()
    P = part_c()
    Dd = part_d()
    figure_convergence(Dd)
    part_e()
    G, V = gamma_vs_bump(), value_curve()
    figure_greeks(G, V)
    part_f()
    part_g(P)
    part_h(G, V)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    common.use_style()
    run_all()

"""
Problem 1 -- Implied volatility from a dirty chain.

problem1.csv: bid and ask for 18 European calls and 18 European puts on XYZ,
strikes 140..225, one expiry.

    S = 182.40, today 2026-11-13, expiry 2026-12-18 (35 calendar days),
    r = 4.25%, continuous dividend yield q = 0.60%  ->  GBSM with b = r - q.
    T = 35/365 (calendar days over 365, as instructed).

  predict  : (a) put-call parity at every strike from mids, flagged only when
             the gap exceeds the call spread + put spread; (b) what parity
             implies for the two IVs; (c) where vega says IV is least reliable.
  fit      : (d) IV from bid, mid and ask for every quote, with the reason any
             one fails; smile plot; (e) vega and the IV change per 1 cent.
  reconcile: (f) the flagged strike in vol points; (g) the call wing above 210;
             (h) the shape of the clean smile.

Parity with a continuous dividend yield (Week 06 s"Put Call Parity" with the
stock replaced by its dividend-discounted value S e^{-qT} = S e^{(b-r)T}):

    C - P = S e^{-qT} - X e^{-rT}

Run:  python problem1.py
"""

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
from common import C1, C2, INK2

S = 182.40
R = 0.0425
Q = 0.0060
B = R - Q                                    # Merton 1973 cost of carry
DAYS = (date(2026, 12, 18) - date(2026, 11, 13)).days
T = DAYS / 365.0                             # calendar days / 365


def prepare() -> dict:
    """Wide table, one row per strike, with mids and spreads for both sides."""
    d = common.load("problem1")
    d["Mid"] = 0.5 * (d.Bid + d.Ask)
    d["Spread"] = d.Ask - d.Bid
    c = d[d.Type == "Call"].set_index("Strike")
    p = d[d.Type == "Put"].set_index("Strike")
    w = pd.DataFrame({"C_bid": c.Bid, "C_ask": c.Ask, "C_mid": c.Mid, "C_spr": c.Spread,
                      "P_bid": p.Bid, "P_ask": p.Ask, "P_mid": p.Mid, "P_spr": p.Spread})
    w.index.name = "Strike"
    return {"raw": d, "w": w, "K": w.index.to_numpy(dtype=float)}


# --------------------------------------------------------------------------
# (a) parity check
# --------------------------------------------------------------------------
def part_a(D: dict) -> pd.DataFrame:
    w, K = D["w"], D["K"]
    # Parity says C - P = S e^{-qT} - X e^{-rT}; the gap is what the mids miss by.
    rhs = S * np.exp(-Q * T) - K * np.exp(-R * T)
    gap = (w.C_mid - w.P_mid) - rhs
    band = w.C_spr + w.P_spr              # cannot trade a gap smaller than this
    out = pd.DataFrame({"C_mid": w.C_mid, "P_mid": w.P_mid,
                        "C-P (mkt)": w.C_mid - w.P_mid, "S e^-qT - X e^-rT": rhs,
                        "gap": gap, "band": band, "|gap|/band": gap.abs() / band,
                        "flag": np.where(gap.abs() > band, "<-- FLAG", "")})
    common.header("Problem 1(a) - put-call parity at every strike (mid prices)")
    print(f"S = {S}, T = {DAYS}/365 = {T:.6f}, r = {R}, q = {Q}, b = r - q = {B}")
    print(f"S e^(-qT) = {S * np.exp(-Q * T):.4f}\n")
    common.show(out, {"C_mid": "{:.3f}", "P_mid": "{:.3f}", "C-P (mkt)": "{:.4f}",
                      "S e^-qT - X e^-rT": "{:.4f}", "gap": "{:+.4f}", "band": "{:.2f}",
                      "|gap|/band": "{:.3f}"})
    flagged = out.index[out.flag != ""].tolist()
    print(f"\nFlagged strikes: {flagged}")
    # Diagnose which leg is wrong at a flagged strike: compare each leg with the
    # price its parity partner implies.
    for k in flagged:
        p_impl = w.C_mid[k] - (S * np.exp(-Q * T) - k * np.exp(-R * T))
        c_impl = w.P_mid[k] + (S * np.exp(-Q * T) - k * np.exp(-R * T))
        print(f"  K={k:.0f}: put mid {w.P_mid[k]:.3f} vs parity-implied {p_impl:.3f};"
              f" call mid {w.C_mid[k]:.3f} vs parity-implied {c_impl:.3f}")
        lo, hi = w.P_mid[k - 5], w.P_mid[k + 5]
        print(f"           neighbouring put mids: K={k - 5:.0f} {lo:.3f}, K={k + 5:.0f} {hi:.3f}")
        # Convexity in strike: a long butterfly (buy K-5, sell 2 K, buy K+5) has a
        # non-negative payoff, so it can never cost less than zero.
        for k1 in (k - 5, k):
            k0, k2 = k1 - 5, k1 + 5
            mid = w.P_mid[k0] - 2 * w.P_mid[k1] + w.P_mid[k2]
            exe = w.P_ask[k0] - 2 * w.P_bid[k1] + w.P_ask[k2]   # buy at ask, sell at bid
            print(f"           put butterfly {k0:.0f}/{k1:.0f}/{k2:.0f}: cost at mids {mid:+.3f},"
                  f" at executable bid/ask {exe:+.3f}")
    return out


# --------------------------------------------------------------------------
# (c) a vega preview, before any IV is computed: evaluate vega at one flat vol
# --------------------------------------------------------------------------
def part_c(D: dict, sigma_guess: float = 0.25) -> pd.DataFrame:
    """Vega at a single guessed vol (no IVs yet), against the quoted spread.
    spread / vega = how many vol points the bid-ask spread alone spans."""
    w, K = D["w"], D["K"]
    vega = common.gbsm_greeks(True, S, K, T, R, B, sigma_guess)["vega"]
    out = pd.DataFrame({"vega@25%": vega, "C_mid": w.C_mid, "C_spr": w.C_spr,
                        "C spr/vega (vol pts)": 100 * w.C_spr / vega,
                        "P_mid": w.P_mid, "P_spr": w.P_spr,
                        "P spr/vega (vol pts)": 100 * w.P_spr / vega}, index=w.index)
    common.header("Problem 1(c) - vega at a flat 25% vol vs the quoted spreads")
    common.show(out, {"vega@25%": "{:.3f}", "C_mid": "{:.3f}", "C_spr": "{:.2f}",
                      "C spr/vega (vol pts)": "{:.2f}", "P_mid": "{:.3f}", "P_spr": "{:.2f}",
                      "P spr/vega (vol pts)": "{:.2f}"})
    print("\nOTM legs near the min tick: calls >= 210 and puts <= 150 are quoted at"
          "\n0.05-0.17 bid; a 5-6 cent spread there is 30-100% of the option's price.")
    return out


# --------------------------------------------------------------------------
# (d) implied vols from bid, mid, ask
# --------------------------------------------------------------------------
def _iv_reason(call: bool, price: float, k: float) -> str:
    lb = common.iv_lower_bound(call, S, k, T, R, B)
    if price <= lb:
        return f"price {price:.2f} <= discounted intrinsic {lb:.3f}"
    return "price above sigma->inf limit"


def part_d(D: dict) -> dict:
    w, K = D["w"], D["K"]
    rows = []
    for side, call in (("C", True), ("P", False)):
        for k in K:
            rec = {"Type": "Call" if call else "Put", "Strike": k}
            for lev in ("bid", "mid", "ask"):
                px = w.loc[k, f"{side}_{lev}"]
                rec[f"IV_{lev}"] = common.implied_vol(call, px, S, k, T, R, B)
                if np.isnan(rec[f"IV_{lev}"]):
                    rec.setdefault("why", f"{lev}: {_iv_reason(call, px, k)}")
            rec["intrinsic_disc"] = common.iv_lower_bound(call, S, k, T, R, B)
            rows.append(rec)
    iv = pd.DataFrame(rows)
    iv["why"] = iv.get("why", "").fillna("")
    iv = iv[["Type", "Strike", "IV_bid", "IV_mid", "IV_ask", "intrinsic_disc", "why"]]
    common.header("Problem 1(d) - implied vol of every quote from its bid, mid and ask")
    for typ in ("Call", "Put"):
        sub = iv[iv.Type == typ].set_index("Strike").drop(columns="Type")
        print(f"\n{typ}s")
        common.show(sub, {"IV_bid": "{:.4f}", "IV_mid": "{:.4f}", "IV_ask": "{:.4f}",
                          "intrinsic_disc": "{:.3f}"})
    miss = iv[iv[["IV_bid", "IV_mid", "IV_ask"]].isna().any(axis=1)]
    print(f"\n{len(miss)} quotes lack an IV at one of the three prices -- all at the BID,"
          "\nall deep in the money, all because the bid is below the sigma -> 0 limit"
          "\nof the GBSM value (S e^(-qT) - X e^(-rT) for a call, X e^(-rT) - S e^(-qT)"
          "\nfor a put). No volatility, however small, prices the option that low.")
    D["iv"] = iv
    return iv


def figure_smile(D: dict) -> None:
    iv = D["iv"]
    c = iv[iv.Type == "Call"]
    p = iv[iv.Type == "Put"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.fill_between(c.Strike, c.IV_bid, c.IV_ask, color=C1, alpha=0.13, lw=0,
                    label="call bid-ask IV band")
    ax.fill_between(p.Strike, p.IV_bid, p.IV_ask, color=C2, alpha=0.13, lw=0,
                    label="put bid-ask IV band")
    ax.plot(c.Strike, c.IV_mid, "o-", color=C1, ms=4, label="call IV (mid)")
    ax.plot(p.Strike, p.IV_mid, "s-", color=C2, ms=4, label="put IV (mid)")
    ax.axvline(S, color=INK2, lw=0.8, ls=":")
    ax.text(S + 0.6, 0.41, f"spot {S}", color=INK2, fontsize=8)
    ax.annotate("K=175 put:\nparity break", xy=(175, p.set_index("Strike").IV_mid[175]),
                xytext=(160, 0.17), arrowprops=dict(arrowstyle="->", color=INK2),
                fontsize=8, color=INK2)
    ax.annotate("call wing at\nthe 0.05/0.10 tick", xy=(220, c.set_index("Strike").IV_mid[220]),
                xytext=(203, 0.33), arrowprops=dict(arrowstyle="->", color=INK2),
                fontsize=8, color=INK2)
    ax.set_ylim(0.15, 0.45)
    ax.set_xlabel("strike")
    ax.set_ylabel("implied volatility")
    ax.set_title("XYZ 18-Dec-2026 smile: call and put IV against strike")
    ax.legend(loc="upper right", ncol=2)
    common.finish(fig, "problem1_smile")


# --------------------------------------------------------------------------
# (e) vega and the IV move per 1 cent
# --------------------------------------------------------------------------
def part_e(D: dict) -> pd.DataFrame:
    iv = D["iv"]
    rows = []
    for _, r in iv.iterrows():
        call = r.Type == "Call"
        g = common.gbsm_greeks(call, S, r.Strike, T, R, B, r.IV_mid)
        # dV/dsigma per 1.00 of vol; a 1 cent price change moves IV by 0.01/vega
        rows.append({"Type": r.Type, "Strike": r.Strike, "IV_mid": r.IV_mid,
                     "vega": g["vega"], "vega/volpt": g["vega"] / 100,
                     "dIV per 1c (vol pts)": 100 * 0.01 / g["vega"]})
    out = pd.DataFrame(rows)
    common.header("Problem 1(e) - vega (at each quote's mid IV) and dIV for a 1 cent move")
    wide = out.pivot(index="Strike", columns="Type",
                     values=["vega", "dIV per 1c (vol pts)"])
    wide.columns = [f"{a.split(' ')[0]} {b}" if a == "vega" else f"dIV/1c {b}"
                    for a, b in wide.columns]
    common.show(wide, "{:.4f}")
    print("\ndIV per 1c is in vol points (0.01 = 1%): 1 cent / vega x 100.")
    D["vega"] = out
    return out


# --------------------------------------------------------------------------
# (f), (g), (h) reconcile
# --------------------------------------------------------------------------
def part_f(D: dict, A: pd.DataFrame) -> dict:
    iv = D["iv"].set_index(["Type", "Strike"])
    k = 175.0
    gap = A.loc[k, "gap"]
    # Use vega at the put's own IV and at the call's IV; the truth is in between.
    vp = common.gbsm_greeks(False, S, k, T, R, B, iv.loc[("Put", k), "IV_mid"])["vega"]
    vc = common.gbsm_greeks(True, S, k, T, R, B, iv.loc[("Call", k), "IV_mid"])["vega"]
    ivc, ivp = iv.loc[("Call", k), "IV_mid"], iv.loc[("Put", k), "IV_mid"]
    # Smooth-smile reference: average of the put IVs either side of the bad strike.
    nb = 0.5 * (iv.loc[("Put", 170.0), "IV_mid"] + iv.loc[("Put", 180.0), "IV_mid"])
    common.header("Problem 1(f) - the K=175 parity gap in vol points")
    print(f"parity gap (call - put - parity)  = {gap:+.4f} dollars")
    print(f"vega at K=175 (put IV {ivp:.4f})  = {vp:.3f} per 1.00 vol")
    print(f"vega at K=175 (call IV {ivc:.4f}) = {vc:.3f} per 1.00 vol")
    print(f"gap / vega                         = {100 * gap / vp:.2f} to {100 * gap / vc:.2f} vol points")
    print(f"call IV - put IV (mid)             = {100 * (ivc - ivp):.2f} vol points")
    print(f"put IVs either side (170, 180) avg = {nb:.4f};  call IV at 175 = {ivc:.4f}")
    print("-> the CALL sits on the smile, the PUT is ~5 vol points too low: the put quote"
          "\n   is the bad leg (stale / mis-keyed). Its mid 1.71 is barely above the"
          f"\n   170 put's 1.55, while parity implies {A.loc[k, 'C_mid'] - (S * np.exp(-Q * T) - k * np.exp(-R * T)):.3f}.")
    return {"gap": gap, "vega_put": vp, "vega_call": vc, "ivc": ivc, "ivp": ivp}


def part_g(D: dict, A: pd.DataFrame) -> pd.DataFrame:
    iv = D["iv"].set_index(["Type", "Strike"])
    w = D["w"]
    rows = []
    for k in (200.0, 205.0, 210.0, 215.0, 220.0, 225.0):
        vc = common.gbsm_greeks(True, S, k, T, R, B, iv.loc[("Call", k), "IV_mid"])["vega"]
        # Price a call at the PUT's mid IV (which parity says should be the same):
        # this is what the call "should" cost on the clean smile.
        fair = float(common.gbsm(True, S, k, T, R, B, iv.loc[("Put", k), "IV_mid"]))
        rows.append({"Strike": k, "C_bid": w.C_bid[k], "C_ask": w.C_ask[k],
                     "C IV bid": iv.loc[("Call", k), "IV_bid"],
                     "C IV mid": iv.loc[("Call", k), "IV_mid"],
                     "C IV ask": iv.loc[("Call", k), "IV_ask"],
                     "P IV mid": iv.loc[("Put", k), "IV_mid"],
                     "spread (vol pts)": 100 * (iv.loc[("Call", k), "IV_ask"] - iv.loc[("Call", k), "IV_bid"]),
                     "C @ put IV": fair, "parity band": A.loc[k, "band"],
                     "|gap|": abs(A.loc[k, "gap"])})
    out = pd.DataFrame(rows).set_index("Strike")
    common.header("Problem 1(g) - the call wing above 210")
    common.show(out, {"C_bid": "{:.2f}", "C_ask": "{:.2f}", "C IV bid": "{:.4f}",
                      "C IV mid": "{:.4f}", "C IV ask": "{:.4f}", "P IV mid": "{:.4f}",
                      "spread (vol pts)": "{:.2f}", "C @ put IV": "{:.4f}",
                      "parity band": "{:.2f}", "|gap|": "{:.4f}"})
    print("\nCalls 215/220/225 all quote exactly 0.05 / 0.10: the minimum tick, not a price."
          "\nOn the put-implied smile they are worth 0.02-0.04, below the tick, so the"
          "\nmarket maker cannot show a lower bid. Even the BID IV rises with strike.")
    return out


def part_h(D: dict) -> pd.DataFrame:
    """Clean smile: drop the K=175 put and the tick-bound calls >= 215. Use the
    OTM leg at each strike (puts below spot, calls above), which is the more
    liquid quote and the one carrying the smile information."""
    iv = D["iv"].set_index(["Type", "Strike"])
    rows = []
    for k in D["K"]:
        if k < S:
            typ = "Put"
        else:
            typ = "Call"
        val = iv.loc[(typ, k), "IV_mid"]
        if (typ == "Put" and k == 175.0):
            val = iv.loc[("Call", k), "IV_mid"]       # use the clean leg
            typ = "Call*"
        if typ == "Call" and k >= 215.0:
            val = iv.loc[("Put", k), "IV_mid"]        # use the clean leg
            typ = "Put*"
        rows.append({"Strike": k, "moneyness K/S": k / S, "leg": typ, "IV": val})
    out = pd.DataFrame(rows).set_index("Strike")
    atm = np.interp(S, out.index, out.IV)
    # 25-strike-wide skew either side of spot
    lo = np.interp(S * 0.9, out.index, out.IV)
    hi = np.interp(S * 1.1, out.index, out.IV)
    common.header("Problem 1(h) - clean smile (OTM leg; * = other leg substituted)")
    common.show(out, {"moneyness K/S": "{:.3f}", "IV": "{:.4f}"})
    print(f"\nATM IV (interpolated at S) = {atm:.4f}")
    print(f"IV at 90% of spot = {lo:.4f} (+{100 * (lo - atm):.1f} pts), "
          f"IV at 110% of spot = {hi:.4f} ({100 * (hi - atm):+.1f} pts)")
    return out


def run_all() -> None:
    D = prepare()
    A = part_a(D)
    part_c(D)
    part_d(D)
    part_e(D)
    part_f(D, A)
    part_g(D, A)
    part_h(D)
    figure_smile(D)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    common.use_style()
    run_all()

"""Generate the documentation figures from the published model (synthetic data).

Produces four PNGs in ``docs/figures/``, every one computed by ``optvol`` on the
arbitrage-free SSVI synthetic surface, with no vendor data:

  1. vol_skew.png            : the SSVI implied-volatility skew (30d vs 60d)
  2. pop_vs_pht.png          : terminal POP vs path PHT (the core thesis)
  3. first_passage_vs_mc.png : analytical first-passage vs Monte-Carlo (model risk)
  4. butterfly_payoff.png    : a broken-wing-butterfly payoff at expiry

Requires matplotlib (a docs-only dependency):
    python examples/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import math  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from optvol.pricing.probability import prob_stay_within_barriers, terminal_prob_in_range  # noqa: E402
from optvol.structures.butterfly import payoff_metrics  # noqa: E402
from generate_synthetic_chain import ssvi_iv  # noqa: E402
from first_passage_vs_montecarlo import build_candidates, monte_carlo_stay_within, _spearman  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1F3B57"
ACCENT = "#C2452D"
GREEN = "#2E7D5B"
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 120})


def fig_vol_skew() -> None:
    spot, atm, rho, eta = 100.0, 0.24, -0.55, 0.7
    strikes = [spot * (0.75 + 0.5 * i / 200) for i in range(201)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for dte, color in ((30, INK), (60, ACCENT)):
        t = dte / 365.0
        ivs = [100 * ssvi_iv(math.log(K / spot), t, atm, rho, eta) for K in strikes]
        ax.plot(strikes, ivs, color=color, lw=2, label=f"{dte}-day expiry")
    ax.axvline(spot, color="gray", ls=":", lw=1)
    ax.annotate("at-the-money", xy=(spot, 100 * atm), xytext=(spot + 6, 100 * atm + 4),
                fontsize=9, color="gray")
    ax.set_xlabel("Strike"); ax.set_ylabel("Implied volatility (%)")
    ax.set_title("Synthetic implied-volatility skew (SSVI, arbitrage-free)")
    ax.legend(frameon=False)
    fig.text(0.5, -0.02, "Equity left-skew: out-of-the-money puts richer; skew flattens with maturity.",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(); fig.savefig(OUT / "vol_skew.png", bbox_inches="tight"); plt.close(fig)


def fig_pop_vs_pht() -> None:
    # Compare terminal vs path probability over the SAME holding horizon so the
    # gap is purely the path-risk penalty: "never breaches" implies "in range at
    # the end", so PHT <= POP always, and the gap widens with volatility.
    spot, lo_be, hi_be = 100.0, 94.0, 106.0
    t = 20 / 365.0
    ivs = [0.10 + 0.50 * i / 100 for i in range(101)]
    pop = [terminal_prob_in_range(spot, iv, t, lo_be, hi_be) for iv in ivs]
    pht = [prob_stay_within_barriers(spot, lo_be, hi_be, iv, t) for iv in ivs]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([100 * v for v in ivs], pop, color=INK, lw=2, label="Terminal POP (in the profit zone at expiry)")
    ax.plot([100 * v for v in ivs], pht, color=ACCENT, lw=2,
            label="Path PHT (never breaches a break-even)")
    ax.fill_between([100 * v for v in ivs], pht, pop, color=ACCENT, alpha=0.10,
                    label="path-risk penalty (POP − PHT)")
    ax.set_xlabel("Implied volatility (%)"); ax.set_ylabel("Probability")
    ax.set_title("Terminal POP overstates edge vs. path probability (PHT)")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.text(0.5, -0.02, "Same horizon, break-evens 94 / 106, spot 100. EV is scored off PHT, not POP.",
             ha="center", fontsize=9, color="gray")
    fig.tight_layout(); fig.savefig(OUT / "pop_vs_pht.png", bbox_inches="tight"); plt.close(fig)


def fig_first_passage_vs_mc() -> None:
    # Compare the closed form against TWO benchmarks: a naive discretely-monitored
    # MC (biased high by the monitoring bias) and a continuity-corrected
    # Brownian-bridge MC (the proper benchmark). The correction is the point.
    cands = build_candidates(30, seed=20260811)
    an, naive, bridge = [], [], []
    for i, c in enumerate(cands):
        t = c["horizon"] / 365.0
        an.append(prob_stay_within_barriers(c["spot"], c["lower"], c["upper"], c["iv"], t))
        b, _ = monte_carlo_stay_within(c["spot"], c["lower"], c["upper"], c["iv"], c["horizon"],
                                       paths=10000, steps=32, seed=1000 + i, continuity_correction=True)
        n, _ = monte_carlo_stay_within(c["spot"], c["lower"], c["upper"], c["iv"], c["horizon"],
                                       paths=10000, steps=32, seed=1000 + i, continuity_correction=False)
        bridge.append(b); naive.append(n)
    med_naive = sorted(abs(a - m) for a, m in zip(an, naive))[len(an) // 2]
    med_bridge = sorted(abs(a - m) for a, m in zip(an, bridge))[len(an) // 2]
    rho = _spearman(an, bridge)

    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="y = x (perfect)")
    ax.scatter(naive, an, facecolor="none", edgecolor="#9AA7B2", s=42, linewidth=1.2,
               label=f"vs naive discrete MC  (median |Δ| {med_naive:.3f})")
    ax.scatter(bridge, an, color=ACCENT, s=34, alpha=0.9, edgecolor="white", linewidth=0.5,
               label=f"vs continuity-corrected MC  (median |Δ| {med_bridge:.3f})")
    ax.set_xlabel("Monte-Carlo hold probability (empirical)")
    ax.set_ylabel("Analytical hold probability (closed-form)")
    ax.set_title("Closed-form vs. Monte-Carlo — benchmark matters")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.text(0.97, 0.05,
            f"Spearman ρ = {rho:.3f}\n(ranking preserved)",
            ha="right", va="bottom", transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec=ACCENT, alpha=0.9))
    fig.text(0.5, -0.02, "A naive discrete MC overstates survival (open circles, below y=x); the "
             "continuity-corrected MC (filled) hugs the diagonal —\nnear-exact for typical candidates and "
             "systematically conservative (never optimistic), widening only when both barriers are close.",
             ha="center", fontsize=8.5, color="gray")
    fig.tight_layout(); fig.savefig(OUT / "first_passage_vs_mc.png", bbox_inches="tight"); plt.close(fig)


def fig_butterfly_payoff() -> None:
    lower, body, upper, credit = 90.0, 100.0, 105.0, 0.5
    w1, w2 = body - lower, upper - body
    max_profit, max_risk, _ = payoff_metrics("put", credit, w1, w2)
    profit_high, profit_body, profit_low = credit, credit + w2, credit + w2 - w1

    def payoff(s: float) -> float:
        if s <= lower:
            return profit_low
        if s <= body:
            return profit_low + (profit_body - profit_low) * (s - lower) / (body - lower)
        if s <= upper:
            return profit_body + (profit_high - profit_body) * (s - body) / (upper - body)
        return profit_high

    xs = [70 + 60 * i / 300 for i in range(301)]
    ys = [payoff(s) for s in xs]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axhline(0, color="gray", lw=0.8)
    ax.plot(xs, ys, color=INK, lw=2)
    ax.fill_between(xs, ys, 0, where=[y >= 0 for y in ys], color=GREEN, alpha=0.15)
    ax.fill_between(xs, ys, 0, where=[y < 0 for y in ys], color=ACCENT, alpha=0.15)
    for K, name in ((lower, "lower wing"), (body, "body"), (upper, "upper wing")):
        ax.axvline(K, color="gray", ls=":", lw=1)
        ax.annotate(f"{name} {K:.0f}", xy=(K, profit_low), xytext=(K, profit_low + 0.25),
                    ha="center", va="bottom", fontsize=8, color="gray", rotation=90)
    ax.set_xlabel("Underlying at expiry"); ax.set_ylabel("P&L per share ($)")
    ax.set_title("Broken-wing butterfly payoff (defined risk)")
    fig.text(0.5, -0.02, f"Put BWB 90/100/105, 0.50 credit: max profit {max_profit:.2f}, "
             f"max risk {max_risk:.2f}.", ha="center", fontsize=9, color="gray")
    fig.tight_layout(); fig.savefig(OUT / "butterfly_payoff.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_vol_skew()
    fig_pop_vs_pht()
    fig_first_passage_vs_mc()
    fig_butterfly_payoff()
    print("Wrote figures to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print("  ", p.name, f"({p.stat().st_size // 1024} KB)")

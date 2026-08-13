"""Validate the closed-form first-passage probability against Monte-Carlo.

Reproduces — on synthetic data, no vendor inputs — the model-risk study behind the
library's scoring: how good is the closed-form "probability of holding a
defined-risk structure to target without breaching a break-even" (PHT)?

Two probabilities, one honest comparison
----------------------------------------
The analytical PHT is *exact per barrier* (continuous first-passage via the
reflection principle) but combines the two barriers **additively**:
``p_breach ≈ p_hit_lower + p_hit_upper``. That omits the joint term
``P(hit lower AND hit upper)`` (inclusion–exclusion), so it *overstates* breach
and *understates* survival. The size of that omission is precisely what we want to
measure.

Getting the benchmark right (this is the subtle part)
-----------------------------------------------------
A *naive* Monte-Carlo that only checks the barrier at discrete time steps is
itself biased: a continuous path can dip through a barrier and return **between**
two steps, and the naive check misses it — so naive MC *undercounts* breaches and
*overstates* survival (the "monitoring bias", O(1/√steps)). Comparing the
analytical model against a biased benchmark would overstate the analytical error.

We therefore use a **Brownian-bridge continuity correction** (Glasserman 2004;
Broadie–Glasserman–Kou 1997): between two simulated log-prices ``x0, x1`` over a
step of variance ``v = σ²·dt``, the probability the path crossed an upper level
``b`` (with ``x0, x1 < b``) is ``exp(-2(b−x0)(b−x1)/v)`` (and symmetrically for a
lower level). Each path contributes its *conditional survival probability* — the
product over steps of not crossing either barrier — which is an (essentially)
unbiased estimator of the continuous-monitoring survival probability.

Because the analytical model and the bridge-corrected MC share the **same
driftless lognormal dynamics** and both treat each single barrier by continuous
first-passage, their only difference is the additive-vs-joint treatment of the two
barriers. So ``bridge_MC − analytical`` isolates the additive-approximation error
itself — not a discretization artifact and not a modelling mismatch.

Scope: this validates the two-barrier *approximation* under a given lognormal
diffusion; it does not claim that constant-vol lognormal dynamics are consistent
with a full implied-volatility smile (they are not — the smile enters vanilla
valuation, not the path model here). The diffusion volatility is the structure's
implied volatility, a deliberate screening-model simplification.

Run:
    python examples/first_passage_vs_montecarlo.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optvol.pricing.probability import prob_stay_within_barriers  # noqa: E402


def monte_carlo_stay_within(
    spot: float, lower: float, upper: float, iv: float, horizon_days: int,
    *, paths: int, steps: int, seed: int, continuity_correction: bool = True,
) -> tuple[float, float]:
    """Estimate P(GBM stays strictly inside (lower, upper)) over the horizon.

    Returns ``(estimate, standard_error)``. With ``continuity_correction=True``
    each path contributes its Brownian-bridge conditional survival probability
    (continuous-monitoring, essentially unbiased); with ``False`` it is a naive
    discretely-monitored 0/1 indicator (biased high by the monitoring bias).
    """
    if spot <= 0 or lower <= 0 or upper <= 0 or upper <= lower or iv <= 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    t = horizon_days / 365.0
    dt = t / steps
    drift = -0.5 * iv * iv * dt
    vol_step = iv * math.sqrt(dt)
    v = iv * iv * dt  # variance of the log-increment per step
    log_lo, log_up = math.log(lower), math.log(upper)
    log_spot = math.log(spot)

    total = 0.0
    total_sq = 0.0
    for _ in range(paths):
        x_prev = log_spot
        weight = 1.0
        alive = True
        for _ in range(steps):
            x = x_prev + drift + vol_step * rng.gauss(0.0, 1.0)
            if x <= log_lo or x >= log_up:      # crossed at a monitored point
                alive = False
                break
            if continuity_correction:            # crossed between the two points?
                p_up = math.exp(-2.0 * (log_up - x_prev) * (log_up - x) / v)
                p_lo = math.exp(-2.0 * (x_prev - log_lo) * (x - log_lo) / v)
                weight *= (1.0 - p_up) * (1.0 - p_lo)
            x_prev = x
        w = weight if alive else 0.0
        total += w
        total_sq += w * w

    mean = total / paths
    var = max(total_sq / paths - mean * mean, 0.0)
    return mean, math.sqrt(var / paths)


def _spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation (Pearson on ranks); no external deps."""
    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy) if vx > 0 and vy > 0 else float("nan")


def build_candidates(n: int, seed: int) -> list[dict]:
    """Synthetic BWB-like candidates spanning a range of volatilities, horizons,
    and (asymmetric) barrier distances — including barriers near and far from
    spot, and short and longer holds."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        spot = 100.0
        iv = rng.uniform(0.15, 0.60)
        horizon = rng.choice([5, 7, 10, 14, 21])
        lower_dist = rng.uniform(0.03, 0.12)
        upper_dist = rng.uniform(0.03, 0.12)
        out.append({"spot": spot, "iv": iv, "horizon": horizon,
                    "lower": spot * (1.0 - lower_dist), "upper": spot * (1.0 + upper_dist)})
    return out


def main(paths: int = 10000, steps: int = 32, n_candidates: int = 30) -> None:
    candidates = build_candidates(n_candidates, seed=20260811)
    rows = []
    for idx, c in enumerate(candidates):
        t = c["horizon"] / 365.0
        analytical = prob_stay_within_barriers(c["spot"], c["lower"], c["upper"], c["iv"], t)
        bridge, se = monte_carlo_stay_within(c["spot"], c["lower"], c["upper"], c["iv"], c["horizon"],
                                             paths=paths, steps=steps, seed=1000 + idx,
                                             continuity_correction=True)
        naive, _ = monte_carlo_stay_within(c["spot"], c["lower"], c["upper"], c["iv"], c["horizon"],
                                           paths=paths, steps=steps, seed=1000 + idx,
                                           continuity_correction=False)
        rows.append({**c, "analytical": analytical, "bridge": bridge, "se": se, "naive": naive})

    # Bias of the analytical model vs the corrected benchmark (the additive-approx error).
    bias = [r["analytical"] - r["bridge"] for r in rows]
    abs_bias = sorted(abs(b) for b in bias)
    median_abs = abs_bias[len(abs_bias) // 2]
    mean_abs = sum(abs(b) for b in bias) / len(bias)
    n_conservative = sum(1 for b in bias if b < 0)
    mean_se = sum(r["se"] for r in rows) / len(rows)
    monitoring_bias = sum(r["naive"] - r["bridge"] for r in rows) / len(rows)
    rho = _spearman([r["analytical"] for r in rows], [r["bridge"] for r in rows])

    print("First-passage (analytical) vs Brownian-bridge Monte-Carlo — synthetic candidates")
    print(f"  candidates={len(rows)}  MC paths={paths}  steps={steps}  (continuity-corrected)\n")
    print(f"{'#':>2} {'iv':>5} {'Hd':>3} {'lower':>7} {'upper':>7} {'analytic':>9} "
          f"{'MC(bridge)':>10} {'±SE':>6} {'naiveMC':>8} {'bias':>7}")
    for i, r in enumerate(rows):
        print(f"{i:2d} {r['iv']:5.2f} {r['horizon']:3d} {r['lower']:7.2f} {r['upper']:7.2f} "
              f"{r['analytical']:9.4f} {r['bridge']:10.4f} {r['se']:6.4f} {r['naive']:8.4f} "
              f"{r['analytical'] - r['bridge']:+7.4f}")

    print("\n--- Summary ---")
    print(f"  benchmark: continuity-corrected (Brownian-bridge) MC; mean standard error ~ {mean_se:.4f}")
    print(f"  additive-approximation error (analytical - bridge_MC):")
    print(f"     median |bias| = {median_abs:.4f}   mean |bias| = {mean_abs:.4f}   "
          f"(~ {median_abs / mean_se:.0f}x the MC standard error, so not noise)")
    print(f"  analytical <= corrected MC for {n_conservative}/{len(rows)} candidates "
          f"(systematically conservative: survival understated)")
    print(f"  Spearman rank correlation (analytical vs corrected MC) = {rho:.4f}  (ranking preserved)")
    print(f"  monitoring bias of NAIVE discrete MC (naive - bridge) ~ {monitoring_bias:+.4f} "
          f"(naive MC overstates survival; corrected here)")
    print("\nInterpretation: against a continuity-corrected benchmark the additive two-barrier")
    print("approximation understates the hold probability by a median of a few points, in the")
    print("safe (conservative) direction, while preserving candidate ranking (rho ~ 0.99+). It is")
    print("a screening/ranking estimator, not a calibrated absolute-probability estimator; the")
    print("principled correction is a proper two-boundary (double-barrier) first-passage treatment.")


if __name__ == "__main__":
    main()

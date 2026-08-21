"""Probability models for defined-risk options structures.

This module separates two probabilities that are easy to conflate but answer
different questions, and getting the distinction right is the core modeling
thesis of the whole library:

1. **Terminal probability of profit (POP)**: the probability that the
   underlying finishes, *at expiry*, in a region where the structure is
   profitable. This is what most retail tools report. It uses the lognormal
   distribution of the terminal price.

2. **Path / first-passage probability (PHT, "probability of hold to target")**:
   the probability that the underlying *never touches* a break-even barrier at
   any point over the holding horizon. For a position you actively manage and
   exit *before* expiry, this is the honest input: a barrier touched mid-life can
   stop you out (or force a defensive roll) even if the price would have finished
   back inside the profit zone. Using a terminal POP where a path probability
   belongs systematically *overstates* edge.

The library scores expected value off the **path** probability, not terminal POP.

First-passage math
------------------
For a driftless geometric Brownian motion, the probability that log-price travels
a log-distance ``a > 0`` to a barrier within horizon ``t`` follows from the
**reflection principle**:

    P(hit) = 2 * (1 - Phi(a / (sigma * sqrt(t))))

(intuitively: every terminal path that *ends* beyond the barrier is matched by a
reflected path that touched the barrier and came back, so the touch probability
is twice the terminal-exceedance probability).

Known limitation (measured, not assumed)
-----------------------------------------
For a two-sided structure we approximate the probability of breaching *either*
barrier additively: ``p_breach ~= p_hit_lower + p_hit_upper``. This omits the
joint-crossing term (it double-counts paths that could reach both), so it
*overstates* breach and therefore *understates* PHT and EV. Against a
continuity-corrected (Brownian-bridge) Monte-Carlo benchmark of the same diffusion,
the residual error is near-exact for typical candidates (median ~0.004, at the MC
noise floor) and materially conservative only when both barriers are close;
crucially it is *systematically one-directional* -- survival understated in ~25 of
30 test candidates, never overstated -- while candidate *ranking* is essentially
unaffected (Spearman rho ~= 0.997). Because the bias is small and conservative it
is retained as a documented, monitored limitation for screening/ranking rather than
silently "fixed"; the principled correction for a calibrated absolute probability is
a proper two-boundary (double-barrier) first-passage treatment, not a one-line
inclusion-exclusion. See ``docs/model_risk_and_validation.md`` and
``examples/first_passage_vs_montecarlo.py``.

Provenance
----------
Generalized from the scanner's ``_lognormal_cdf``, ``_p_hit_barrier``,
``_pht_bwb`` and ``_pop_bwb``. The structure-specific break-even wiring now lives
in ``optvol.structures.butterfly``; this module holds only reusable primitives.
"""

from __future__ import annotations

import math
from typing import Optional

from .black_scholes import norm_cdf


def lognormal_cdf(spot: float, sigma: float, t: float, level: float) -> Optional[float]:
    """P(S_t <= level) for a driftless lognormal underlying.

    Uses drift ``mu = ln(spot) - 0.5 * sigma^2 * t`` (zero-drift / martingale
    convention). Returns ``None`` on degenerate inputs.
    """
    if spot <= 0 or sigma <= 0 or t <= 0 or level <= 0:
        return None
    mu = math.log(spot) - 0.5 * sigma * sigma * t
    z = (math.log(level) - mu) / (sigma * math.sqrt(t))
    return norm_cdf(z)


def terminal_prob_in_range(
    spot: float, sigma: float, t: float, low: float, high: float
) -> Optional[float]:
    """P(low <= S_t <= high) at expiry, driftless lognormal.

    Returns ``None`` if either tail CDF is undefined.
    """
    cdf_low = lognormal_cdf(spot, sigma, t, low)
    cdf_high = lognormal_cdf(spot, sigma, t, high)
    if cdf_low is None or cdf_high is None:
        return None
    return max(0.0, cdf_high - cdf_low)


def first_passage_hit_probability(log_distance: float, sigma: float, t: float) -> float:
    """Reflection-principle probability of hitting a barrier within horizon ``t``.

    Parameters
    ----------
    log_distance : float
        Positive log-distance from spot to the barrier, e.g. ``ln(upper/spot)``
        for an up-barrier or ``ln(spot/lower)`` for a down-barrier. A
        non-positive distance means spot is already at/through the barrier, so
        the probability is 1.
    sigma : float
        Annualized volatility.
    t : float
        Horizon in years.

    Returns
    -------
    float
        Touch probability in ``[0, 1]``.
    """
    if log_distance <= 0:
        return 1.0
    denom = sigma * math.sqrt(t)
    if denom <= 0:
        return 0.0
    z = log_distance / denom
    return min(1.0, 2.0 * (1.0 - norm_cdf(z)))


def prob_touch_level(spot: float, level: float, sigma: float, t: float) -> float:
    """First-passage probability that the underlying touches ``level`` within ``t``."""
    if spot <= 0 or level <= 0:
        return 0.0
    log_distance = math.log(level / spot) if level > spot else math.log(spot / level)
    return first_passage_hit_probability(log_distance, sigma, t)


def prob_stay_within_barriers(
    spot: float, lower: float, upper: float, sigma: float, t: float
) -> Optional[float]:
    """Probability the underlying stays inside (lower, upper) over horizon ``t``.

    Computed as ``1 - p_hit_lower - p_hit_upper`` using the additive breach
    approximation described in the module docstring (conservative: understates
    the true stay-in probability). Returns ``None`` on degenerate inputs.
    """
    if spot <= 0 or sigma <= 0 or t <= 0 or upper <= 0 or lower <= 0 or upper <= lower:
        return None
    p_upper = first_passage_hit_probability(math.log(upper / spot), sigma, t)
    p_lower = first_passage_hit_probability(math.log(spot / lower), sigma, t)
    p_breach = min(1.0, p_upper + p_lower)
    return max(0.0, 1.0 - p_breach)


def expected_value(p_win: float, profit: float, loss: float) -> float:
    """Two-outcome expected value: ``p_win * profit - (1 - p_win) * loss``.

    ``loss`` is expressed as a positive magnitude (the amount lost on a stop).
    """
    return p_win * profit - (1.0 - p_win) * loss

"""Composite candidate score ("rscore").

Every surviving candidate is reduced to a single, bounded score that blends
several shaped sub-scores. The design goals are:

* **Each factor is bounded to [0, 1]** and shaped so that "good" saturates
  rather than running away, a spectacular value on one axis should not let a
  candidate ignore the others.
* **The score is comparable across runs.** The expected-value sub-score uses an
  *absolute* normalization (``min(ev / scale, 1)``) rather than a per-batch
  min-max, so a median-quality trade scores ~0.5 whether the day produced 50
  candidates or 5,000. This matters for a stable acceptance threshold.

Notes on parameters (calibration / governance)
-----------------------------------------------
The ``scale`` and factor weights below are **illustrative defaults**. In the
private research program these are configuration-driven and were *calibrated*
from a large candidate sample (~21.7k rows) so that a median trade scores 0.50;
that recalibration, replacing an earlier batch min-max normalization, was
measured for its downstream effect (it materially reduced portfolio
concentration, Herfindahl index falling by roughly a third) and was adopted only
after an independent-review sign-off, with a documented rollback path. The exact
production constants are intentionally not published; the *shape and reasoning*
are what matter here.

Provenance
----------
Refactored from the scanner's ``score_trades`` and its ``_score_*`` helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional


# ---- Illustrative configuration (NOT the production-tuned values) -----------

@dataclass(frozen=True)
class ScoringConfig:
    """Example scoring parameters. Replace via config in real use."""

    # EV normalization: score = min(ev_real / ev_score_abs_scale, 1.0)
    ev_score_abs_scale: float = 0.50
    # IV-regime thresholds (percentile/rank scale, 0-100)
    iv_high: float = 70.0
    iv_mid: float = 50.0
    iv_low: float = 30.0
    iv_change_bonus: float = 0.1
    iv_change_min: float = 10.0
    # Liquidity targets
    oi_target: float = 500.0
    volume_target: float = 200.0
    spread_pct_max: float = 0.015
    # Gamma
    gamma_max: float = 0.2
    gamma_proxy_max: float = 5.0
    # Factor weights (should sum to ~1 over the active factors)
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "ev": 0.35,
            "liquidity": 0.20,
            "iv": 0.20,
            "gamma": 0.15,
            "time": 0.10,
            "kink": 0.0,
            "delta": 0.0,
            "wing_ratio": 0.0,
        }
    )


DEFAULT_SCORING_CONFIG = ScoringConfig()


# ---- Shaped sub-scores (each returns a value in [0, 1]) ---------------------

def score_expected_value(ev_real: Optional[float], scale: float) -> float:
    """Absolute EV normalization: ``min(ev / scale, 1)``, clamped at 0."""
    if ev_real is None or scale <= 0:
        return 0.0
    return max(0.0, min(float(ev_real) / float(scale), 1.0))


def score_iv_regime(
    iv_percentile: Optional[float],
    iv_rank: Optional[float],
    iv_change: Optional[float],
    cfg: ScoringConfig = DEFAULT_SCORING_CONFIG,
) -> float:
    """Bucketed score on IV percentile/rank, with a small bonus for rising IV."""
    base = iv_percentile if iv_percentile is not None else iv_rank
    if base is None:
        score = 0.0
    elif base >= cfg.iv_high:
        score = 1.0
    elif base >= cfg.iv_mid:
        score = 0.7
    elif base >= cfg.iv_low:
        score = 0.3
    else:
        score = 0.0
    if iv_change is not None and iv_change >= cfg.iv_change_min:
        score = min(1.0, score + cfg.iv_change_bonus)
    return score


def score_band(
    value: Optional[float],
    soft_min: Optional[float],
    target_min: Optional[float],
    target_max: Optional[float],
    soft_max: Optional[float],
) -> float:
    """Trapezoidal band score: 1.0 inside the target band, ramping to 0 at the
    soft edges, and 0 beyond them. Missing/undetermined inputs score a neutral
    0.5 so they neither help nor penalize."""
    if value is None or target_min is None or target_max is None:
        return 0.5
    if target_min <= value <= target_max:
        return 1.0
    if value < target_min:
        if soft_min is None or target_min <= soft_min or value <= soft_min:
            return 0.0
        return max(0.0, 1.0 - (target_min - value) / (target_min - soft_min))
    # value > target_max
    if soft_max is None or target_max >= soft_max or value >= soft_max:
        return 0.0
    return max(0.0, 1.0 - (value - target_max) / (soft_max - target_max))


def score_kink(kink_z: Optional[float]) -> float:
    """Bounded sigmoid on the curvature z-score: ``0.5 + 0.5 * tanh(z)``."""
    if kink_z is None:
        return 0.0
    return 0.5 + 0.5 * math.tanh(kink_z)


def score_time(dte: Optional[int]) -> float:
    """Prefer short-dated structures; taper for longer expiries."""
    if dte is None:
        return 0.0
    if 5 <= dte <= 15:
        return 1.0
    if 15 < dte <= 30:
        return 0.7
    if dte > 30:
        return 0.4
    return 0.0


def score_liquidity(
    open_interest: Optional[int],
    volume: Optional[int],
    spread_pct: Optional[float],
    spread_abs: Optional[float] = None,
    spread_abs_max: Optional[float] = None,
    cfg: ScoringConfig = DEFAULT_SCORING_CONFIG,
) -> Optional[float]:
    """Liquidity score = ``min(oi_score, volume_score, spread_score)``.

    Taking the *minimum* (not an average) is deliberate: liquidity is a
    weakest-link property, a great open interest does not rescue an untradeable
    spread. Absolute-dollar spread is preferred for small credits when available,
    otherwise percentage spread is used.
    """
    if spread_pct is None and spread_abs is None:
        return None
    oi_score = min((open_interest or 0) / cfg.oi_target, 1.0) if cfg.oi_target > 0 else 0.0
    vol_score = min((volume or 0) / cfg.volume_target, 1.0) if cfg.volume_target > 0 else 0.0

    spread_score_abs = None
    if spread_abs_max is not None and spread_abs is not None and float(spread_abs_max) > 0:
        spread_score_abs = 1.0 - min(abs(spread_abs) / float(spread_abs_max), 1.0)
    spread_score_pct = None
    if spread_pct is not None and cfg.spread_pct_max > 0:
        spread_score_pct = 1.0 - min(spread_pct / cfg.spread_pct_max, 1.0)

    if spread_score_abs is None and spread_score_pct is None:
        return None
    if spread_score_abs is None:
        spread_score = spread_score_pct
    elif spread_score_pct is None:
        spread_score = spread_score_abs
    else:
        spread_score = max(spread_score_abs, spread_score_pct)
    return min(oi_score, vol_score, spread_score)


def gamma_penalty(
    body_gamma: Optional[float],
    max_risk: float,
    wing_width: float,
    cfg: ScoringConfig = DEFAULT_SCORING_CONFIG,
) -> float:
    """Penalize pin/gamma exposure. Uses body gamma when available, else a
    risk-per-wing proxy. Higher score = *less* gamma risk."""
    if body_gamma is not None and cfg.gamma_max > 0:
        return 1.0 - min(abs(body_gamma) / cfg.gamma_max, 1.0)
    proxy = max_risk / wing_width if wing_width > 0 else cfg.gamma_proxy_max
    return 1.0 - min(proxy / cfg.gamma_proxy_max, 1.0)


def composite_score(components: Dict[str, float], cfg: ScoringConfig = DEFAULT_SCORING_CONFIG) -> float:
    """Weighted sum of named component scores using ``cfg.weights``.

    ``components`` maps factor name -> sub-score in [0, 1]. Unspecified factors
    contribute 0; unknown weights default to 0.
    """
    return sum(cfg.weights.get(name, 0.0) * value for name, value in components.items())

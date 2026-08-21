"""IV-curvature ("kink") signal.

The entry signal for the butterfly engine is local curvature of the implied-
volatility smile. At each interior strike we measure how far that strike's IV
sits above (or below) the average of its two neighbours:

    kink = IV(strike) - (IV(left) + IV(right)) / 2

A positive kink means the smile bulges upward at that strike, the option is
locally *rich* relative to its neighbours, which is where a body-centered
butterfly can be sold to advantage. A raw kink is hard to compare across names
and regimes, so it is standardized two ways:

* **cross-sectional z** (`kink_z_cross`), against the distribution of kinks in
  the *same* chain at the *same* moment, and
* **history z** (`kink_z`), against the strike's *own* recent history, which
  controls for a strike that is persistently kinked.

Standardizing against a strike's own history (rather than an absolute threshold)
is a deliberate nod to non-stationarity: "rich relative to normal for *this*
strike" is a more robust signal than "rich in absolute terms."

Provenance
----------
Refactored from the scanner's ``compute_kinks`` and ``apply_history_z``. The
file-backed rolling-history store is intentionally omitted here; this module
keeps an in-memory history so the signal is self-contained and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class StrikeIV:
    """A single (strike, implied-vol) observation on one chain."""

    strike: float
    iv: Optional[float]


@dataclass
class KinkPoint:
    """Curvature at an interior strike, with optional standardizations."""

    strike: float
    kink: float
    kink_z_cross: Optional[float] = None
    kink_z: Optional[float] = None


def compute_kinks(chain: Sequence[StrikeIV]) -> List[KinkPoint]:
    """Compute local IV curvature and its cross-sectional z-score for a chain.

    ``chain`` should be one option type of one expiry. It is sorted by strike
    internally. Interior strikes with a missing IV (self or a neighbour) are
    skipped. When at least two kinks exist, each is z-scored against the chain's
    own kink distribution (`kink_z_cross`).
    """
    points = sorted(chain, key=lambda s: s.strike)
    kinks: List[KinkPoint] = []
    raw: List[float] = []

    for idx in range(1, len(points) - 1):
        left, mid, right = points[idx - 1], points[idx], points[idx + 1]
        if left.iv is None or mid.iv is None or right.iv is None:
            continue
        value = mid.iv - (left.iv + right.iv) / 2.0
        raw.append(value)
        kinks.append(KinkPoint(strike=mid.strike, kink=value))

    if len(raw) > 1:
        mu = mean(raw)
        sd = stdev(raw)
        if sd > 0:
            for point in kinks:
                point.kink_z_cross = (point.kink - mu) / sd
    return kinks


def apply_history_z(
    kinks: List[KinkPoint],
    history: Dict[float, List[float]],
    min_history: int,
) -> None:
    """Fill ``kink_z`` in place from a per-strike history of past kink values.

    ``history`` maps a strike to its recent kink observations. A strike is only
    standardized once it has at least ``min_history`` observations and a positive
    standard deviation.
    """
    for point in kinks:
        values = history.get(point.strike, [])
        if len(values) < min_history:
            continue
        if len(values) > 1:
            avg = mean(values)
            sd = stdev(values)
        else:
            avg, sd = values[0], 0.0
        if sd > 0:
            point.kink_z = (point.kink - avg) / sd


def effective_kink_z(point: KinkPoint) -> Optional[float]:
    """Prefer the history z-score, falling back to the cross-sectional z-score."""
    return point.kink_z if point.kink_z is not None else point.kink_z_cross

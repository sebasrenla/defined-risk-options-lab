"""Broken-wing butterfly (BWB) structure: payoff, break-evens, and probabilities.

A broken-wing butterfly is a three-strike, defined-risk structure: a body at
``body_strike`` and two wings at ``lower_strike`` / ``upper_strike`` with
*unequal* wing widths (that asymmetry is what lets the structure be opened for a
credit while keeping one side's risk defined). This module computes, from the
three strikes and the net credit:

* the payoff metrics (max profit, max risk, whether the structure is "free-risk"),
* the break-even prices, and
* two probabilities — a **terminal** probability of profit (POP) and a
  **path** probability of holding to target without breaching a break-even (PHT) —
  and the resulting expected value.

The probability primitives live in :mod:`optvol.pricing.probability`; this module
only supplies the structure-specific geometry (break-evens, wing widths, and the
credit-vs-debit / call-vs-put payoff branches).

Provenance
----------
Refactored from the scanner's ``_bwb_metrics_call`` / ``_bwb_metrics_put``,
``_pop_bwb`` and ``_pht_bwb``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..pricing.probability import (
    expected_value,
    lognormal_cdf,
    prob_stay_within_barriers,
    terminal_prob_in_range,
)


def payoff_metrics(
    option_type: str, net_credit: float, w1: float, w2: float
) -> Tuple[float, float, bool]:
    """Max profit, max risk, and free-risk flag for a BWB.

    Parameters
    ----------
    option_type : str
        ``"call"`` or ``"put"``.
    net_credit : float
        Net credit received (negative for a debit) per unit.
    w1, w2 : float
        Lower wing width (``body - lower``) and upper wing width
        (``upper - body``).

    Returns
    -------
    (max_profit, max_risk, free_risk)
        ``max_risk`` is a non-negative magnitude. ``free_risk`` is ``True`` when
        the structure has no downside and was opened for a credit.
    """
    if option_type == "call":
        profit_low = net_credit
        profit_high = net_credit + w1 - w2
        max_profit = max(profit_low, net_credit + w1, profit_high)
        min_profit = min(profit_low, profit_high)
    else:  # put
        profit_high = net_credit
        profit_low = net_credit + w2 - w1
        max_profit = max(profit_high, net_credit + w2, profit_low)
        min_profit = min(profit_high, profit_low)

    max_risk = max(0.0, -min_profit)
    free_risk = max_risk == 0.0 and net_credit >= 0
    return max_profit, max_risk, free_risk


def break_evens(
    option_type: str, lower: float, body: float, upper: float, net_credit: float
) -> Tuple[float, float]:
    """Lower and upper break-even prices for a BWB."""
    if option_type == "call":
        lower_be = lower - net_credit
        upper_be = 2 * body - lower + net_credit
    else:  # put
        lower_be = 2 * body - upper - net_credit
        upper_be = upper + net_credit
    return lower_be, upper_be


@dataclass(frozen=True)
class BrokenWingButterfly:
    """A defined-risk broken-wing butterfly, with derived analytics.

    Only the four economic inputs are required (option type, the three strikes,
    and the net credit). All metrics are computed on demand.
    """

    option_type: str  # "call" or "put"
    lower_strike: float
    body_strike: float
    upper_strike: float
    net_credit: float

    @property
    def lower_wing(self) -> float:
        """Lower wing width (``body - lower``)."""
        return self.body_strike - self.lower_strike

    @property
    def upper_wing(self) -> float:
        """Upper wing width (``upper - body``)."""
        return self.upper_strike - self.body_strike

    def metrics(self) -> Tuple[float, float, bool]:
        """``(max_profit, max_risk, free_risk)`` — see :func:`payoff_metrics`."""
        return payoff_metrics(
            self.option_type, self.net_credit, self.lower_wing, self.upper_wing
        )

    def break_evens(self) -> Tuple[float, float]:
        """``(lower_be, upper_be)`` — see :func:`break_evens`."""
        return break_evens(
            self.option_type,
            self.lower_strike,
            self.body_strike,
            self.upper_strike,
            self.net_credit,
        )

    def probability_of_profit(
        self, spot: Optional[float], iv: Optional[float], dte: Optional[int]
    ) -> Optional[float]:
        """Terminal probability of profit (POP) under a driftless lognormal.

        Handles the four cases (call/put x credit/debit) explicitly, because the
        profit region differs: a credit BWB is profitable on a wide interval
        around the body, whereas a debit BWB profits only in a bounded window
        between break-evens.
        """
        if spot is None or iv is None or dte is None or spot <= 0 or iv <= 0:
            return None
        t = dte / 365.0
        w1, w2 = self.lower_wing, self.upper_wing
        lower_be, upper_be = self.break_evens()

        if self.option_type == "call":
            profit_high = self.net_credit + w1 - w2
            if self.net_credit >= 0:
                if profit_high >= 0:
                    return 1.0
                return lognormal_cdf(spot, iv, t, upper_be)
            # debit
            if profit_high >= 0:
                cdf = lognormal_cdf(spot, iv, t, lower_be)
                return (1.0 - cdf) if cdf is not None else None
            return terminal_prob_in_range(spot, iv, t, lower_be, upper_be)

        # put
        profit_low = self.net_credit + w2 - w1
        if self.net_credit >= 0:
            if profit_low >= 0:
                return 1.0
            cdf = lognormal_cdf(spot, iv, t, lower_be)
            return (1.0 - cdf) if cdf is not None else None
        # debit
        if profit_low >= 0:
            return lognormal_cdf(spot, iv, t, upper_be)
        return terminal_prob_in_range(spot, iv, t, lower_be, upper_be)

    def hold_probability(
        self,
        spot: Optional[float],
        iv: Optional[float],
        dte: Optional[int],
        exit_days: int,
    ) -> Optional[float]:
        """Path probability of holding to target without breaching a break-even.

        Uses the first-passage stay-within-barriers primitive over an effective
        horizon of ``min(dte - 1, exit_days)`` days — i.e. we assume the position
        is worked and closed before expiry rather than held to settlement.
        """
        if spot is None or iv is None or dte is None or spot <= 0 or iv <= 0:
            return None
        lower_be, upper_be = self.break_evens()
        if upper_be <= 0 or lower_be <= 0 or upper_be <= lower_be:
            return None
        horizon_days = min(max(dte - 1, 1), max(exit_days, 1))
        t = horizon_days / 365.0
        return prob_stay_within_barriers(spot, lower_be, upper_be, iv, t)

    def expected_value(
        self,
        spot: Optional[float],
        iv: Optional[float],
        dte: Optional[int],
        exit_days: int,
        profit_target_pct: float,
        stop_loss_pct: float,
    ) -> Optional[float]:
        """Expected value using the *path* hold-probability, not terminal POP.

        ``profit_target`` is ``max_profit * profit_target_pct`` and ``stop_loss``
        is ``max_risk * stop_loss_pct``; EV weights them by the hold probability.
        Returns ``None`` if the hold probability is undefined.
        """
        pht = self.hold_probability(spot, iv, dte, exit_days)
        if pht is None:
            return None
        max_profit, max_risk, _ = self.metrics()
        profit_target = max_profit * profit_target_pct
        stop_loss = max_risk * stop_loss_pct
        return expected_value(pht, profit_target, stop_loss)

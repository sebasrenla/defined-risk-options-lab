"""Black-Scholes pricing and implied-volatility inversion.

Provenance
----------
Refactored from the research engine's scanner module (`_norm_cdf`, `_bs_price`,
`infer_implied_volatility`). The math is unchanged; the API is cleaned up
(volatility inversion takes time-to-expiry in *years* rather than calendar days,
and keyword-only optionals make call sites explicit).

Design notes
------------
* We invert price -> implied volatility with **bisection** rather than
  Newton-Raphson. Bisection is slower but unconditionally stable: vega collapses
  for deep-ITM/OTM or near-expiry options, and a Newton step can then diverge.
  For a research pipeline that scans thousands of contracts of uneven quality,
  robustness matters more than the extra iterations.
* Before inverting we reject prices that violate the no-arbitrage bounds, so we
  never waste iterations on an un-invertible quote.
"""

from __future__ import annotations

import math
from typing import Optional

_SQRT_2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """Standard-normal cumulative distribution function, via ``math.erf``."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def black_scholes_price(
    spot: float,
    strike: float,
    t: float,
    vol: float,
    *,
    rate: float = 0.0,
    is_call: bool = True,
) -> float:
    """European option price under Black-Scholes.

    Parameters
    ----------
    spot, strike : float
        Underlying and strike prices.
    t : float
        Time to expiry in years.
    vol : float
        Annualized volatility (e.g. ``0.30`` for 30%).
    rate : float, keyword-only
        Continuously-compounded risk-free rate. Defaults to 0.0; for the short
        holding horizons this library targets, discounting is a second-order
        effect (see ``probability`` module and the Monte-Carlo drift study).
    is_call : bool, keyword-only
        ``True`` for a call, ``False`` for a put.

    Returns
    -------
    float
        The option's theoretical value. Degenerates to intrinsic value when
        ``t <= 0`` or ``vol <= 0``.
    """
    if t <= 0.0 or vol <= 0.0:
        intrinsic = (spot - strike) if is_call else (strike - spot)
        return max(0.0, intrinsic)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    if is_call:
        return spot * norm_cdf(d1) - strike * math.exp(-rate * t) * norm_cdf(d2)
    return strike * math.exp(-rate * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def implied_volatility(
    price: float,
    spot: float,
    strike: float,
    t: float,
    *,
    is_call: bool = True,
    rate: float = 0.0,
    min_vol: float = 1e-4,
    max_vol: float = 5.0,
    tol: float = 1e-5,
    max_iter: int = 100,
) -> Optional[float]:
    """Recover implied volatility from an option price by bisection.

    Returns ``None`` (rather than raising) when the price is outside the
    no-arbitrage bounds or otherwise not invertible, so callers can filter out
    bad quotes without exception handling on the hot path.

    Parameters
    ----------
    price : float
        Observed option price (mid, say).
    spot, strike, t : float
        Underlying, strike, and time to expiry in years.
    is_call, rate : keyword-only
        Option type and risk-free rate.
    min_vol, max_vol : float, keyword-only
        Bracketing volatilities for the bisection search.
    tol : float, keyword-only
        Absolute price tolerance for convergence.
    max_iter : int, keyword-only
        Maximum bisection iterations.
    """
    if price <= 0.0 or spot <= 0.0 or strike <= 0.0 or t <= 0.0:
        return None

    discount = math.exp(-rate * t)
    # No-arbitrage price bounds for a European option.
    if is_call:
        lower_bound = max(0.0, spot - strike * discount)
        upper_bound = spot
    else:
        lower_bound = max(0.0, strike * discount - spot)
        upper_bound = strike * discount
    if price < lower_bound - 1e-6 or price > upper_bound + 1e-6:
        return None

    low, high = min_vol, max_vol
    low_price = black_scholes_price(spot, strike, t, low, rate=rate, is_call=is_call)
    high_price = black_scholes_price(spot, strike, t, high, rate=rate, is_call=is_call)
    # If the target is not bracketed by [min_vol, max_vol] prices, give up.
    if price < low_price or price > high_price:
        return None

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = black_scholes_price(spot, strike, t, mid, rate=rate, is_call=is_call)
        diff = mid_price - price
        if abs(diff) <= tol:
            return mid
        if diff > 0:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)

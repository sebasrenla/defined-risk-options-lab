"""Generate a synthetic, arbitrage-free option chain (no vendor data).

The implied-volatility surface is built with the **SSVI** (Surface Stochastic
Volatility Inspired) parameterization of Gatheral & Jacquier, the industry/
academic standard for a realistic, *statically arbitrage-free* equity surface.
SSVI models the **total implied variance** ``w(k, θ) = σ(k)² · T`` as a function
of log-moneyness ``k = ln(K/S)``:

    w(k, θ) = (θ/2) · [ 1 + ρ·φ·k + sqrt((φ·k + ρ)² + (1 − ρ²)) ],   φ = η / √θ

where ``θ = σ_atm² · T`` is the at-the-money total variance. This gives:

* a realistic **equity skew** (``ρ < 0`` ⇒ out-of-the-money puts carry higher
  IV, the structural left-skew from downside/crash risk and put demand),
* **no calendar arbitrage**: with ``φ = η/√θ`` and ``θ`` increasing in maturity,
  total variance is non-decreasing in ``T``,
* **no butterfly arbitrage** for parameters satisfying the Gatheral–Jacquier
  conditions (``θφ(1+|ρ|) < 4`` and ``θφ²(1+|ρ|) ≤ 4``), which the defaults meet.

The generator is deliberately *neutral*: it does not tilt strikes, spreads, or
liquidity to flatter any strategy. Every option is priced from the SSVI IV with
the library's own Black–Scholes model; put–call parity holds by construction
(same IV per strike). :func:`check_static_arbitrage` verifies the surface is
free of butterfly and calendar arbitrage on a dense grid.

References: Gatheral & Jacquier, "Arbitrage-free SVI volatility surfaces" (2014).

Run directly to print a sample, an arbitrage report, and skew diagnostics:
    python examples/generate_synthetic_chain.py
"""

from __future__ import annotations

import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optvol.pricing import black_scholes_price, norm_cdf  # noqa: E402
from optvol.sleeves import OptionQuote  # noqa: E402


# --- SSVI implied-volatility surface -----------------------------------------

def ssvi_total_variance(k: float, theta: float, rho: float, eta: float) -> float:
    """SSVI total implied variance at log-moneyness ``k`` for ATM variance θ."""
    phi = eta / math.sqrt(max(theta, 1e-12))
    return 0.5 * theta * (1.0 + rho * phi * k + math.sqrt((phi * k + rho) ** 2 + (1.0 - rho * rho)))


def ssvi_iv(k: float, t: float, atm_vol: float, rho: float, eta: float) -> float:
    """Implied vol at log-moneyness ``k`` and maturity ``t`` (years) via SSVI."""
    theta = atm_vol * atm_vol * t
    w = ssvi_total_variance(k, theta, rho, eta)
    return math.sqrt(max(w, 1e-10) / t)


def _bs_delta(spot: float, strike: float, t: float, vol: float, is_call: bool) -> float:
    """Black-Scholes delta (rate = 0)."""
    if t <= 0 or vol <= 0:
        if is_call:
            return 1.0 if spot > strike else 0.0
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + 0.5 * vol * vol * t) / (vol * math.sqrt(t))
    return norm_cdf(d1) if is_call else norm_cdf(d1) - 1.0


def generate_synthetic_chain(
    *,
    symbol: str = "DEMO",
    spot: float = 100.0,
    as_of: date = date(2026, 6, 1),
    expiries_dte: tuple[int, ...] = (30, 60),
    strikes_per_side: int = 8,
    strike_step_pct: float = 0.025,
    atm_vol: float = 0.24,
    rho: float = -0.55,
    eta: float = 0.7,
    inject_kink_at_offset: int | None = None,
    kink_iv_bump: float = 0.03,
    seed: int = 7,
) -> list[OptionQuote]:
    """Build a reproducible, arbitrage-free synthetic option chain (SSVI surface).

    Parameters
    ----------
    symbol, spot, as_of : chain identity and underlying level/date.
    expiries_dte : days-to-expiry for each expiry.
    strikes_per_side, strike_step_pct : strike grid around spot.
    atm_vol : at-the-money volatility level (SSVI θ = atm_vol² · T).
    rho : SSVI skew (negative for equity left-skew).
    eta : SSVI wing parameter (curvature of the smile).
    inject_kink_at_offset : if set, add ``kink_iv_bump`` to IV at this strike
        offset, a *deliberate* local surface anomaly to exercise the curvature
        signal (this intentionally perturbs the otherwise arbitrage-free surface).
    seed : RNG seed for small quote/liquidity noise (reproducible).
    """
    rng = random.Random(seed)
    rows: list[OptionQuote] = []

    for dte in expiries_dte:
        expiry = as_of + timedelta(days=dte)
        t = dte / 365.0
        for offset in range(-strikes_per_side, strikes_per_side + 1):
            strike = round(spot * (1.0 + offset * strike_step_pct), 2)
            if strike <= 0:
                continue
            k = math.log(strike / spot)
            iv = ssvi_iv(k, t, atm_vol, rho, eta)
            if inject_kink_at_offset is not None and offset == inject_kink_at_offset:
                iv += kink_iv_bump

            for is_call in (True, False):
                price = black_scholes_price(spot, strike, t, iv, is_call=is_call)
                mid = max(0.02, price)
                rel_spread = 0.02 + 0.06 * min(abs(offset) / strikes_per_side, 1.0)
                half = max(0.08, mid * rel_spread * 0.5) * rng.uniform(0.9, 1.1)
                bid = round(max(0.01, mid - half), 2)
                ask = round(mid + half, 2)
                delta = round(_bs_delta(spot, strike, t, iv, is_call), 4)
                liq = math.exp(-((offset / (strikes_per_side * 0.6)) ** 2))
                open_interest = int(200 + 4000 * liq * rng.uniform(0.7, 1.3))
                volume = int(10 + 600 * liq * rng.uniform(0.6, 1.4))
                rows.append(OptionQuote(
                    symbol=symbol, expiry=expiry, dte=dte,
                    option_type="call" if is_call else "put", strike=strike,
                    bid=bid, ask=ask, delta=delta, open_interest=open_interest,
                    volume=volume, underlying_price=spot, iv=round(iv, 4)))
    return rows


# --- No-arbitrage verification (butterfly + calendar) ------------------------

def check_static_arbitrage(
    *, spot: float = 100.0, expiries_dte: tuple[int, ...] = (30, 60),
    atm_vol: float = 0.24, rho: float = -0.55, eta: float = 0.7,
    k_lo: float = 0.70, k_hi: float = 1.30, n: int = 241,
) -> dict:
    """Verify the SSVI surface is free of butterfly and calendar arbitrage.

    * **Butterfly** (no negative risk-neutral density): on a **uniform strike**
      grid over ``[k_lo·spot, k_hi·spot]`` at each maturity, the model call price
      must be non-increasing in strike and convex, equivalently the discrete
      second derivative ``(C[i-1] − 2C[i] + C[i+1]) / h²`` (∝ the risk-neutral
      density) must be ≥ 0. (A uniform K grid is required for the second
      difference to be a valid convexity estimate.)
    * **Calendar**: total implied variance ``w(k) = σ(k)²·T`` must be
      non-decreasing across maturities at every log-moneyness.
    """
    strikes = [spot * (k_lo + (k_hi - k_lo) * i / (n - 1)) for i in range(n)]
    h = strikes[1] - strikes[0]
    ks_cal = [math.log(K / spot) for K in strikes]
    density_tol = -1e-6  # a hair below zero to absorb float noise
    report: dict = {"butterfly_ok": True, "calendar_ok": True, "min_density": math.inf,
                    "max_call_slope": -math.inf, "details": []}

    prev_w = None
    for dte in expiries_dte:
        t = dte / 365.0
        ivs = [ssvi_iv(math.log(K / spot), t, atm_vol, rho, eta) for K in strikes]
        calls = [black_scholes_price(spot, K, t, v, is_call=True) for K, v in zip(strikes, ivs)]
        for i in range(1, n):
            slope = (calls[i] - calls[i - 1]) / h
            report["max_call_slope"] = max(report["max_call_slope"], slope)
            if slope > 1e-6:
                report["butterfly_ok"] = False
                report["details"].append(f"dte={dte}: call not monotone at K~{strikes[i]:.1f} (slope {slope:.2e})")
        for i in range(1, n - 1):
            dens = (calls[i - 1] - 2 * calls[i] + calls[i + 1]) / (h * h)
            report["min_density"] = min(report["min_density"], dens)
            if dens < density_tol:
                report["butterfly_ok"] = False
                report["details"].append(f"dte={dte}: negative density at K~{strikes[i]:.1f} ({dens:.2e})")
        w = [ssvi_total_variance(k, atm_vol * atm_vol * t, rho, eta) for k in ks_cal]
        if prev_w is not None:
            for i, (w0, w1) in enumerate(zip(prev_w, w)):
                if w1 - w0 < -1e-9:
                    report["calendar_ok"] = False
                    report["details"].append(f"calendar arb at k={ks_cal[i]:.3f}: w drops {w0:.4f}->{w1:.4f}")
        prev_w = w

    report["arbitrage_free"] = report["butterfly_ok"] and report["calendar_ok"]
    return report


def _skew_diagnostics(spot: float, dte: int, atm_vol: float, rho: float, eta: float) -> dict:
    """ATM vol and a skew measure (10% OTM put IV − 10% OTM call IV)."""
    t = dte / 365.0
    return {
        "atm_vol": ssvi_iv(0.0, t, atm_vol, rho, eta),
        "otm_put_iv_10pct": ssvi_iv(math.log(0.90), t, atm_vol, rho, eta),
        "otm_call_iv_10pct": ssvi_iv(math.log(1.10), t, atm_vol, rho, eta),
    }


if __name__ == "__main__":
    chain = generate_synthetic_chain(inject_kink_at_offset=2)
    puts = sorted((q for q in chain if q.option_type == "put" and q.dte == 30), key=lambda x: x.strike)
    print(f"Synthetic SSVI chain: {len(chain)} quotes; 30-DTE puts near the money:\n")
    print(f"{'strike':>8} {'iv':>7} {'delta':>8} {'bid':>7} {'ask':>7} {'OI':>7} {'vol':>6}")
    for q in puts:
        print(f"{q.strike:8.2f} {q.iv:7.4f} {q.delta:8.4f} {q.bid:7.2f} {q.ask:7.2f} {q.open_interest:7d} {q.volume:6d}")

    print("\n--- Static no-arbitrage check (clean surface, no injected kink) ---")
    rep = check_static_arbitrage()
    print(f"  arbitrage_free = {rep['arbitrage_free']}  (butterfly_ok={rep['butterfly_ok']}, "
          f"calendar_ok={rep['calendar_ok']})")
    print(f"  min risk-neutral density (>=0 required) = {rep['min_density']:.3e}")
    print(f"  max call dC/dK (<=0 required)           = {rep['max_call_slope']:.3e}")
    if rep["details"]:
        for d in rep["details"][:5]:
            print("   ", d)

    print("\n--- Skew diagnostics (realistic equity left-skew) ---")
    for dte in (30, 60):
        s = _skew_diagnostics(100.0, dte, 0.24, -0.55, 0.7)
        skew = s["otm_put_iv_10pct"] - s["otm_call_iv_10pct"]
        print(f"  {dte}d: ATM IV={s['atm_vol']:.4f}  10%-OTM put IV={s['otm_put_iv_10pct']:.4f}  "
              f"10%-OTM call IV={s['otm_call_iv_10pct']:.4f}  put-call skew={skew:+.4f}")

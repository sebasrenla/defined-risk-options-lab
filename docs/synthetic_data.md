# Synthetic Data

A model demo is only meaningful if the data it runs on resembles a real market —
neither understated nor tilted to flatter the strategy. This repository ships no
vendor data; instead it generates a **synthetic option-chain surface** that meets
the institutional bar for a realistic, arbitrage-free equity surface.

## The surface: SSVI

The implied-volatility surface is built with **SSVI** (Surface Stochastic
Volatility Inspired), the Gatheral–Jacquier parameterization that is the standard
for a statically arbitrage-free equity surface. SSVI models the **total implied
variance** `w(k) = σ(k)²·T` as a function of log-moneyness `k = ln(K/S)`:

    w(k, θ) = (θ/2)·[ 1 + ρ·φ·k + √((φ·k + ρ)² + (1 − ρ²)) ],   φ = η/√θ,
    θ = σ_atm²·T

with defaults chosen for a **representative single-name** underlying
(`σ_atm = 0.24`, `ρ = −0.55`, `η = 0.7`). Implied vol is `IV(k,T) = √(w/T)`, and
every option is then priced with the library's own Black-Scholes model, so
**put-call parity holds by construction**.

![Synthetic SSVI skew](figures/vol_skew.png)

## What makes it realistic

- **Equity left-skew.** `ρ < 0` produces the structural left-skew of equity
  options (out-of-the-money puts carry higher IV — the market's price for
  downside/crash risk and persistent put demand). At 30 days the surface shows a
  ≈ +11 vol-point skew between the 10%-OTM put and the 10%-OTM call, flattening to
  ≈ +9 at 60 days. That magnitude sits squarely in the observed single-name range
  (roughly 5–20 vol points at 30 days) — neither under- nor over-stated.
- **Correct term-structure behavior.** Skew flattens with maturity, as real
  surfaces do.
- **Realistic microstructure.** Bid/ask spreads widen away from the money and
  with lower price (with a small absolute floor — cheap out-of-the-money options
  genuinely have poor liquidity economics), and open interest / volume peak at
  the money.

## Verified arbitrage-free

`generate_synthetic_chain.check_static_arbitrage()` verifies, on a dense
**uniform-strike** grid at each maturity:

- **no butterfly arbitrage** — the model call price is non-increasing and convex
  in strike, i.e. the implied risk-neutral density `∂²C/∂K² ≥ 0`; and
- **no calendar arbitrage** — total variance is non-decreasing across maturities.

At the defaults both checks pass (minimum risk-neutral density is strictly
positive). One of the published tests asserts this, so a regression that made the
surface arbitrageable would fail. (An early version of the checker used a
log-moneyness grid, which makes the discrete second-derivative test invalid and
flagged spurious "arbitrage"; the fix — a uniform-strike grid — is the correct
way to test convexity numerically.)

## Neutrality

The generator is deliberately **neutral**. Strikes, spreads, and liquidity come
from the smooth surface and distance-based rules, not from any attempt to make a
strategy look good. The demo's sleeve candidates and the butterfly's expected
value are *outcomes* of the surface, not planted results. An optional local IV
"kink" can be injected at one strike to exercise the curvature signal — that is a
deliberate, labeled anomaly, and the arbitrage self-check is run on the *un-kinked*
surface.

## Honest scope

- **Flat ATM term structure** across the demo expiries (constant `σ_atm`). Real
  surfaces have a mild ATM term structure; omitting it does not affect the
  single-expiry demos and keeps the calendar-arbitrage check trivially satisfied.
- **Snapshot, not a path.** The chain is a single-moment surface — all the demos
  and tests need — not a simulated time series of surfaces.
- **Stylized microstructure.** Spreads and liquidity are realistic in *shape*,
  not calibrated to a specific name.

No real vendor chain was read or copied to build this; the skew magnitude was set
from publicly known single-name ranges and then verified for realism and
no-arbitrage.

**Reference:** Gatheral, J. & Jacquier, A., *Arbitrage-free SVI volatility
surfaces*, Quantitative Finance (2014).

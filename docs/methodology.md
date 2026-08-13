# Methodology

How the library reasons, module by module. This document reveals the *framework
and the why*; specific production-tuned thresholds and weights from the original
program are intentionally replaced by illustrative example values (the functions
are identical — only the constants differ).

---

## 1. Pricing and implied volatility

Options are priced with **Black-Scholes**. Implied volatility is recovered by
**bisection** rather than Newton-Raphson: vega collapses for deep-in/out-of-the-
money or near-expiry options, and a Newton step can then diverge, so for a
research pipeline that scans thousands of uneven-quality quotes, unconditional
stability matters more than a few extra iterations. Before inverting, a quote is
rejected if its price violates the no-arbitrage bounds, so no iterations are
wasted on an un-invertible quote.

## 2. Probability: terminal vs. path — the core thesis

Two different probabilities are easy to conflate:

- **Terminal probability of profit (POP)** — the chance the underlying finishes,
  *at expiry*, in a profitable region. Uses the lognormal distribution of the
  terminal price.
- **Path / first-passage probability (PHT)** — the chance the underlying *never
  touches* a break-even barrier over the holding horizon.

For a defined-risk structure you actively manage and exit before expiry, the
**path** probability is the honest input: a barrier touched mid-life can stop you
out even if price would have finished back inside the profit zone. Over the same
horizon, "never breaches" implies "in the profit zone at expiry", so
`PHT ≤ POP` always — and the gap is pure path risk:

![POP vs PHT](figures/pop_vs_pht.png)

**The library scores expected value from PHT, not POP.**

### First-passage math (reflection principle)

For a driftless geometric Brownian motion, the probability of travelling a
log-distance `a > 0` to a barrier within horizon `t` follows from the reflection
principle:

    P(hit) = 2 · (1 − Φ( a / (σ√t) ))

Intuition: every terminal path that *ends* beyond the barrier is matched by a
reflected path that touched the barrier and came back, so the touch probability
is twice the terminal-exceedance probability. For a two-sided structure the
probability of breaching *either* barrier is approximated additively,
`p_breach ≈ p_hit_lower + p_hit_upper`. That omits the joint-crossing term (it
double-counts paths that could reach both), so it *understates* PHT — a bias we
measure against a continuity-corrected Monte-Carlo benchmark rather than hide (see
the [model-risk doc](model_risk_and_validation.md)). The probability is computed
in a constant-volatility lognormal world, using the structure's implied volatility
as the diffusion parameter — a deliberate *screening*-model simplification; the
implied-volatility **smile** enters vanilla valuation, not this path model, and a
smile-consistent path dynamic is a natural extension.

Expected value is then `EV = PHT · profit_target − (1 − PHT) · stop_loss`.

## 3. Structures: the broken-wing butterfly

A broken-wing butterfly is a three-strike, defined-risk structure with *unequal*
wings — the asymmetry lets it be opened for a credit while keeping one side's
risk bounded. From the three strikes and the net credit the library derives max
profit, max risk, the "free-risk" case (a credit structure with no downside), the
break-even prices, and — via the probability primitives above — POP, PHT, and EV.

![Butterfly payoff](figures/butterfly_payoff.png)

## 4. Signal: implied-volatility curvature ("kink")

The entry signal is local curvature of the IV smile:

    kink(K) = IV(K) − ½·(IV(K⁻) + IV(K⁺))

A positive kink means the smile bulges at that strike — the option is locally
*rich* relative to its neighbours, which is where a body-centered butterfly can be
sold to advantage. Because a raw kink is hard to compare across names and
regimes, it is standardized two ways: a **cross-sectional** z-score (against the
other kinks in the same chain right now) and a **history** z-score (against the
strike's own recent distribution). Standardizing against a strike's own history —
rather than an absolute threshold — is a deliberate nod to non-stationarity:
"rich relative to normal for *this* strike" is more robust than "rich in absolute
terms."

## 5. Scoring: a bounded, shaped composite

Each surviving candidate is reduced to one bounded score that blends several
sub-scores (expected value, liquidity, IV regime, gamma/pin risk, time-to-expiry,
and optionally curvature, delta, and wing ratio). Two design choices matter:

- **Every factor is bounded to [0, 1] and shaped** so "good" saturates rather
  than running away — a spectacular value on one axis cannot let a candidate
  ignore the others. Liquidity, for instance, is the *minimum* of open-interest,
  volume, and spread sub-scores (a weakest-link property: great open interest
  does not rescue an untradeable spread).
- **The EV sub-score is normalized absolutely** (`min(EV / scale, 1)`), not by
  the day's batch, so a median-quality trade scores ~0.5 whether the day produced
  50 candidates or 5,000. That stability matters for a fixed acceptance
  threshold. In the original program this scale was *calibrated* on a large
  candidate sample so a median trade scores exactly 0.50, and the recalibration
  was adopted only after it was shown to materially reduce portfolio
  concentration — a governed change, not an ad-hoc tweak. (This repo ships an
  illustrative scale; the reasoning is what transfers.)

## 6. Execution realism

Backtests that ignore costs lie. Two fee models handle this:

- A **time-varying regulatory schedule** (options ORF/TAF, equity TAF and its
  per-order cap, and the SEC Section 31 fee) keyed to each charge's **historical
  effective date**, so a historical backtest applies the fees actually in force
  on each trade date. These schedules come from **public** SEC/exchange filings.
- A **per-structure** model with per-leg commission caps and correct
  body/wing sell-side counts (the FINRA TAF applies to sales, and the number of
  contracts sold differs between opening and closing a butterfly).

Modeled slippage (a floored fraction of the spread, capped in dollars) is applied
as an eligibility gate before a candidate can be selected.

## 7. Event gates

The income sleeves refuse to *open* new risk into known scheduled events rather
than trying to price the tail: an **earnings blackout** window, a
**corporate-action** block, and **macro-event** (FOMC / CPI / NFP) session
blocks, evaluated from a point-in-time context row. Missing or invalid context
fields can themselves block an entry — a fail-closed posture: if we cannot
confirm the event context, we do not open.

## 8. Portfolio-risk overlay

Sleeve scanners propose candidates in isolation; the risk layer decides how many
(if any) may actually open given the whole book. In one place it enforces an
**aggregate defined-risk cap**, a **per-underlying cap** (with per-symbol
overrides for high-volatility names), a **sector cap**, **per-sleeve caps**, a
**max-new-entries-per-run** throttle, and a **VIX-regime exposure cut** that
scales every candidate down when volatility is elevated. Sizing is greedy and
stateful — each accepted fill consumes capacity and the next candidate sees the
reduced headroom — and every decision carries reason codes for a full audit
trail.

## 9. Cash-ledger economics

The right way to measure a defined-risk program's P&L is a **cash ledger**, not a
sum of per-leg realized P&L. The two disagree exactly where it matters: when a
bull-put-spread short leg is assigned, the per-leg realized P&L captures only the
equity loss, while the opening credit lives on in the cash ledger — so leg-summing
understates the position. The economics engine books every real cash flow as an
immutable ledger event: trades (with fees), **assignment/exercise**,
**dividends** (ex-date receivable → pay-date cash), **tiered margin interest**,
and **T+N settlement**. Snapshots then partition events into settled / unsettled
/ receivable balances and compute margin, net liquidation value, and excess
liquidity.

## 10. Deterministic backtest contract

Inputs are governed by a **versioned data contract** (a JSON schema declaring
each dataset's required columns, types, bounds, and enums), so tightening a
validation rule is a data change, not a code change. The replay/settlement layer
uses an **injectable trading calendar** — a holiday-free weekday calendar by
default (zero dependencies, runnable anywhere), or a real exchange calendar when
holiday-accurate settlement is required.

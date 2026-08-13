# Architecture

## The pipeline

A single research pipeline flows through the modules in order — signal to sizing
to costs to economics:

```
                 synthetic chain (SSVI, arbitrage-free)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
    pricing/            signals/              gates/
  BS + IV inversion   IV-curvature "kink"   earnings / macro
  POP + first-passage   + z-scores          blackout gates
        │                     │                     │
        └───────────┬─────────┘                     │
                    ▼                               │
              structures/                           │
        broken-wing butterfly:                      │
        payoff, break-evens,                        │
        POP / PHT / EV                              │
                    │                               │
                    ▼                               ▼
              scoring/  ───────────────────►  sleeves/
        bounded composite rank            covered call · bull put
                                          spread · tail hedge
                                          (selection + roll/exit)
                                                    │
                    ┌───────────────────────────────┤
                    ▼                               ▼
              execution/                        risk/
        per-structure fees +               portfolio caps +
        regulatory schedules               VIX-regime cut
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                              economics/
                    cash-ledger: assignment / exercise /
                    dividends / margin / settlement
                                    │
                                    ▼
                              backtest/
                    data contract + schema validation
```

## Modules

| Module | Responsibility |
|---|---|
| `pricing/` | Black-Scholes pricing; IV by bisection; lognormal POP; reflection-principle first-passage & path-hold (PHT) probabilities; EV |
| `structures/` | broken-wing butterfly payoff geometry, break-evens, and POP/PHT/EV wiring |
| `signals/` | IV-curvature ("kink") signal with cross-sectional and history z-scores |
| `scoring/` | shaped, bounded sub-scores blended into a composite rank |
| `execution/` | per-structure fee model + time-varying regulatory fee schedules |
| `gates/` | earnings / corporate-action / macro-event blackout gates |
| `risk/` | portfolio caps (aggregate/symbol/sector/sleeve), VIX-regime cut, stateful sizer |
| `economics/` | cash-ledger events (assignment/exercise/dividends/margin) + settlement + snapshots |
| `backtest/` | versioned data contract + dataset validators |
| `sleeves/` | covered-call, bull-put-spread, tail-hedge decision logic |
| `utils/` | keyring-based secret loading |

## Design decisions worth noting

- **Decoupling over convenience.** The portfolio-risk engine originally consumed
  the scanners' rich candidate objects directly; here it depends only on small
  *risk-input* dataclasses carrying the few fields the risk math needs, so it is
  independent of any particular scanner and trivially testable.
- **Dependency injection for the calendar.** Settlement arithmetic depends on a
  trading calendar. Rather than hard-wire `exchange_calendars`, the economics
  engine takes an injectable `SessionCalendar`: a pure-Python weekday calendar by
  default (so the library runs anywhere with zero dependencies), or a real
  exchange calendar when holiday-accurate settlement is required.
- **Contract-driven validation.** Dataset rules live in a versioned JSON data
  contract, not in code, so tightening a rule is a data change.
- **Weakest-link and saturating scores.** Sub-scores are bounded and combined so
  no single dimension can dominate (liquidity is a *minimum* of its components;
  EV is absolutely normalized so scores are comparable across days).
- **Zero third-party dependencies in the core.** Every `optvol` module imports
  only the Python standard library. Optional extras (`matplotlib` for figures,
  `pytest` for tests, `exchange_calendars` for holiday-accurate settlement) are
  never required to import or use the library.

## Relationship to the original system

This library is the reusable **research core** of a larger private options
program. The published modules are refactors of the originals — reorganized into
a clean package, decoupled from heavy dependencies, and stripped of tuned
constants and live plumbing — and each was verified **behavior-identical to the
original** by randomized differential testing (~108,000 comparisons; see
[model risk & validation](model_risk_and_validation.md)). The original's live
order-routing, assignment-remediation, vendor-data ingestion, and
production-calibrated parameters are described here but not shipped: they carry
operational risk and tuned IP, not research insight.

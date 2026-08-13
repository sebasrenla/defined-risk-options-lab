"""optvol — a systematic options research library and deterministic backtester.

This package is a cleaned, reorganized extract of a larger private research
program for defined-risk equity-options strategies (butterflies, covered calls,
bull put spreads, tail hedges). It exposes the *reasoning and methods* — options
pricing, first-passage / probability modeling, execution-cost realism, event
gating, portfolio risk, and cash-ledger economics — together with a synthetic
data layer so every component is runnable without any licensed market data.

No third-party or vendor market data is included; see DATA_NOTICE.md. Nothing
here is investment advice; see DISCLAIMER.md.
"""

__version__ = "0.1.0"

# Data Notice

**This repository contains original research code. No third-party or vendor
market data is included, and none can be reconstructed from anything here.**

## What data the demos and tests use

Everything in this repository runs on a **synthetic** option-chain surface
generated locally by `examples/generate_synthetic_chain.py`. That surface is
built with the SSVI (Gatheral–Jacquier) parameterization, is verified free of
static arbitrage, and is not derived from any market feed. See
[`docs/synthetic_data.md`](docs/synthetic_data.md).

## The original research

The larger private program from which this library was extracted was developed
using licensed data sources under their respective subscriber agreements, which
prohibit redistribution. **None of that data — and no dataset from which it could
be reverse-engineered — is present in this repository.** Where the docs cite
results computed on real data, they are high-level summary statistics only
(for example, a correlation coefficient or an approximation-bias magnitude), not
data.

If you build on this code with data of your own, respect your data providers'
terms. In particular:

- **Exchange / options data** (e.g. CBOE and similar) is typically licensed for
  personal or internal use only and may not be redistributed; only derived works
  that cannot be reverse-engineered to the original data may be shared.
- **Academic data services** (e.g. CRSP / Compustat via WRDS) are for
  non-commercial research and may not be redistributed; published results should
  carry the provider's required attribution.
- **Broker market data** (e.g. via a brokerage API) generally may not be
  redistributed to non-clients, and access tokens must never be stored in source.

## Attributions for methods referenced

- Implied-volatility surface: Gatheral, J. & Jacquier, A., *Arbitrage-free SVI
  volatility surfaces* (2014).

No credentials, personal identifiers, or account information appear anywhere in
this repository. Secrets are loaded at runtime from the OS keyring — see
`src/optvol/utils/secret_loader.py`.

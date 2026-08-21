"""Per-structure fee estimation for a broken-wing butterfly.

Broker commissions and per-contract charges are account-specific, so this module
is configuration-driven: it reads a ``cfg`` mapping of fee parameters (prefixed
``fee_*``) and returns the total fees for one *side* of a butterfly (opening or
closing). It handles the details that a naive ``commission * legs`` estimate
misses:

* **Per-leg commission cap**: many brokers cap commission per leg.
* **Sell-side-only charges**: the FINRA TAF applies to *sales*; in a butterfly
  the number of contracts sold differs between the open (short the body) and the
  close (buy the body back / sell the wings), so the fee is side-aware.
* **Per-contract vs per-order vs per-symbol** components, clearing/ORF/TAF are
  per contract, some fees are per order, and a few names carry symbol add-ons.

For the *regulatory* rate schedules (ORF/TAF/SEC), see
:mod:`optvol.execution.fees_regulatory`.

Provenance
----------
Ported from the engine's ``fee_model.py``.
"""

from __future__ import annotations

from typing import Optional


def _cfg_float(cfg: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def _symbol_fee(cfg: dict, key: str, symbol: Optional[str]) -> float:
    if not symbol:
        return 0.0
    raw_map = cfg.get(key, {}) or {}
    if not isinstance(raw_map, dict):
        return 0.0
    raw = raw_map.get(str(symbol).upper())
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _leg_contracts(cfg: dict, leg_contracts_key: str, fallback_legs_key: str) -> list[float]:
    """Per-leg contract counts. Defaults to three equal legs (a 1x1x1 fly)."""
    raw = cfg.get(leg_contracts_key)
    if isinstance(raw, list) and raw:
        out: list[float] = []
        for leg in raw:
            try:
                out.append(max(float(leg), 0.0))
            except (TypeError, ValueError):
                out.append(0.0)
        if out:
            return out
    try:
        legs = max(int(cfg.get(fallback_legs_key, 3)), 1)
    except (TypeError, ValueError):
        legs = 3
    return [1.0] * legs


def bwb_contract_side_counts(
    cfg: dict,
    quantity: int,
    *,
    side: str,
    leg_contracts_key: str = "fee_spread_leg_contracts",
    fallback_legs_key: str = "fee_legs_per_spread",
) -> tuple[float, float]:
    """Return ``(total_contracts, sell_side_contracts)`` for ``quantity`` spreads.

    On the open the body is sold; on the close the wings are sold. The sell count
    drives sale-only fees like the TAF.
    """
    if quantity <= 0:
        return 0.0, 0.0
    legs = _leg_contracts(cfg, leg_contracts_key, fallback_legs_key)
    total_per_spread = sum(legs)
    if len(legs) >= 3:
        body_per_spread = legs[1]
        wing_per_spread = max(total_per_spread - body_per_spread, 0.0)
        sell_per_spread = body_per_spread if side == "open" else wing_per_spread
    else:
        sell_per_spread = total_per_spread
    return total_per_spread * quantity, sell_per_spread * quantity


def estimate_bwb_side_fee_total(
    cfg: dict,
    quantity: int,
    *,
    side: str,
    symbol: Optional[str] = None,
    prefix: str = "fee_",
    fallback_legs_key: str = "fee_legs_per_spread",
) -> float:
    """Total fees (commission + clearing + ORF + TAF + per-order + symbol add-on)
    for one side of ``quantity`` butterflies."""
    if quantity <= 0:
        return 0.0

    commission = _cfg_float(cfg, f"{prefix}{side}_commission", 0.0)
    commission_cap_raw = cfg.get(f"{prefix}commission_cap_per_leg")
    try:
        commission_cap = float(commission_cap_raw) if commission_cap_raw is not None else None
    except (TypeError, ValueError):
        commission_cap = None

    clearing = _cfg_float(cfg, f"{prefix}clearing", 0.0)
    orf = _cfg_float(cfg, f"{prefix}orf", 0.0)
    taf = _cfg_float(cfg, f"{prefix}taf", 0.0)
    per_order = _cfg_float(cfg, f"{prefix}per_order", 0.0)
    symbol_fee = _symbol_fee(cfg, f"{prefix}symbol_addons", symbol)

    legs = _leg_contracts(cfg, f"{prefix}spread_leg_contracts", fallback_legs_key)
    total_contracts, sell_contracts = bwb_contract_side_counts(
        cfg,
        quantity,
        side=side,
        leg_contracts_key=f"{prefix}spread_leg_contracts",
        fallback_legs_key=fallback_legs_key,
    )

    total = per_order
    for leg in legs:
        contracts = leg * quantity
        leg_commission = commission * contracts
        if commission_cap is not None:
            leg_commission = min(leg_commission, commission_cap)
        total += leg_commission

    total += total_contracts * (clearing + orf + symbol_fee)
    total += sell_contracts * taf
    return total


def estimate_bwb_side_fee_per_share(
    cfg: dict,
    quantity: int,
    contract_multiplier: float,
    *,
    side: str,
    symbol: Optional[str] = None,
    prefix: str = "fee_",
    fallback_legs_key: str = "fee_legs_per_spread",
) -> float:
    """Side fees expressed per underlying share (fees / (multiplier * quantity))."""
    if contract_multiplier <= 0 or quantity <= 0:
        return 0.0
    total = estimate_bwb_side_fee_total(
        cfg, quantity, side=side, symbol=symbol, prefix=prefix, fallback_legs_key=fallback_legs_key
    )
    return total / (contract_multiplier * quantity)

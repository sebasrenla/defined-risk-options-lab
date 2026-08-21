"""Covered-call sleeve: candidate selection, sizing, and roll/exit decisions.

The covered-call income sleeve sells an out-of-the-money call against long stock.
This module holds the *decision logic* only, no data-vendor adapters, no output
writers:

* :func:`select_covered_call_candidate`: from an option chain, pick the single
  best short call inside the target **delta** and **DTE** windows that clears the
  **liquidity** gates (open interest, volume, bid/ask spread) and a **slippage**
  ceiling, then rank the survivors by proximity to a target delta (tie-broken by
  spread, then open interest, then volume).
* :func:`determine_allocation_multiplier`: size the position by IV rank: a full
  allocation when IV rank is in the preferred regime, a haircut when it is low or
  unavailable (a documented risk rule, not an ad-hoc tweak).
* :func:`estimate_covered_call_roundtrip_fees`: the full round-trip cost (open +
  close, option + stock legs) using the regulatory fee models.
* :func:`evaluate_covered_call_roll_action`: the management rule set: roll on
  assignment risk near an ex-dividend date, take profit at half the entry credit,
  roll on an adverse move late in the cycle, or roll on time.

Decoupling note
---------------
The original selector consumed a pandas DataFrame; here it operates on a plain
sequence of :class:`OptionQuote` rows so the sleeve has no heavy dependencies.
The filtering, ranking, and selection are otherwise identical.

Provenance
----------
Ported from the program's ``cc_scanner.py`` (decision functions only).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Optional, Sequence

from ..execution.fees_regulatory import (
    CURRENT_EQUITY_CLEARING_PER_SHARE,
    CURRENT_EQUITY_OPEN_COMMISSION_PER_ORDER,
    CURRENT_EQUITY_SELL_TAF_CAP_PER_ORDER,
    CURRENT_EQUITY_SELL_TAF_PER_SHARE,
    CURRENT_OPTION_CLEARING_PER_CONTRACT,
    CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT,
    CURRENT_OPTION_COMMISSION_CAP_PER_LEG,
    CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT,
    CURRENT_OPTION_ORF_PER_CONTRACT,
    CURRENT_OPTION_TAF_PER_CONTRACT,
    estimate_equity_transaction_fees,
    estimate_option_transaction_fees,
)


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _coerce_optional_date(value: date | datetime | str | None) -> date | None:
    if value is None or value == "":
        return None
    return _coerce_date(value)


@dataclass(frozen=True)
class CoveredCallConfig:
    """Covered-call sleeve parameters (illustrative defaults)."""

    dte_min: int = 25
    dte_max: int = 45
    call_delta_min: float = 0.20
    call_delta_max: float = 0.35
    target_delta: float = 0.275
    short_leg_open_interest_min: int = 500
    short_leg_volume_min: int = 50
    max_call_spread_pct_of_mid: float = 0.08
    estimated_slippage_pct_of_spread: float = 0.10
    estimated_slippage_abs_floor: float = 0.02
    estimated_slippage_abs_cap: float = 0.20
    max_estimated_slippage_pct_of_spread: float = 0.15
    contract_multiplier: int = 100
    default_allocation_multiplier: float = 1.0
    low_iv_rank_allocation_multiplier: float = 0.5
    iv_rank_unknown_allocation_multiplier: float = 0.5
    iv_rank_preferred_min: float = 30.0
    ex_div_proximity_days_trigger: int = 14
    fee_open_commission: float = CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT
    fee_close_commission: float = CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT
    fee_commission_cap_per_leg: float = CURRENT_OPTION_COMMISSION_CAP_PER_LEG
    fee_clearing: float = CURRENT_OPTION_CLEARING_PER_CONTRACT
    fee_orf: float = CURRENT_OPTION_ORF_PER_CONTRACT
    fee_taf: float = CURRENT_OPTION_TAF_PER_CONTRACT
    equity_fee_open_commission_per_order: float = CURRENT_EQUITY_OPEN_COMMISSION_PER_ORDER
    equity_fee_close_commission_per_order: float = 0.0
    equity_fee_clearing_per_share: float = CURRENT_EQUITY_CLEARING_PER_SHARE
    equity_fee_sell_taf_per_share: float = CURRENT_EQUITY_SELL_TAF_PER_SHARE
    equity_fee_sell_taf_cap_per_order: float = CURRENT_EQUITY_SELL_TAF_CAP_PER_ORDER
    fee_symbol_addons: Mapping[str, float] = field(default_factory=dict)


@dataclass
class OptionQuote:
    """One option-chain row (decoupled from any data vendor)."""

    symbol: str
    expiry: date
    dte: Optional[int]
    option_type: str
    strike: float
    bid: Optional[float]
    ask: Optional[float]
    delta: Optional[float]
    open_interest: Optional[int]
    volume: Optional[int]
    underlying_price: Optional[float]
    iv: Optional[float] = None


@dataclass
class CoveredCallCandidate:
    symbol: str
    as_of_date: date
    expiry: date
    dte: int
    strike: float
    delta: float
    bid: float
    ask: float
    mid: float
    iv: float | None
    underlying_price: float
    open_interest: int
    volume: int
    spread_abs: float
    spread_pct_of_mid: float
    estimated_slippage_abs: float
    estimated_slippage_pct_of_spread: float
    nav_cap_pct: float
    allocation_multiplier: float
    allocation_reason: str
    iv_rank: float | None
    iv_percentile: float | None
    beta: float | None
    estimated_roundtrip_fees: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["as_of_date"] = self.as_of_date.isoformat()
        payload["expiry"] = self.expiry.isoformat()
        return payload


@dataclass
class CoveredCallRollDecision:
    action_code: str
    should_close: bool
    should_reopen_next_cycle: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_option_slippage(spread_abs: float, config: CoveredCallConfig) -> tuple[float, float]:
    """Modeled slippage: a floored fraction of the spread, capped in dollars."""
    estimate = max(config.estimated_slippage_abs_floor, spread_abs * config.estimated_slippage_pct_of_spread)
    estimate = min(estimate, config.estimated_slippage_abs_cap)
    pct_of_spread = estimate / spread_abs if spread_abs > 0 else 0.0
    return estimate, pct_of_spread


def determine_allocation_multiplier(iv_rank: float | None, config: CoveredCallConfig) -> tuple[float, str]:
    """Size by IV regime: full in the preferred regime, haircut when low/unknown."""
    if iv_rank is None:
        return config.iv_rank_unknown_allocation_multiplier, "iv_rank_unavailable_default"
    if iv_rank < config.iv_rank_preferred_min:
        return config.low_iv_rank_allocation_multiplier, "low_iv_rank_haircut"
    return config.default_allocation_multiplier, "preferred_iv_rank"


def compute_near_ex_div(
    ex_div_date: date | datetime | str | None, as_of_date: date | datetime | str, window_days: int
) -> bool:
    """True if an ex-dividend date falls within ``window_days`` ahead of today."""
    resolved_ex_div = _coerce_optional_date(ex_div_date)
    if resolved_ex_div is None:
        return False
    window = max(int(window_days), 0)
    days_until = (resolved_ex_div - _coerce_date(as_of_date)).days
    return 0 <= days_until <= window


def estimate_covered_call_roundtrip_fees(
    config: CoveredCallConfig,
    *,
    quantity: int = 1,
    symbol: str | None = None,
    underlying_price: float | None = None,
    as_of_date: date | datetime | str | None = None,
) -> float:
    """Full round-trip cost: option open+close and stock open+close legs."""
    contracts = max(0, int(quantity))
    if contracts == 0:
        return 0.0
    symbol_addon = 0.0
    if symbol:
        raw = config.fee_symbol_addons.get(symbol.upper().strip())
        if raw is not None:
            try:
                symbol_addon = float(raw)
            except (TypeError, ValueError):
                symbol_addon = 0.0
    share_count = contracts * config.contract_multiplier
    option_open = estimate_option_transaction_fees(
        contracts=contracts, commission_per_contract=config.fee_open_commission,
        commission_cap_per_leg=config.fee_commission_cap_per_leg, clearing_per_contract=config.fee_clearing,
        orf_per_contract=config.fee_orf, taf_per_contract=config.fee_taf,
        symbol_addon_per_contract=symbol_addon, is_sale=True)
    option_close = estimate_option_transaction_fees(
        contracts=contracts, commission_per_contract=config.fee_close_commission,
        commission_cap_per_leg=config.fee_commission_cap_per_leg, clearing_per_contract=config.fee_clearing,
        orf_per_contract=config.fee_orf, taf_per_contract=config.fee_taf,
        symbol_addon_per_contract=symbol_addon, is_sale=False)
    stock_open = estimate_equity_transaction_fees(
        shares=share_count, commission_per_order=config.equity_fee_open_commission_per_order,
        clearing_per_share=config.equity_fee_clearing_per_share,
        sell_taf_per_share=config.equity_fee_sell_taf_per_share,
        sell_taf_cap_per_order=config.equity_fee_sell_taf_cap_per_order, is_sale=False)
    stock_close = estimate_equity_transaction_fees(
        shares=share_count, commission_per_order=config.equity_fee_close_commission_per_order,
        clearing_per_share=config.equity_fee_clearing_per_share,
        sell_taf_per_share=config.equity_fee_sell_taf_per_share,
        sell_taf_cap_per_order=config.equity_fee_sell_taf_cap_per_order,
        trade_date=as_of_date, price_per_share=underlying_price, is_sale=True)
    return option_open + option_close + stock_open + stock_close


def select_covered_call_candidate(
    *,
    quotes: Sequence[OptionQuote],
    symbol: str,
    as_of_date: date | datetime | str,
    nav_cap_pct: float,
    config: CoveredCallConfig,
    iv_rank: float | None = None,
    iv_percentile: float | None = None,
    beta: float | None = None,
) -> tuple[CoveredCallCandidate | None, dict[str, Any]]:
    """Select the best short-call candidate for ``symbol`` from ``quotes``.

    Returns ``(candidate_or_None, diagnostics)``. Diagnostics record how many rows
    survived each stage, which is invaluable when a symbol produces no candidate.
    """
    session_date = _coerce_date(as_of_date)
    symbol_code = symbol.upper().strip()

    # Prepare: keep this symbol's quotes with a valid two-sided market.
    prepared = []
    for q in quotes:
        if str(q.symbol).upper().strip() != symbol_code:
            continue
        if q.bid is None or q.ask is None:
            continue
        mid = (q.bid + q.ask) / 2.0
        if q.ask < q.bid or mid <= 0:
            continue
        prepared.append((q, mid, q.ask - q.bid, (q.ask - q.bid) / mid))

    diagnostics: dict[str, Any] = {
        "rows_seen": len(prepared),
        "iv_rank": iv_rank,
        "iv_rank_unavailable": iv_rank is None,
        "iv_percentile": iv_percentile,
        "beta": beta,
    }
    if not prepared:
        return None, diagnostics

    call_rows = [row for row in prepared if str(row[0].option_type).lower().strip() == "call"]
    diagnostics["call_rows"] = len(call_rows)
    if not call_rows:
        return None, diagnostics

    def _passes_primary(row) -> bool:
        q, mid, spread_abs, spread_pct = row
        if q.dte is None or q.delta is None or q.open_interest is None or q.volume is None:
            return False
        return (
            config.dte_min <= q.dte <= config.dte_max
            and config.call_delta_min <= q.delta <= config.call_delta_max
            and q.open_interest >= config.short_leg_open_interest_min
            and q.volume >= config.short_leg_volume_min
            and spread_pct <= config.max_call_spread_pct_of_mid
        )

    filtered = [row for row in call_rows if _passes_primary(row)]
    diagnostics["eligible_after_primary_filters"] = len(filtered)
    if not filtered:
        return None, diagnostics

    # Slippage filter.
    enriched = []
    for q, mid, spread_abs, spread_pct in filtered:
        slip_abs, slip_pct = estimate_option_slippage(spread_abs, config)
        if slip_pct <= config.max_estimated_slippage_pct_of_spread:
            enriched.append((q, mid, spread_abs, spread_pct, slip_abs, slip_pct))
    diagnostics["eligible_after_slippage"] = len(enriched)
    if not enriched:
        return None, diagnostics

    # Rank: nearest target delta, then tighter spread, then more OI, then more volume.
    enriched.sort(key=lambda r: (abs(r[0].delta - config.target_delta), r[3], -r[0].open_interest, -r[0].volume))
    q, mid, spread_abs, spread_pct, slip_abs, slip_pct = enriched[0]

    allocation_multiplier, allocation_reason = determine_allocation_multiplier(iv_rank, config)
    estimated_roundtrip_fees = estimate_covered_call_roundtrip_fees(
        config, symbol=symbol_code, underlying_price=float(q.underlying_price), as_of_date=as_of_date)
    diagnostics["allocation_reason"] = allocation_reason
    diagnostics["estimated_roundtrip_fees"] = estimated_roundtrip_fees

    candidate = CoveredCallCandidate(
        symbol=symbol_code, as_of_date=session_date, expiry=q.expiry, dte=int(q.dte),
        strike=float(q.strike), delta=float(q.delta), bid=float(q.bid), ask=float(q.ask), mid=float(mid),
        iv=(float(q.iv) if q.iv is not None else None), underlying_price=float(q.underlying_price),
        open_interest=int(q.open_interest), volume=int(q.volume), spread_abs=float(spread_abs),
        spread_pct_of_mid=float(spread_pct), estimated_slippage_abs=float(slip_abs),
        estimated_slippage_pct_of_spread=float(slip_pct), nav_cap_pct=float(nav_cap_pct),
        allocation_multiplier=float(allocation_multiplier), allocation_reason=allocation_reason,
        iv_rank=iv_rank, iv_percentile=iv_percentile, beta=beta,
        estimated_roundtrip_fees=float(estimated_roundtrip_fees), diagnostics=diagnostics)
    return candidate, diagnostics


def evaluate_covered_call_roll_action(
    *,
    entry_credit: float,
    current_option_mark: float,
    short_strike: float,
    spot_price: float,
    dte_remaining: int,
    near_ex_div: bool | None = None,
    as_of_date: date | datetime | str | None = None,
    ex_div_date: date | datetime | str | None = None,
    ex_div_proximity_days_trigger: int = 14,
) -> CoveredCallRollDecision:
    """Management rule set for an open covered call.

    Priority: assignment-risk near ex-div (short call deep ITM with no extrinsic)
    -> profit-take at half the entry credit -> adverse move late in the cycle ->
    time exit -> otherwise hold.
    """
    if entry_credit <= 0:
        raise ValueError("entry_credit must be positive")
    if current_option_mark < 0:
        raise ValueError("current_option_mark must be non-negative")
    if short_strike <= 0:
        raise ValueError("short_strike must be positive")
    if spot_price <= 0:
        raise ValueError("spot_price must be positive")
    if dte_remaining < 0:
        raise ValueError("dte_remaining must be non-negative")

    intrinsic = max(spot_price - short_strike, 0.0)
    extrinsic = max(current_option_mark - intrinsic, 0.0)
    resolved_ex_div_date = _coerce_optional_date(ex_div_date)
    derived_near_ex_div = bool(near_ex_div)
    if near_ex_div is None:
        derived_near_ex_div = (
            compute_near_ex_div(ex_div_date=resolved_ex_div_date, as_of_date=as_of_date,
                                window_days=ex_div_proximity_days_trigger)
            if as_of_date is not None else False
        )
    assignment_risk_condition = derived_near_ex_div and intrinsic > 0.0 and extrinsic <= 0.05
    profit_take_threshold = entry_credit * 0.5
    profit_take_condition = current_option_mark <= profit_take_threshold
    adverse_move_condition = spot_price >= short_strike and dte_remaining <= 10
    time_exit_condition = dte_remaining <= 7
    diagnostics = {
        "intrinsic": intrinsic, "extrinsic": extrinsic, "dte_remaining": dte_remaining,
        "near_ex_div": derived_near_ex_div,
        "as_of_date": _coerce_date(as_of_date).isoformat() if as_of_date is not None else None,
        "ex_div_date": resolved_ex_div_date.isoformat() if resolved_ex_div_date is not None else None,
        "ex_div_proximity_days_trigger": max(int(ex_div_proximity_days_trigger), 0),
        "profit_take_option_mark_threshold": profit_take_threshold,
        "assignment_risk_condition": assignment_risk_condition,
        "profit_take_condition": profit_take_condition,
        "adverse_move_condition": adverse_move_condition,
        "time_exit_condition": time_exit_condition,
    }

    if assignment_risk_condition:
        return CoveredCallRollDecision("roll_assignment_risk", True, True, diagnostics)
    if profit_take_condition:
        return CoveredCallRollDecision("close_profit_take", True, False, diagnostics)
    if adverse_move_condition:
        return CoveredCallRollDecision("roll_adverse_move", True, True, diagnostics)
    if time_exit_condition:
        return CoveredCallRollDecision("roll_time_exit", True, True, diagnostics)
    return CoveredCallRollDecision("hold", False, False, diagnostics)

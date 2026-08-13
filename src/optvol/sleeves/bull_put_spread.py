"""Bull-put-spread sleeve: candidate selection, sizing, and exit decisions.

A bull put spread sells an out-of-the-money put and buys a further-OTM put for
protection — a defined-risk way to collect premium with a bullish/neutral lean.
Selection is a two-leg search:

* **Short leg** — a put in the target short-delta band (≈0.20–0.30 abs) that
  clears the liquidity gates (open interest, volume, spread).
* **Long leg** — a cheaper, further-OTM put in the target long-delta band
  (≈0.08–0.15 abs) with an acceptable spread.
* **Pairing** — for each same-expiry (short, long) pair with the long strike
  below the short strike, build the spread and require a sensible **width**
  (as a % of spot), a **structure spread** within tolerance, and a **credit
  floor** (a minimum credit both in absolute terms and as a % of width).
* **Ranking** — nearest to target short delta, then long delta, then tighter
  structure spread, then richer credit-to-width, then richer short premium.

Exit rules (:func:`evaluate_bull_put_spread_exit_action`): profit-take at half
the entry credit, stop-loss at 2x, plus time and breach controls for the "slow
drift into expiry" case the stop-loss alone misses.

Provenance
----------
Ported from the program's ``put_spread_scanner.py`` (decision functions only;
selector reimplemented on :class:`OptionQuote` rows without pandas — the pair
ranking already used a stable sort, so tie behavior is preserved exactly).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Optional, Sequence

from ..execution.fees_regulatory import (
    CURRENT_OPTION_CLEARING_PER_CONTRACT,
    CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT,
    CURRENT_OPTION_COMMISSION_CAP_PER_LEG,
    CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT,
    CURRENT_OPTION_ORF_PER_CONTRACT,
    CURRENT_OPTION_TAF_PER_CONTRACT,
    estimate_option_transaction_fees,
)
from .covered_call import OptionQuote, _coerce_date


@dataclass(frozen=True)
class BullPutSpreadConfig:
    """Bull-put-spread sleeve parameters (illustrative defaults)."""

    dte_min: int = 20
    dte_max: int = 45
    short_put_delta_min_abs: float = 0.20
    short_put_delta_max_abs: float = 0.30
    target_short_put_delta_abs: float = 0.25
    long_put_delta_min_abs: float = 0.08
    long_put_delta_max_abs: float = 0.15
    target_long_put_delta_abs: float = 0.115
    short_leg_open_interest_min: int = 500
    short_leg_volume_min: int = 50
    width_pct_spot_min: float = 0.03
    width_pct_spot_max: float = 0.07
    min_credit_abs: float = 0.20
    min_credit_pct_of_width: float = 0.08
    max_short_leg_spread_pct_of_mid: float = 0.08
    max_long_leg_spread_pct_of_mid: float = 0.12
    max_structure_spread_pct_of_mid: float = 0.15
    contract_multiplier: int = 100
    default_allocation_multiplier: float = 1.0
    low_iv_rank_allocation_multiplier: float = 0.75
    iv_rank_unknown_allocation_multiplier: float = 0.75
    iv_rank_preferred_min: float = 30.0
    fee_open_commission: float = CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT
    fee_close_commission: float = CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT
    fee_commission_cap_per_leg: float = CURRENT_OPTION_COMMISSION_CAP_PER_LEG
    fee_clearing: float = CURRENT_OPTION_CLEARING_PER_CONTRACT
    fee_orf: float = CURRENT_OPTION_ORF_PER_CONTRACT
    fee_taf: float = CURRENT_OPTION_TAF_PER_CONTRACT
    fee_symbol_addons: dict[str, float] = field(default_factory=dict)


@dataclass
class BullPutSpreadCandidate:
    symbol: str
    as_of_date: date
    expiry: date
    dte: int
    short_strike: float
    long_strike: float
    width: float
    width_pct_of_spot: float
    short_delta: float
    long_delta: float
    short_bid: float
    short_ask: float
    short_mid: float
    long_bid: float
    long_ask: float
    long_mid: float
    structure_bid: float
    structure_ask: float
    structure_mid: float
    structure_spread_abs: float
    structure_spread_pct_of_mid: float
    underlying_price: float
    nav_cap_pct: float
    allocation_multiplier: float
    allocation_reason: str
    credit_floor: float
    max_profit_model: float
    max_loss_model: float
    estimated_roundtrip_fees: float
    iv_rank: float | None
    iv_percentile: float | None
    beta: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["as_of_date"] = self.as_of_date.isoformat()
        payload["expiry"] = self.expiry.isoformat()
        return payload


@dataclass
class BullPutSpreadExitDecision:
    action_code: str
    should_close: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def determine_allocation_multiplier(iv_rank: float | None, config: BullPutSpreadConfig) -> tuple[float, str]:
    """Size by IV regime (same rule as the covered-call sleeve)."""
    if iv_rank is None:
        return config.iv_rank_unknown_allocation_multiplier, "iv_rank_unavailable_default"
    if iv_rank < config.iv_rank_preferred_min:
        return config.low_iv_rank_allocation_multiplier, "low_iv_rank_haircut"
    return config.default_allocation_multiplier, "preferred_iv_rank"


def estimate_bull_put_spread_roundtrip_fees(
    config: BullPutSpreadConfig, *, quantity: int = 1, symbol: str | None = None
) -> float:
    """Round-trip cost for both legs (short open/close + long open/close)."""
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

    # short open(sale) + short close(buy) + long open(buy) + long close(sale)
    short_open = estimate_option_transaction_fees(
        contracts=contracts, commission_per_contract=config.fee_open_commission,
        commission_cap_per_leg=config.fee_commission_cap_per_leg, clearing_per_contract=config.fee_clearing,
        orf_per_contract=config.fee_orf, taf_per_contract=config.fee_taf,
        symbol_addon_per_contract=symbol_addon, is_sale=True)
    short_close = estimate_option_transaction_fees(
        contracts=contracts, commission_per_contract=config.fee_close_commission,
        commission_cap_per_leg=config.fee_commission_cap_per_leg, clearing_per_contract=config.fee_clearing,
        orf_per_contract=config.fee_orf, taf_per_contract=config.fee_taf,
        symbol_addon_per_contract=symbol_addon, is_sale=False)
    long_open = estimate_option_transaction_fees(
        contracts=contracts, commission_per_contract=config.fee_open_commission,
        commission_cap_per_leg=config.fee_commission_cap_per_leg, clearing_per_contract=config.fee_clearing,
        orf_per_contract=config.fee_orf, taf_per_contract=config.fee_taf,
        symbol_addon_per_contract=symbol_addon, is_sale=False)
    long_close = estimate_option_transaction_fees(
        contracts=contracts, commission_per_contract=config.fee_close_commission,
        commission_cap_per_leg=config.fee_commission_cap_per_leg, clearing_per_contract=config.fee_clearing,
        orf_per_contract=config.fee_orf, taf_per_contract=config.fee_taf,
        symbol_addon_per_contract=symbol_addon, is_sale=True)
    return short_open + short_close + long_open + long_close


def _prepare(quotes: Sequence[OptionQuote], symbol_code: str) -> list[tuple[OptionQuote, float, float, float]]:
    """Keep this symbol's quotes with a valid two-sided market; attach mid/spread."""
    out = []
    for q in quotes:
        if str(q.symbol).upper().strip() != symbol_code:
            continue
        if q.bid is None or q.ask is None:
            continue
        mid = (q.bid + q.ask) / 2.0
        if q.ask < q.bid or mid <= 0:
            continue
        out.append((q, mid, q.ask - q.bid, (q.ask - q.bid) / mid))
    return out


def _build_candidate(
    *, short: tuple[OptionQuote, float, float, float], long: tuple[OptionQuote, float, float, float],
    as_of_date: date, symbol: str, nav_cap_pct: float, config: BullPutSpreadConfig,
    iv_rank: float | None, iv_percentile: float | None, beta: float | None,
) -> BullPutSpreadCandidate | None:
    sq, s_mid, _, _ = short
    lq, l_mid, _, _ = long
    underlying_price = float(sq.underlying_price)
    if underlying_price <= 0:
        return None
    width = float(sq.strike) - float(lq.strike)
    if width <= 0:
        return None
    width_pct_of_spot = width / underlying_price
    if width_pct_of_spot < config.width_pct_spot_min or width_pct_of_spot > config.width_pct_spot_max:
        return None

    structure_bid = float(sq.bid) - float(lq.ask)
    structure_ask = float(sq.ask) - float(lq.bid)
    structure_mid = s_mid - l_mid
    if structure_ask < structure_bid or structure_mid <= 0:
        return None
    structure_spread_abs = structure_ask - structure_bid
    structure_spread_pct_of_mid = structure_spread_abs / abs(structure_mid) if structure_mid != 0 else float("inf")
    if structure_spread_pct_of_mid > config.max_structure_spread_pct_of_mid:
        return None

    credit_floor = max(config.min_credit_abs, config.min_credit_pct_of_width * width)
    if structure_mid < credit_floor:
        return None

    allocation_multiplier, allocation_reason = determine_allocation_multiplier(iv_rank, config)
    estimated_roundtrip_fees = estimate_bull_put_spread_roundtrip_fees(config, symbol=symbol)
    max_profit_model = structure_mid * config.contract_multiplier
    max_loss_model = max(width - structure_mid, 0.0) * config.contract_multiplier
    diagnostics = {
        "credit_floor": credit_floor,
        "credit_to_width_ratio": structure_mid / width if width > 0 else None,
        "allocation_multiplier": allocation_multiplier, "allocation_reason": allocation_reason,
        "estimated_roundtrip_fees": estimated_roundtrip_fees,
        "iv_rank": iv_rank, "iv_percentile": iv_percentile, "beta": beta,
    }
    return BullPutSpreadCandidate(
        symbol=symbol, as_of_date=as_of_date, expiry=sq.expiry, dte=int(sq.dte),
        short_strike=float(sq.strike), long_strike=float(lq.strike), width=float(width),
        width_pct_of_spot=float(width_pct_of_spot), short_delta=float(sq.delta), long_delta=float(lq.delta),
        short_bid=float(sq.bid), short_ask=float(sq.ask), short_mid=float(s_mid),
        long_bid=float(lq.bid), long_ask=float(lq.ask), long_mid=float(l_mid),
        structure_bid=float(structure_bid), structure_ask=float(structure_ask), structure_mid=float(structure_mid),
        structure_spread_abs=float(structure_spread_abs), structure_spread_pct_of_mid=float(structure_spread_pct_of_mid),
        underlying_price=float(underlying_price), nav_cap_pct=float(nav_cap_pct),
        allocation_multiplier=float(allocation_multiplier), allocation_reason=allocation_reason,
        credit_floor=float(credit_floor), max_profit_model=float(max_profit_model),
        max_loss_model=float(max_loss_model), estimated_roundtrip_fees=float(estimated_roundtrip_fees),
        iv_rank=iv_rank, iv_percentile=iv_percentile, beta=beta, diagnostics=diagnostics)


def select_bull_put_spread_candidate(
    *,
    quotes: Sequence[OptionQuote],
    symbol: str,
    as_of_date: date | datetime | str,
    nav_cap_pct: float,
    config: BullPutSpreadConfig,
    iv_rank: float | None = None,
    iv_percentile: float | None = None,
    beta: float | None = None,
) -> tuple[BullPutSpreadCandidate | None, dict[str, Any]]:
    """Select the best bull-put-spread pair for ``symbol`` from ``quotes``."""
    session_date = _coerce_date(as_of_date)
    symbol_code = symbol.upper().strip()
    prepared = _prepare(quotes, symbol_code)
    diagnostics: dict[str, Any] = {
        "rows_seen": len(prepared), "iv_rank": iv_rank, "iv_percentile": iv_percentile, "beta": beta,
    }
    if not prepared:
        return None, diagnostics

    put_rows = [row for row in prepared if str(row[0].option_type).lower().strip() == "put"]
    diagnostics["put_rows"] = len(put_rows)
    if not put_rows:
        return None, diagnostics

    def _valid_common(q) -> bool:
        return q.dte is not None and q.delta is not None

    short_candidates = [
        row for row in put_rows
        if _valid_common(row[0]) and row[0].open_interest is not None and row[0].volume is not None
        and config.dte_min <= row[0].dte <= config.dte_max
        and row[0].delta <= -config.short_put_delta_min_abs
        and row[0].delta >= -config.short_put_delta_max_abs
        and row[0].open_interest >= config.short_leg_open_interest_min
        and row[0].volume >= config.short_leg_volume_min
        and row[3] <= config.max_short_leg_spread_pct_of_mid
    ]
    long_candidates = [
        row for row in put_rows
        if _valid_common(row[0])
        and config.dte_min <= row[0].dte <= config.dte_max
        and row[0].delta <= -config.long_put_delta_min_abs
        and row[0].delta >= -config.long_put_delta_max_abs
        and row[3] <= config.max_long_leg_spread_pct_of_mid
    ]
    diagnostics["eligible_short_legs"] = len(short_candidates)
    diagnostics["eligible_long_legs"] = len(long_candidates)
    if not short_candidates or not long_candidates:
        return None, diagnostics

    pair_candidates: list[BullPutSpreadCandidate] = []
    for short in short_candidates:
        sq = short[0]
        for long in long_candidates:
            lq = long[0]
            if lq.expiry != sq.expiry or lq.strike >= sq.strike:
                continue
            cand = _build_candidate(
                short=short, long=long, as_of_date=session_date, symbol=symbol_code,
                nav_cap_pct=nav_cap_pct, config=config, iv_rank=iv_rank, iv_percentile=iv_percentile, beta=beta)
            if cand is not None:
                pair_candidates.append(cand)

    diagnostics["eligible_spread_pairs"] = len(pair_candidates)
    if not pair_candidates:
        return None, diagnostics

    ranked = sorted(pair_candidates, key=lambda c: (
        abs(abs(c.short_delta) - config.target_short_put_delta_abs),
        abs(abs(c.long_delta) - config.target_long_put_delta_abs),
        c.structure_spread_pct_of_mid,
        -(c.structure_mid / c.width if c.width > 0 else 0.0),
        -c.short_mid,
    ))
    winner = ranked[0]
    winner.diagnostics["eligible_spread_pairs"] = len(pair_candidates)
    winner.diagnostics["selection_rank"] = 1
    return winner, diagnostics


def evaluate_bull_put_spread_exit_action(
    *,
    entry_credit: float,
    current_close_cost: float,
    short_strike: float,
    spot_price: float,
    dte_remaining: int,
) -> BullPutSpreadExitDecision:
    """Exit stack: profit-take (half credit) -> stop-loss (2x) -> time exit
    (<=7 DTE) -> breach exit (spot below short strike inside the expiry window).

    The time/breach controls address the "slow drift into expiry" case that a
    2x stop-loss alone does not catch.
    """
    if entry_credit <= 0:
        raise ValueError("entry_credit must be positive")
    if current_close_cost < 0:
        raise ValueError("current_close_cost must be non-negative")
    if short_strike <= 0:
        raise ValueError("short_strike must be positive")
    if spot_price <= 0:
        raise ValueError("spot_price must be positive")
    if dte_remaining < 0:
        raise ValueError("dte_remaining must be non-negative")

    profit_take_threshold = entry_credit * 0.5
    stop_loss_threshold = entry_credit * 2.0
    profit_take_condition = current_close_cost <= profit_take_threshold
    stop_loss_condition = current_close_cost >= stop_loss_threshold
    time_exit_condition = dte_remaining <= 7
    breach_condition = spot_price < short_strike and dte_remaining <= 10
    diagnostics = {
        "entry_credit": entry_credit, "current_close_cost": current_close_cost,
        "short_strike": short_strike, "spot_price": spot_price, "dte_remaining": dte_remaining,
        "profit_take_close_cost_threshold": profit_take_threshold,
        "stop_loss_close_cost_threshold": stop_loss_threshold,
        "profit_take_condition": profit_take_condition, "stop_loss_condition": stop_loss_condition,
        "time_exit_condition": time_exit_condition, "breach_condition": breach_condition,
    }
    if profit_take_condition:
        return BullPutSpreadExitDecision("close_profit_take", True, diagnostics)
    if stop_loss_condition:
        return BullPutSpreadExitDecision("close_stop_loss", True, diagnostics)
    if time_exit_condition:
        return BullPutSpreadExitDecision("close_time_exit", True, diagnostics)
    if breach_condition:
        return BullPutSpreadExitDecision("close_breach_exit", True, diagnostics)
    return BullPutSpreadExitDecision("hold", False, diagnostics)

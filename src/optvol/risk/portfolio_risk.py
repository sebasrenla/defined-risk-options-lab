"""Portfolio-level risk overlay.

Sleeve scanners propose candidates in isolation; this layer decides how many (if
any) contracts may actually be opened given the *whole book*. It sits between
signal generation and execution and enforces, in one place:

* an **aggregate defined-risk cap** (% of NAV across all defined-risk positions),
* a **per-underlying cap** (with per-symbol overrides, e.g. a tighter cap on a
  high-volatility name),
* a **sector cap**,
* **per-sleeve caps** (covered-call notional; bull-put-spread risk),
* a **max-new-entries-per-run** throttle, and
* a **VIX-regime exposure cut**: when volatility is elevated, every candidate's
  effective size is scaled down by a multiplier (risk-off sizing).

Sizing is greedy and stateful: candidates are evaluated in order, each accepted
fill consumes capacity, and the next candidate sees the reduced remaining
headroom. Every decision (accept or reject) is returned with reason codes and
diagnostics for a full audit trail.

Decoupling note
---------------
The original engine consumed the scanners' rich candidate objects directly. Here
the engine depends only on small **risk-input** dataclasses carrying the few
fields the risk math needs, so it is independent of any particular scanner.

Provenance
----------
Ported from the program's ``portfolio_risk_engine.py`` (candidate types
decoupled into local risk-input dataclasses; risk logic unchanged).
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REJECT_RUN_ENTRY_LIMIT_REACHED = "run_entry_limit_reached"
REJECT_SINGLE_UNDERLYING_CAP = "single_underlying_cap"
REJECT_SECTOR_CAP = "sector_cap"
REJECT_CC_SLEEVE_CAP = "cc_sleeve_cap"
REJECT_BPS_SLEEVE_CAP = "bps_sleeve_cap"
REJECT_AGGREGATE_DEFINED_RISK_CAP = "aggregate_defined_risk_cap"
REJECT_MISSING_SYMBOL_CONTEXT = "missing_symbol_context"
REJECT_INSUFFICIENT_CAPACITY_FOR_MIN_SIZE = "insufficient_capacity_for_min_size"


# --- Risk inputs (decoupled from the scanners) -------------------------------

@dataclass(frozen=True)
class CoveredCallRiskInput:
    """The fields the risk layer needs from a covered-call candidate."""

    symbol: str
    nav_cap_pct: float
    allocation_multiplier: float
    underlying_price: float


@dataclass(frozen=True)
class BullPutSpreadRiskInput:
    """The fields the risk layer needs from a bull-put-spread candidate."""

    symbol: str
    nav_cap_pct: float
    allocation_multiplier: float
    max_loss_model: float
    estimated_roundtrip_fees: float


@dataclass(frozen=True)
class TailHedgeCandidateInput:
    symbol: str
    recommended_contracts: int
    total_estimated_entry_outlay: float
    total_estimated_notional: float
    purchase_mode: str = ""


@dataclass(frozen=True)
class TailHedgePlan:
    """Either an accepted tail-hedge candidate or a rejection with reasons."""

    candidate: TailHedgeCandidateInput | None = None
    reject_reason_codes: tuple[str, ...] = ()
    reject_diagnostics: dict[str, Any] = field(default_factory=dict)


Program1RiskInput = CoveredCallRiskInput | BullPutSpreadRiskInput


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _normalize_pct(value: float) -> float:
    """Accept either a fraction (0.04) or a percent (4.0); return a fraction."""
    return value / 100.0 if value > 1.0 else value


@dataclass(frozen=True)
class PortfolioRiskConfig:
    aggregate_defined_risk_cap_pct_nav: float = 0.35
    single_underlying_cap_pct_nav: float = 0.04
    sector_cap_pct_nav: float = 0.20
    cc_sleeve_notional_cap_pct_nav: float = 0.40
    bps_sleeve_risk_cap_pct_nav: float = 0.20
    max_new_program1_entries_per_run: int = 3
    vix_regime_cut_threshold: float | None = None
    vix_regime_cut_multiplier: float = 0.50
    symbol_cap_overrides_pct_nav: Mapping[str, float] = field(default_factory=lambda: {"TSLA": 0.02})
    tail_hedge_counts_toward_entry_limit: bool = False


@dataclass(frozen=True)
class ExistingPositionRisk:
    symbol: str
    sector: str
    sub_sleeve: str
    exposure_usd: float
    defined_risk_usd: float
    cc_notional_usd: float
    bps_risk_usd: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioExposureState:
    nav: float
    aggregate_defined_risk_used_usd: float
    cc_notional_used_usd: float
    bps_risk_used_usd: float
    symbol_risk_used_usd: dict[str, float]
    sector_risk_used_usd: dict[str, float]
    open_positions: int
    new_program1_entries_accepted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "nav": self.nav,
            "aggregate_defined_risk_used_usd": self.aggregate_defined_risk_used_usd,
            "cc_notional_used_usd": self.cc_notional_used_usd,
            "bps_risk_used_usd": self.bps_risk_used_usd,
            "symbol_risk_used_usd": dict(self.symbol_risk_used_usd),
            "sector_risk_used_usd": dict(self.sector_risk_used_usd),
            "open_positions": self.open_positions,
            "new_program1_entries_accepted": self.new_program1_entries_accepted,
        }


@dataclass(frozen=True)
class PortfolioRiskDecision:
    as_of_date: date
    sub_sleeve: str
    symbol: str
    sector: str
    action_code: str
    approved_quantity: int
    effective_allocation_multiplier: float
    incremental_risk_usd: float
    incremental_notional_usd: float
    reason_codes: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["as_of_date"] = self.as_of_date.isoformat()
        payload["reason_codes"] = "|".join(self.reason_codes)
        payload["diagnostics"] = repr(self.diagnostics)
        return payload


@dataclass(frozen=True)
class PortfolioRiskEvaluation:
    as_of_date: date
    vix_level: float
    regime_cut_applied: bool
    regime_cut_multiplier: float
    accepted_program1: tuple[PortfolioRiskDecision, ...]
    rejected_program1: tuple[PortfolioRiskDecision, ...]
    tail_hedge_decision: PortfolioRiskDecision | None
    starting_state: PortfolioExposureState
    ending_state: PortfolioExposureState
    diagnostics: dict[str, Any] = field(default_factory=dict)


def build_symbol_context_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
    context_map: dict[str, dict[str, str]] = {}
    for row in rows:
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        context_map[symbol] = {
            "sector": str(row.get("sector", "")).strip() or "Unknown",
            "industry": str(row.get("industry", "")).strip() or "",
        }
    return context_map


def build_existing_position_risk(
    position_rows: Iterable[Mapping[str, Any]],
    *,
    symbol_context: Mapping[str, Mapping[str, Any]],
) -> tuple[ExistingPositionRisk, ...]:
    slices: list[ExistingPositionRisk] = []
    for row in position_rows:
        if str(row.get("exit_timestamp", "") or "").strip():
            continue
        symbol = str(row.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        sub_sleeve = str(row.get("sub_sleeve", "")).strip().lower()
        option_type = str(row.get("option_type", "")).strip().lower()
        sector = str(symbol_context.get(symbol, {}).get("sector", "Unknown")).strip() or "Unknown"
        size = float(row.get("size", 0) or 0)
        entry_price = float(row.get("entry_price", 0) or 0)
        risk_at_entry = float(row.get("risk_at_entry", 0) or 0)

        if sub_sleeve == "covered_call" and option_type == "equity":
            cc_notional = max(entry_price * size, 0.0)
            slices.append(ExistingPositionRisk(
                symbol=symbol, sector=sector, sub_sleeve=sub_sleeve,
                exposure_usd=cc_notional, defined_risk_usd=0.0,
                cc_notional_usd=cc_notional, bps_risk_usd=0.0,
            ))
            continue

        defined_risk = max(risk_at_entry, 0.0)
        bps_risk = defined_risk if sub_sleeve == "bull_put_spread" else 0.0
        slices.append(ExistingPositionRisk(
            symbol=symbol, sector=sector, sub_sleeve=sub_sleeve,
            exposure_usd=defined_risk, defined_risk_usd=defined_risk,
            cc_notional_usd=0.0, bps_risk_usd=bps_risk,
        ))
    return tuple(slices)


def summarize_existing_exposure(
    *, nav: float, existing_positions: Sequence[ExistingPositionRisk]
) -> PortfolioExposureState:
    symbol_risk_used_usd: defaultdict[str, float] = defaultdict(float)
    sector_risk_used_usd: defaultdict[str, float] = defaultdict(float)
    aggregate = cc_notional = bps_risk = 0.0
    for position in existing_positions:
        aggregate += position.defined_risk_usd
        cc_notional += position.cc_notional_usd
        bps_risk += position.bps_risk_usd
        symbol_risk_used_usd[position.symbol] += position.exposure_usd
        sector_risk_used_usd[position.sector] += position.exposure_usd
    return PortfolioExposureState(
        nav=float(nav),
        aggregate_defined_risk_used_usd=float(aggregate),
        cc_notional_used_usd=float(cc_notional),
        bps_risk_used_usd=float(bps_risk),
        symbol_risk_used_usd=dict(symbol_risk_used_usd),
        sector_risk_used_usd=dict(sector_risk_used_usd),
        open_positions=len(existing_positions),
        new_program1_entries_accepted=0,
    )


def _detect_regime_cut(vix_level: float, config: PortfolioRiskConfig) -> tuple[bool, float]:
    threshold = config.vix_regime_cut_threshold
    if threshold is None:
        return False, 1.0
    if vix_level >= threshold:
        return True, config.vix_regime_cut_multiplier
    return False, 1.0


def _resolve_symbol_cap_pct(symbol: str, candidate_nav_cap_pct: float, config: PortfolioRiskConfig) -> float:
    candidate_cap = _normalize_pct(candidate_nav_cap_pct)
    baseline_cap = _normalize_pct(config.single_underlying_cap_pct_nav)
    override_cap = config.symbol_cap_overrides_pct_nav.get(symbol.upper().strip())
    if override_cap is not None:
        baseline_cap = min(baseline_cap, _normalize_pct(float(override_cap)))
    return min(candidate_cap, baseline_cap)


def _copy_state(state: PortfolioExposureState) -> PortfolioExposureState:
    return _replace_state(state)


def _replace_state(
    state: PortfolioExposureState,
    *,
    aggregate_defined_risk_used_usd: float | None = None,
    cc_notional_used_usd: float | None = None,
    bps_risk_used_usd: float | None = None,
    symbol_risk_used_usd: dict[str, float] | None = None,
    sector_risk_used_usd: dict[str, float] | None = None,
    new_program1_entries_accepted: int | None = None,
) -> PortfolioExposureState:
    return PortfolioExposureState(
        nav=state.nav,
        aggregate_defined_risk_used_usd=(
            state.aggregate_defined_risk_used_usd if aggregate_defined_risk_used_usd is None
            else float(aggregate_defined_risk_used_usd)
        ),
        cc_notional_used_usd=state.cc_notional_used_usd if cc_notional_used_usd is None else float(cc_notional_used_usd),
        bps_risk_used_usd=state.bps_risk_used_usd if bps_risk_used_usd is None else float(bps_risk_used_usd),
        symbol_risk_used_usd=dict(state.symbol_risk_used_usd) if symbol_risk_used_usd is None else dict(symbol_risk_used_usd),
        sector_risk_used_usd=dict(state.sector_risk_used_usd) if sector_risk_used_usd is None else dict(sector_risk_used_usd),
        open_positions=state.open_positions,
        new_program1_entries_accepted=(
            state.new_program1_entries_accepted if new_program1_entries_accepted is None
            else int(new_program1_entries_accepted)
        ),
    )


def _evaluate_covered_call_candidate(
    *, candidate: CoveredCallRiskInput, as_of_date: date, nav: float, sector: str,
    regime_multiplier: float, regime_cut_applied: bool, state: PortfolioExposureState,
    config: PortfolioRiskConfig,
) -> tuple[PortfolioRiskDecision, PortfolioExposureState | None]:
    symbol_cap_pct = _resolve_symbol_cap_pct(candidate.symbol, candidate.nav_cap_pct, config)
    effective_allocation_multiplier = candidate.allocation_multiplier * regime_multiplier
    effective_symbol_cap_usd = nav * symbol_cap_pct * effective_allocation_multiplier
    per_lot_notional = candidate.underlying_price * 100.0
    remaining_symbol = effective_symbol_cap_usd - state.symbol_risk_used_usd.get(candidate.symbol, 0.0)
    remaining_sector = nav * config.sector_cap_pct_nav - state.sector_risk_used_usd.get(sector, 0.0)
    remaining_cc = nav * config.cc_sleeve_notional_cap_pct_nav - state.cc_notional_used_usd
    allowed_usd = min(remaining_symbol, remaining_sector, remaining_cc)
    quantity = int(max(allowed_usd, 0.0) // per_lot_notional) if per_lot_notional > 0 else 0

    diagnostics = {
        "candidate_nav_cap_pct": candidate.nav_cap_pct,
        "symbol_cap_pct_applied": symbol_cap_pct,
        "effective_symbol_cap_usd": effective_symbol_cap_usd,
        "remaining_symbol_usd": remaining_symbol,
        "remaining_sector_usd": remaining_sector,
        "remaining_cc_sleeve_usd": remaining_cc,
        "per_lot_notional_usd": per_lot_notional,
        "regime_cut_applied": regime_cut_applied,
    }

    if quantity < 1:
        reason_codes: list[str] = []
        if sector == "Unknown":
            reason_codes.append(REJECT_MISSING_SYMBOL_CONTEXT)
        if remaining_symbol < per_lot_notional:
            reason_codes.append(REJECT_SINGLE_UNDERLYING_CAP)
        if remaining_sector < per_lot_notional:
            reason_codes.append(REJECT_SECTOR_CAP)
        if remaining_cc < per_lot_notional:
            reason_codes.append(REJECT_CC_SLEEVE_CAP)
        reason_codes.append(REJECT_INSUFFICIENT_CAPACITY_FOR_MIN_SIZE)
        return PortfolioRiskDecision(
            as_of_date=as_of_date, sub_sleeve="covered_call", symbol=candidate.symbol, sector=sector,
            action_code="reject", approved_quantity=0,
            effective_allocation_multiplier=effective_allocation_multiplier,
            incremental_risk_usd=0.0, incremental_notional_usd=0.0,
            reason_codes=tuple(dict.fromkeys(reason_codes)), diagnostics=diagnostics,
        ), None

    incremental_notional = quantity * per_lot_notional
    symbol_risk_used_usd = dict(state.symbol_risk_used_usd)
    sector_risk_used_usd = dict(state.sector_risk_used_usd)
    symbol_risk_used_usd[candidate.symbol] = symbol_risk_used_usd.get(candidate.symbol, 0.0) + incremental_notional
    sector_risk_used_usd[sector] = sector_risk_used_usd.get(sector, 0.0) + incremental_notional

    next_state = _replace_state(
        state, cc_notional_used_usd=state.cc_notional_used_usd + incremental_notional,
        symbol_risk_used_usd=symbol_risk_used_usd, sector_risk_used_usd=sector_risk_used_usd,
        new_program1_entries_accepted=state.new_program1_entries_accepted + 1,
    )
    return PortfolioRiskDecision(
        as_of_date=as_of_date, sub_sleeve="covered_call", symbol=candidate.symbol, sector=sector,
        action_code="accept", approved_quantity=quantity,
        effective_allocation_multiplier=effective_allocation_multiplier,
        incremental_risk_usd=incremental_notional, incremental_notional_usd=incremental_notional,
        diagnostics=diagnostics,
    ), next_state


def _evaluate_bull_put_spread_candidate(
    *, candidate: BullPutSpreadRiskInput, as_of_date: date, nav: float, sector: str,
    regime_multiplier: float, regime_cut_applied: bool, state: PortfolioExposureState,
    config: PortfolioRiskConfig,
) -> tuple[PortfolioRiskDecision, PortfolioExposureState | None]:
    symbol_cap_pct = _resolve_symbol_cap_pct(candidate.symbol, candidate.nav_cap_pct, config)
    effective_allocation_multiplier = candidate.allocation_multiplier * regime_multiplier
    effective_symbol_cap_usd = nav * symbol_cap_pct * effective_allocation_multiplier
    per_contract_risk = candidate.max_loss_model + candidate.estimated_roundtrip_fees
    remaining_symbol = effective_symbol_cap_usd - state.symbol_risk_used_usd.get(candidate.symbol, 0.0)
    remaining_sector = nav * config.sector_cap_pct_nav - state.sector_risk_used_usd.get(sector, 0.0)
    remaining_bps = nav * config.bps_sleeve_risk_cap_pct_nav - state.bps_risk_used_usd
    remaining_defined = nav * config.aggregate_defined_risk_cap_pct_nav - state.aggregate_defined_risk_used_usd
    allowed_usd = min(remaining_symbol, remaining_sector, remaining_bps, remaining_defined)
    quantity = int(max(allowed_usd, 0.0) // per_contract_risk) if per_contract_risk > 0 else 0

    diagnostics = {
        "candidate_nav_cap_pct": candidate.nav_cap_pct,
        "symbol_cap_pct_applied": symbol_cap_pct,
        "effective_symbol_cap_usd": effective_symbol_cap_usd,
        "remaining_symbol_usd": remaining_symbol,
        "remaining_sector_usd": remaining_sector,
        "remaining_bps_sleeve_usd": remaining_bps,
        "remaining_aggregate_defined_risk_usd": remaining_defined,
        "per_contract_risk_usd": per_contract_risk,
        "regime_cut_applied": regime_cut_applied,
    }

    if quantity < 1:
        reason_codes: list[str] = []
        if sector == "Unknown":
            reason_codes.append(REJECT_MISSING_SYMBOL_CONTEXT)
        if remaining_symbol < per_contract_risk:
            reason_codes.append(REJECT_SINGLE_UNDERLYING_CAP)
        if remaining_sector < per_contract_risk:
            reason_codes.append(REJECT_SECTOR_CAP)
        if remaining_bps < per_contract_risk:
            reason_codes.append(REJECT_BPS_SLEEVE_CAP)
        if remaining_defined < per_contract_risk:
            reason_codes.append(REJECT_AGGREGATE_DEFINED_RISK_CAP)
        reason_codes.append(REJECT_INSUFFICIENT_CAPACITY_FOR_MIN_SIZE)
        return PortfolioRiskDecision(
            as_of_date=as_of_date, sub_sleeve="bull_put_spread", symbol=candidate.symbol, sector=sector,
            action_code="reject", approved_quantity=0,
            effective_allocation_multiplier=effective_allocation_multiplier,
            incremental_risk_usd=0.0, incremental_notional_usd=0.0,
            reason_codes=tuple(dict.fromkeys(reason_codes)), diagnostics=diagnostics,
        ), None

    incremental_risk = quantity * per_contract_risk
    symbol_risk_used_usd = dict(state.symbol_risk_used_usd)
    sector_risk_used_usd = dict(state.sector_risk_used_usd)
    symbol_risk_used_usd[candidate.symbol] = symbol_risk_used_usd.get(candidate.symbol, 0.0) + incremental_risk
    sector_risk_used_usd[sector] = sector_risk_used_usd.get(sector, 0.0) + incremental_risk

    next_state = _replace_state(
        state, aggregate_defined_risk_used_usd=state.aggregate_defined_risk_used_usd + incremental_risk,
        bps_risk_used_usd=state.bps_risk_used_usd + incremental_risk,
        symbol_risk_used_usd=symbol_risk_used_usd, sector_risk_used_usd=sector_risk_used_usd,
        new_program1_entries_accepted=state.new_program1_entries_accepted + 1,
    )
    return PortfolioRiskDecision(
        as_of_date=as_of_date, sub_sleeve="bull_put_spread", symbol=candidate.symbol, sector=sector,
        action_code="accept", approved_quantity=quantity,
        effective_allocation_multiplier=effective_allocation_multiplier,
        incremental_risk_usd=incremental_risk, incremental_notional_usd=0.0,
        diagnostics=diagnostics,
    ), next_state


def _evaluate_tail_hedge_passthrough(
    *, as_of_date: date, tail_hedge_plan: TailHedgePlan | None
) -> PortfolioRiskDecision | None:
    if tail_hedge_plan is None:
        return None
    if tail_hedge_plan.candidate is not None:
        candidate = tail_hedge_plan.candidate
        return PortfolioRiskDecision(
            as_of_date=as_of_date, sub_sleeve="tail_hedge", symbol=candidate.symbol, sector="TailHedge",
            action_code="pass_through", approved_quantity=candidate.recommended_contracts,
            effective_allocation_multiplier=1.0,
            incremental_risk_usd=candidate.total_estimated_entry_outlay,
            incremental_notional_usd=candidate.total_estimated_notional,
            diagnostics={"purchase_mode": candidate.purchase_mode,
                         "recommended_contracts": candidate.recommended_contracts},
        )
    return PortfolioRiskDecision(
        as_of_date=as_of_date, sub_sleeve="tail_hedge", symbol="", sector="TailHedge",
        action_code="tail_plan_reject", approved_quantity=0, effective_allocation_multiplier=1.0,
        incremental_risk_usd=0.0, incremental_notional_usd=0.0,
        reason_codes=tail_hedge_plan.reject_reason_codes,
        diagnostics=tail_hedge_plan.reject_diagnostics,
    )


def _build_portfolio_diagnostics(
    *, ending_state: PortfolioExposureState, tail_hedge_decision: PortfolioRiskDecision | None
) -> dict[str, Any]:
    short_premium_exposure = ending_state.cc_notional_used_usd + ending_state.bps_risk_used_usd
    tail_hedge_notional = (
        tail_hedge_decision.incremental_notional_usd
        if tail_hedge_decision is not None and tail_hedge_decision.action_code == "pass_through" else 0.0
    )
    coverage_ratio = (
        (tail_hedge_notional / short_premium_exposure)
        if short_premium_exposure > 0 and tail_hedge_notional > 0 else None
    )
    return {
        "short_premium_exposure_usd": short_premium_exposure,
        "tail_hedge_notional_usd": tail_hedge_notional,
        "tail_hedge_coverage_ratio": coverage_ratio,
    }


def evaluate_portfolio_risk_layer(
    *,
    as_of_date: date | datetime | str,
    nav: float,
    vix_level: float,
    program1_candidates: Sequence[Program1RiskInput],
    symbol_context: Mapping[str, Mapping[str, Any]],
    existing_positions: Sequence[ExistingPositionRisk] = (),
    tail_hedge_plan: TailHedgePlan | None = None,
    config: PortfolioRiskConfig = PortfolioRiskConfig(),
) -> PortfolioRiskEvaluation:
    """Evaluate all candidates against portfolio caps and return a full decision set."""
    resolved_date = _coerce_date(as_of_date)
    starting_state = summarize_existing_exposure(nav=nav, existing_positions=existing_positions)
    working_state = _copy_state(starting_state)
    regime_cut_applied, regime_multiplier = _detect_regime_cut(vix_level, config)

    accepted: list[PortfolioRiskDecision] = []
    rejected: list[PortfolioRiskDecision] = []

    for candidate in program1_candidates:
        symbol = candidate.symbol.upper().strip()
        sector = str(symbol_context.get(symbol, {}).get("sector", "Unknown")).strip() or "Unknown"

        if working_state.new_program1_entries_accepted >= config.max_new_program1_entries_per_run:
            rejected.append(PortfolioRiskDecision(
                as_of_date=resolved_date,
                sub_sleeve=("covered_call" if isinstance(candidate, CoveredCallRiskInput) else "bull_put_spread"),
                symbol=symbol, sector=sector, action_code="reject", approved_quantity=0,
                effective_allocation_multiplier=(candidate.allocation_multiplier * regime_multiplier),
                incremental_risk_usd=0.0, incremental_notional_usd=0.0,
                reason_codes=(REJECT_RUN_ENTRY_LIMIT_REACHED,),
                diagnostics={"max_new_program1_entries_per_run": config.max_new_program1_entries_per_run},
            ))
            continue

        if isinstance(candidate, CoveredCallRiskInput):
            decision, next_state = _evaluate_covered_call_candidate(
                candidate=candidate, as_of_date=resolved_date, nav=nav, sector=sector,
                regime_multiplier=regime_multiplier, regime_cut_applied=regime_cut_applied,
                state=working_state, config=config,
            )
        elif isinstance(candidate, BullPutSpreadRiskInput):
            decision, next_state = _evaluate_bull_put_spread_candidate(
                candidate=candidate, as_of_date=resolved_date, nav=nav, sector=sector,
                regime_multiplier=regime_multiplier, regime_cut_applied=regime_cut_applied,
                state=working_state, config=config,
            )
        else:
            raise TypeError(f"Unsupported candidate type: {type(candidate)!r}")

        if next_state is None:
            rejected.append(decision)
            continue
        accepted.append(decision)
        working_state = next_state

    tail_hedge_decision = _evaluate_tail_hedge_passthrough(as_of_date=resolved_date, tail_hedge_plan=tail_hedge_plan)
    diagnostics = _build_portfolio_diagnostics(ending_state=working_state, tail_hedge_decision=tail_hedge_decision)
    return PortfolioRiskEvaluation(
        as_of_date=resolved_date, vix_level=float(vix_level),
        regime_cut_applied=regime_cut_applied, regime_cut_multiplier=regime_multiplier,
        accepted_program1=tuple(accepted), rejected_program1=tuple(rejected),
        tail_hedge_decision=tail_hedge_decision,
        starting_state=starting_state, ending_state=working_state, diagnostics=diagnostics,
    )


def write_portfolio_risk_decisions(path: Path, decisions: Sequence[PortfolioRiskDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "as_of_date", "sub_sleeve", "symbol", "sector", "action_code", "approved_quantity",
        "effective_allocation_multiplier", "incremental_risk_usd", "incremental_notional_usd",
        "reason_codes", "diagnostics",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in decisions:
            writer.writerow(row.to_dict())

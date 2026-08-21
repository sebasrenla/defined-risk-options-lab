"""Cash-ledger economics for an options book.

The right way to measure a defined-risk options program's P&L is a **cash
ledger**, not a sum of per-leg realized P&L. The two disagree in exactly the
cases that matter: when a bull-put-spread short leg is assigned, the per-leg
realized P&L captures only the equity loss, while the opening credit lives on in
the cash ledger, so leg-summing understates the position. This module builds
every real cash flow of the book as immutable ledger events:

* **Trades**: equity and option buys/sells become two rows each (a cash
  movement + a fees row), with settlement dated by a trading calendar.
* **Assignment / exercise**: covered-call assignment, short-put assignment, and
  long-put exercise each generate the equity leg plus an assignment fee.
* **Dividends**: an ex-date receivable accrual and a pay-date cash receipt (or a
  single ex-date booking when no pay date is known).
* **Margin interest**: a tiered borrow rate applied to any settled-cash debit,
  accrued daily and posted the next session.

Snapshots then partition events into **settled / unsettled / receivable**
balances as of a session date, and compute a margin/NLV/excess-liquidity view.

Settlement calendar
-------------------
Settlement uses business-day lags (equity T+3 -> T+2 -> T+1 across the historical
schedule; options T+1). The trading calendar is **injectable**: the default is a
holiday-free weekday calendar so the module runs with no external dependencies;
for holiday-accurate settlement, pass a calendar built from a real exchange
calendar via :func:`exchange_calendar_sessions`.

Provenance
----------
Ported from the program's ``replay_economics.py``; the pandas/exchange_calendars
settlement internals are replaced by the injectable :class:`SessionCalendar`.
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Literal, Sequence

from ..execution.fees_regulatory import (
    CURRENT_EQUITY_CLEARING_PER_SHARE,
    CURRENT_EQUITY_OPEN_COMMISSION_PER_ORDER,
    CURRENT_OPTION_CLEARING_PER_CONTRACT,
    CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT,
    CURRENT_OPTION_COMMISSION_CAP_PER_LEG,
    CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT,
    CURRENT_OPTION_ORF_PER_CONTRACT,
    estimate_equity_transaction_fees,
    estimate_option_transaction_fees,
)

FundingPolicy = Literal["cash_first_broker_margin"]
AssetClass = Literal["equity", "equity_option"]

DEFAULT_MARGIN_BASE_RATE = 0.10
DEFAULT_ASSIGNMENT_FEE_USD = 5.00

# Margin rate spread to the base rate, by settled-debit tier.
MARGIN_RATE_SPREADS_BY_TIER: tuple[tuple[float, float | None, float], ...] = (
    (0.0, 25_000.0, 0.0100),
    (25_000.0, 50_000.0, 0.0050),
    (50_000.0, 100_000.0, 0.0000),
    (100_000.0, 250_000.0, -0.0050),
    (250_000.0, 500_000.0, -0.0100),
    (500_000.0, 1_000_000.0, -0.0150),
    (1_000_000.0, None, -0.0200),
)

# Equity settlement shortened T+3 -> T+2 (2017) -> T+1 (2024); options are T+1.
EQUITY_SETTLEMENT_LAG_SCHEDULE: tuple[tuple[date, int], ...] = (
    (date(1900, 1, 1), 3),
    (date(2017, 9, 5), 2),
    (date(2024, 5, 28), 1),
)
OPTION_SETTLEMENT_LAG_SCHEDULE: tuple[tuple[date, int], ...] = (
    (date(1900, 1, 1), 1),
)


# --- Injectable trading calendar ---------------------------------------------

def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


class SessionCalendar:
    """A sorted set of trading-session dates with settlement arithmetic."""

    def __init__(self, sessions: Sequence[date]):
        self._sessions = sorted({_coerce_date(s) for s in sessions})

    @property
    def sessions(self) -> list[date]:
        return list(self._sessions)

    def _index_of(self, session_date: date) -> int:
        idx = bisect.bisect_left(self._sessions, session_date)
        if idx >= len(self._sessions) or self._sessions[idx] != session_date:
            raise ValueError(f"{session_date.isoformat()} is not a valid trading session")
        return idx

    def settlement_date(self, trade_date: date, lag: int) -> date:
        idx = self._index_of(trade_date)
        if idx + lag >= len(self._sessions):
            raise ValueError(f"{trade_date.isoformat()} is too close to the end of the calendar")
        return self._sessions[idx + lag]

    def next_session(self, session_date: date) -> date:
        idx = self._index_of(session_date)
        if idx + 1 >= len(self._sessions):
            raise ValueError(f"{session_date.isoformat()} is too close to the end of the calendar")
        return self._sessions[idx + 1]


def weekday_calendar(start: date, end: date) -> SessionCalendar:
    """A holiday-free Monday-Friday calendar spanning ``[start, end]``."""
    days: list[date] = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon-Fri
            days.append(current)
        current += timedelta(days=1)
    return SessionCalendar(days)


def exchange_calendar_sessions(
    name: str = "XNYS", start: date | str = "2015-01-01", end: date | str = "2035-12-31"
) -> SessionCalendar:
    """Build a holiday-accurate calendar from ``exchange_calendars`` (optional dep)."""
    import exchange_calendars as xcals  # imported lazily so it stays optional

    cal = xcals.get_calendar(name)
    sessions = [ts.date() for ts in cal.sessions if _coerce_date(start) <= ts.date() <= _coerce_date(end)]
    return SessionCalendar(sessions)


# Default: dependency-free weekday calendar. Replace for holiday accuracy.
DEFAULT_CALENDAR = weekday_calendar(date(2015, 1, 1), date(2035, 12, 31))


def _resolve_business_day_lag(trade_date: date | datetime | str, schedule: Sequence[tuple[date, int]]) -> int:
    resolved = _coerce_date(trade_date)
    lag = schedule[0][1]
    for effective_date, effective_lag in schedule:
        if resolved >= effective_date:
            lag = effective_lag
    return lag


def resolve_settlement_business_days(trade_date: date | datetime | str, *, asset_class: AssetClass) -> int:
    if asset_class == "equity":
        return _resolve_business_day_lag(trade_date, EQUITY_SETTLEMENT_LAG_SCHEDULE)
    if asset_class == "equity_option":
        return _resolve_business_day_lag(trade_date, OPTION_SETTLEMENT_LAG_SCHEDULE)
    raise ValueError(f"Unsupported asset_class: {asset_class}")


def resolve_settlement_date(
    trade_date: date | datetime | str, *, asset_class: AssetClass, calendar: SessionCalendar = DEFAULT_CALENDAR
) -> date:
    resolved = _coerce_date(trade_date)
    lag = resolve_settlement_business_days(resolved, asset_class=asset_class)
    return calendar.settlement_date(resolved, lag)


def resolve_next_session_date(
    session_date: date | datetime | str, *, calendar: SessionCalendar = DEFAULT_CALENDAR
) -> date:
    return calendar.next_session(_coerce_date(session_date))


# --- Margin -------------------------------------------------------------------

@dataclass(frozen=True)
class MarginRateTier:
    debit_floor_usd: float
    debit_ceiling_usd: float | None
    annual_rate: float
    spread_to_base_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_margin_rate_tiers(base_rate: float) -> tuple[MarginRateTier, ...]:
    """Tiered margin schedule: a spread to ``base_rate`` per settled-debit band."""
    return tuple(
        MarginRateTier(
            debit_floor_usd=floor,
            debit_ceiling_usd=ceiling,
            annual_rate=max(base_rate + spread, 0.0),
            spread_to_base_rate=spread,
        )
        for floor, ceiling, spread in MARGIN_RATE_SPREADS_BY_TIER
    )


def resolve_margin_interest_rate(
    debit_balance_usd: float, *, margin_rate_tiers: Sequence[MarginRateTier] | None = None
) -> float:
    if debit_balance_usd <= 0:
        return 0.0
    tiers = tuple(margin_rate_tiers or build_margin_rate_tiers(DEFAULT_MARGIN_BASE_RATE))
    for tier in tiers:
        ceiling = float("inf") if tier.debit_ceiling_usd is None else tier.debit_ceiling_usd
        if tier.debit_floor_usd <= debit_balance_usd < ceiling:
            return tier.annual_rate
    return tiers[-1].annual_rate if tiers else 0.0


def estimate_daily_margin_interest(debit_balance_usd: float, *, annual_rate: float, day_basis: int = 360) -> float:
    debit = max(float(debit_balance_usd), 0.0)
    if debit == 0.0 or annual_rate <= 0.0:
        return 0.0
    if day_basis <= 0:
        raise ValueError("day_basis must be positive")
    return debit * annual_rate / day_basis


@dataclass(frozen=True)
class ReplayEconomicsConfig:
    funding_policy: FundingPolicy = "cash_first_broker_margin"
    margin_interest_day_basis: int = 360
    margin_interest_compounds_daily: bool = True
    long_equity_initial_requirement_pct: float = 0.50
    long_option_requirement_pct: float = 1.00
    defined_risk_requirement_pct: float = 1.00
    assignment_fee_usd: float = DEFAULT_ASSIGNMENT_FEE_USD
    margin_rate_tiers: tuple[MarginRateTier, ...] = field(
        default_factory=lambda: build_margin_rate_tiers(DEFAULT_MARGIN_BASE_RATE)
    )


# --- Ledger events ------------------------------------------------------------

@dataclass(frozen=True)
class EconomicsLedgerEvent:
    event_type: str
    symbol: str
    sub_sleeve: str
    effective_date: date
    settlement_date: date
    cash_effect_usd: float
    receivable_effect_usd: float = 0.0
    position_group_id: str | None = None
    description: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["effective_date"] = self.effective_date.isoformat()
        payload["settlement_date"] = self.settlement_date.isoformat()
        payload["diagnostics"] = repr(self.diagnostics)
        return payload


def build_equity_trade_event(
    *, trade_date: date | datetime | str, symbol: str, sub_sleeve: str, shares: int,
    fill_price: float, side: Literal["buy", "sell"], position_group_id: str | None = None,
    calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, EconomicsLedgerEvent]:
    """One equity trade -> two ledger rows (cash movement + fees)."""
    resolved = _coerce_date(trade_date)
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    share_count = int(shares)
    if share_count <= 0:
        raise ValueError("shares must be positive")
    unit_price = float(fill_price)
    if unit_price <= 0.0:
        raise ValueError("fill_price must be positive")
    notional = share_count * unit_price
    settlement_date = resolve_settlement_date(resolved, asset_class="equity", calendar=calendar)
    fees = estimate_equity_transaction_fees(
        shares=share_count, commission_per_order=CURRENT_EQUITY_OPEN_COMMISSION_PER_ORDER,
        clearing_per_share=CURRENT_EQUITY_CLEARING_PER_SHARE, sell_taf_per_share=None,
        sell_taf_cap_per_order=None, trade_date=resolved, price_per_share=unit_price,
        is_sale=(normalized_side == "sell"),
    )
    cash_effect = notional if normalized_side == "sell" else -notional
    return (
        EconomicsLedgerEvent(
            event_type=f"equity_{normalized_side}", symbol=symbol, sub_sleeve=sub_sleeve,
            effective_date=resolved, settlement_date=settlement_date, cash_effect_usd=cash_effect,
            position_group_id=position_group_id,
            description=f"{normalized_side} {share_count} shares @ {unit_price:.4f}",
            diagnostics={"shares": share_count, "fill_price": unit_price},
        ),
        EconomicsLedgerEvent(
            event_type="equity_trade_fees", symbol=symbol, sub_sleeve=sub_sleeve,
            effective_date=resolved, settlement_date=settlement_date, cash_effect_usd=-fees,
            position_group_id=position_group_id, description=f"equity {normalized_side} fees",
            diagnostics={"shares": share_count, "fees_usd": fees},
        ),
    )


def build_option_trade_event(
    *, trade_date: date | datetime | str, symbol: str, sub_sleeve: str, contracts: int,
    premium_per_contract: float, side: Literal["buy", "sell"], opening_trade: bool = True,
    position_group_id: str | None = None, calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, EconomicsLedgerEvent]:
    """One option trade -> two ledger rows (premium movement + fees)."""
    resolved = _coerce_date(trade_date)
    normalized_side = str(side).strip().lower()
    if normalized_side not in {"buy", "sell"}:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    contract_count = int(contracts)
    if contract_count <= 0:
        raise ValueError("contracts must be positive")
    premium = float(premium_per_contract)
    if premium <= 0.0:
        raise ValueError("premium_per_contract must be positive")
    premium_cash = contract_count * 100.0 * premium
    settlement_date = resolve_settlement_date(resolved, asset_class="equity_option", calendar=calendar)
    fees = estimate_option_transaction_fees(
        contracts=contract_count,
        commission_per_contract=(
            CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT if opening_trade
            else CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT
        ),
        commission_cap_per_leg=CURRENT_OPTION_COMMISSION_CAP_PER_LEG,
        clearing_per_contract=CURRENT_OPTION_CLEARING_PER_CONTRACT,
        orf_per_contract=CURRENT_OPTION_ORF_PER_CONTRACT, taf_per_contract=None,
        trade_date=resolved, is_sale=(normalized_side == "sell"),
    )
    cash_effect = premium_cash if normalized_side == "sell" else -premium_cash
    return (
        EconomicsLedgerEvent(
            event_type=f"option_premium_{'in' if normalized_side == 'sell' else 'out'}",
            symbol=symbol, sub_sleeve=sub_sleeve, effective_date=resolved,
            settlement_date=settlement_date, cash_effect_usd=cash_effect,
            position_group_id=position_group_id,
            description=f"{normalized_side} {contract_count} contracts @ {premium:.4f}",
            diagnostics={"contracts": contract_count, "premium_per_contract": premium,
                         "opening_trade": bool(opening_trade)},
        ),
        EconomicsLedgerEvent(
            event_type="option_trade_fees", symbol=symbol, sub_sleeve=sub_sleeve,
            effective_date=resolved, settlement_date=settlement_date, cash_effect_usd=-fees,
            position_group_id=position_group_id, description=f"option {normalized_side} fees",
            diagnostics={"contracts": contract_count, "fees_usd": fees, "opening_trade": bool(opening_trade)},
        ),
    )


def build_buy_write_open_events(
    *, trade_date, symbol, stock_fill_price, short_call_fill_price, shares=100, contracts=1,
    position_group_id=None, calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, ...]:
    """Covered-call open: buy stock + sell call."""
    stock = build_equity_trade_event(
        trade_date=trade_date, symbol=symbol, sub_sleeve="covered_call", shares=shares,
        fill_price=stock_fill_price, side="buy", position_group_id=position_group_id, calendar=calendar)
    call = build_option_trade_event(
        trade_date=trade_date, symbol=symbol, sub_sleeve="covered_call", contracts=contracts,
        premium_per_contract=short_call_fill_price, side="sell",
        position_group_id=position_group_id, calendar=calendar)
    return stock + call


def build_bull_put_spread_open_events(
    *, trade_date, symbol, short_put_fill_price, long_put_fill_price, contracts,
    position_group_id=None, calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, ...]:
    """Bull put spread open: sell short put + buy long put."""
    short = build_option_trade_event(
        trade_date=trade_date, symbol=symbol, sub_sleeve="bull_put_spread", contracts=contracts,
        premium_per_contract=short_put_fill_price, side="sell",
        position_group_id=position_group_id, calendar=calendar)
    long = build_option_trade_event(
        trade_date=trade_date, symbol=symbol, sub_sleeve="bull_put_spread", contracts=contracts,
        premium_per_contract=long_put_fill_price, side="buy",
        position_group_id=position_group_id, calendar=calendar)
    return short + long


def build_tail_hedge_open_events(
    *, trade_date, symbol, premium_per_contract, contracts, position_group_id=None,
    calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, ...]:
    """Tail hedge open: buy protective options."""
    return build_option_trade_event(
        trade_date=trade_date, symbol=symbol, sub_sleeve="tail_hedge", contracts=contracts,
        premium_per_contract=premium_per_contract, side="buy",
        position_group_id=position_group_id, calendar=calendar)


def _build_equity_assignment_or_exercise_events(
    *, event_date, symbol, sub_sleeve, fill_price, shares, side, fee_description,
    fee_diagnostics, position_group_id=None, config: ReplayEconomicsConfig | None = None,
    calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, ...]:
    resolved = _coerce_date(event_date)
    base_config = config or ReplayEconomicsConfig()
    sale_event, fee_event = build_equity_trade_event(
        trade_date=resolved, symbol=symbol, sub_sleeve=sub_sleeve, shares=shares,
        fill_price=fill_price, side=side, position_group_id=position_group_id, calendar=calendar)
    assignment_fee_event = EconomicsLedgerEvent(
        event_type="assignment_fee", symbol=symbol, sub_sleeve=sub_sleeve, effective_date=resolved,
        settlement_date=sale_event.settlement_date, cash_effect_usd=-base_config.assignment_fee_usd,
        position_group_id=position_group_id, description=fee_description,
        diagnostics={"assignment_fee_usd": base_config.assignment_fee_usd, **dict(fee_diagnostics)},
    )
    return sale_event, fee_event, assignment_fee_event


def build_covered_call_assignment_events(
    *, assignment_date, symbol, strike_price, shares=100, position_group_id=None,
    config: ReplayEconomicsConfig | None = None, calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, ...]:
    """Short call assigned: sell stock at strike + assignment fee."""
    return _build_equity_assignment_or_exercise_events(
        event_date=assignment_date, symbol=symbol, sub_sleeve="covered_call", fill_price=strike_price,
        shares=shares, side="sell", fee_description="short call assignment fee",
        fee_diagnostics={"assignment_fee_usd": (config or ReplayEconomicsConfig()).assignment_fee_usd},
        position_group_id=position_group_id, config=config, calendar=calendar)


def build_short_put_assignment_events(
    *, assignment_date, symbol, strike_price, shares=100, position_group_id=None,
    config: ReplayEconomicsConfig | None = None, calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, ...]:
    """Short put assigned: buy stock at strike + assignment fee."""
    return _build_equity_assignment_or_exercise_events(
        event_date=assignment_date, symbol=symbol, sub_sleeve="bull_put_assignment", fill_price=strike_price,
        shares=shares, side="buy", fee_description="short put assignment fee",
        fee_diagnostics={"assignment_fee_usd": (config or ReplayEconomicsConfig()).assignment_fee_usd},
        position_group_id=position_group_id, config=config, calendar=calendar)


def build_long_put_exercise_events(
    *, exercise_date, symbol, strike_price, shares=100, position_group_id=None,
    config: ReplayEconomicsConfig | None = None, calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> tuple[EconomicsLedgerEvent, ...]:
    """Long put exercised: sell stock at strike + fee."""
    return _build_equity_assignment_or_exercise_events(
        event_date=exercise_date, symbol=symbol, sub_sleeve="bull_put_assignment", fill_price=strike_price,
        shares=shares, side="sell", fee_description="long put exercise fee",
        fee_diagnostics={"assignment_fee_usd": (config or ReplayEconomicsConfig()).assignment_fee_usd},
        position_group_id=position_group_id, config=config, calendar=calendar)


def build_dividend_ledger_events(
    *, symbol, shares, dividend_per_share, ex_div_date, pay_date=None, position_group_id=None,
) -> tuple[EconomicsLedgerEvent, ...]:
    """Dividend: ex-date receivable accrual + pay-date cash receipt (or ex-date
    fallback when no pay date is known)."""
    ex_date = _coerce_date(ex_div_date)
    share_count = int(shares)
    if share_count < 0:
        raise ValueError("shares cannot be negative")
    dividend = float(dividend_per_share)
    if dividend < 0.0:
        raise ValueError("dividend_per_share cannot be negative")
    amount = share_count * dividend
    if amount <= 0:
        return ()
    if pay_date is None:
        return (EconomicsLedgerEvent(
            event_type="dividend_cash_ex_date_fallback", symbol=symbol, sub_sleeve="covered_call",
            effective_date=ex_date, settlement_date=ex_date, cash_effect_usd=amount,
            position_group_id=position_group_id,
            description="dividend booked on ex-date because pay date is unavailable",
            diagnostics={"shares": share_count, "dividend_per_share": dividend}),)
    pay = _coerce_date(pay_date)
    if pay < ex_date:
        raise ValueError("pay_date cannot be earlier than ex_div_date")
    return (
        EconomicsLedgerEvent(
            event_type="dividend_receivable_accrual", symbol=symbol, sub_sleeve="covered_call",
            effective_date=ex_date, settlement_date=ex_date, cash_effect_usd=0.0,
            receivable_effect_usd=amount, position_group_id=position_group_id,
            description="dividend entitlement recorded on ex-date",
            diagnostics={"shares": share_count, "dividend_per_share": dividend}),
        EconomicsLedgerEvent(
            event_type="dividend_cash_receipt", symbol=symbol, sub_sleeve="covered_call",
            effective_date=pay, settlement_date=pay, cash_effect_usd=amount,
            receivable_effect_usd=-amount, position_group_id=position_group_id,
            description="dividend cash received on pay date",
            diagnostics={"shares": share_count, "dividend_per_share": dividend}),
    )


def build_margin_interest_event(
    *, session_date, settled_cash_usd, config: ReplayEconomicsConfig | None = None,
    calendar: SessionCalendar = DEFAULT_CALENDAR,
) -> EconomicsLedgerEvent | None:
    """Daily margin interest on any settled-cash debit, posted next session."""
    resolved = _coerce_date(session_date)
    base_config = config or ReplayEconomicsConfig()
    debit_balance = max(-float(settled_cash_usd), 0.0)
    if debit_balance <= 0:
        return None
    annual_rate = resolve_margin_interest_rate(debit_balance, margin_rate_tiers=base_config.margin_rate_tiers)
    interest = estimate_daily_margin_interest(
        debit_balance, annual_rate=annual_rate, day_basis=base_config.margin_interest_day_basis)
    posting_date = resolve_next_session_date(resolved, calendar=calendar)
    return EconomicsLedgerEvent(
        event_type="margin_interest", symbol="PORTFOLIO", sub_sleeve="portfolio_financing",
        effective_date=posting_date, settlement_date=posting_date, cash_effect_usd=-interest,
        description="daily margin interest accrual",
        diagnostics={"accrual_session_date": resolved.isoformat(), "debit_balance_usd": debit_balance,
                     "annual_rate": annual_rate, "day_basis": base_config.margin_interest_day_basis,
                     "compounds_daily": base_config.margin_interest_compounds_daily},
    )


# --- Snapshots ----------------------------------------------------------------

@dataclass(frozen=True)
class ReplayEconomicsSnapshot:
    session_date: date
    settled_cash_usd: float
    unsettled_cash_usd: float
    receivable_balance_usd: float
    margin_debit_usd: float
    annual_margin_rate: float
    daily_margin_interest_estimate_usd: float
    long_equity_market_value_usd: float
    option_market_value_usd: float
    defined_risk_requirement_usd: float
    long_option_requirement_usd: float
    estimated_initial_requirement_usd: float
    net_liquidation_value_usd: float
    estimated_excess_liquidity_usd: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["session_date"] = self.session_date.isoformat()
        payload["diagnostics"] = repr(self.diagnostics)
        return payload


@dataclass(frozen=True)
class PositionGroupEconomicsSnapshot:
    """Group-scoped cash/receivable attribution from the immutable ledger, the
    correct surface when per-leg realized P&L is not a valid group rollup (e.g.
    assignment paths where the opening credit stays in the cash ledger)."""

    position_group_id: str
    session_date: date
    settled_cash_usd: float
    unsettled_cash_usd: float
    receivable_balance_usd: float
    net_economic_value_usd: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["session_date"] = self.session_date.isoformat()
        payload["diagnostics"] = repr(self.diagnostics)
        return payload


def build_replay_economics_snapshot(
    *, session_date, opening_settled_cash_usd, ledger_events: Sequence[EconomicsLedgerEvent],
    long_equity_market_value_usd, option_market_value_usd, defined_risk_requirement_usd,
    config: ReplayEconomicsConfig | None = None,
) -> ReplayEconomicsSnapshot:
    """Partition ledger events into settled/unsettled/receivable as of the session
    date and compute margin/NLV/excess-liquidity."""
    resolved = _coerce_date(session_date)
    base_config = config or ReplayEconomicsConfig()
    settled_cash = float(opening_settled_cash_usd)
    unsettled_cash = 0.0
    receivable_balance = 0.0
    for event in ledger_events:
        if event.settlement_date <= resolved:
            settled_cash += event.cash_effect_usd
        elif event.effective_date <= resolved < event.settlement_date:
            unsettled_cash += event.cash_effect_usd
        if event.effective_date <= resolved:
            receivable_balance += event.receivable_effect_usd

    margin_debit = max(-settled_cash, 0.0)
    annual_margin_rate = resolve_margin_interest_rate(margin_debit, margin_rate_tiers=base_config.margin_rate_tiers)
    daily_margin_interest_estimate = estimate_daily_margin_interest(
        margin_debit, annual_rate=annual_margin_rate, day_basis=base_config.margin_interest_day_basis)
    long_option_requirement = max(float(option_market_value_usd), 0.0) * base_config.long_option_requirement_pct
    initial_requirement = (
        max(float(long_equity_market_value_usd), 0.0) * base_config.long_equity_initial_requirement_pct
        + max(float(defined_risk_requirement_usd), 0.0) * base_config.defined_risk_requirement_pct
        + long_option_requirement
    )
    net_liquidation_value = (
        settled_cash + unsettled_cash + receivable_balance
        + float(long_equity_market_value_usd) + float(option_market_value_usd)
    )
    return ReplayEconomicsSnapshot(
        session_date=resolved, settled_cash_usd=settled_cash, unsettled_cash_usd=unsettled_cash,
        receivable_balance_usd=receivable_balance, margin_debit_usd=margin_debit,
        annual_margin_rate=annual_margin_rate,
        daily_margin_interest_estimate_usd=daily_margin_interest_estimate,
        long_equity_market_value_usd=float(long_equity_market_value_usd),
        option_market_value_usd=float(option_market_value_usd),
        defined_risk_requirement_usd=float(defined_risk_requirement_usd),
        long_option_requirement_usd=long_option_requirement,
        estimated_initial_requirement_usd=initial_requirement,
        net_liquidation_value_usd=net_liquidation_value,
        estimated_excess_liquidity_usd=net_liquidation_value - initial_requirement,
        diagnostics={"funding_policy": base_config.funding_policy,
                     "margin_interest_day_basis": base_config.margin_interest_day_basis,
                     "margin_interest_compounds_daily": base_config.margin_interest_compounds_daily},
    )


def build_position_group_economics_snapshot(
    *, position_group_id: str, session_date, ledger_events: Sequence[EconomicsLedgerEvent],
) -> PositionGroupEconomicsSnapshot:
    """Group-scoped settled/unsettled/receivable snapshot (no market marks)."""
    resolved = _coerce_date(session_date)
    group_id = str(position_group_id).strip()
    if not group_id:
        raise ValueError("position_group_id must be non-empty")
    settled_cash = unsettled_cash = receivable_balance = 0.0
    matching_event_count = 0
    for event in ledger_events:
        if event.position_group_id != group_id:
            continue
        matching_event_count += 1
        if event.settlement_date <= resolved:
            settled_cash += event.cash_effect_usd
        elif event.effective_date <= resolved < event.settlement_date:
            unsettled_cash += event.cash_effect_usd
        if event.effective_date <= resolved:
            receivable_balance += event.receivable_effect_usd
    return PositionGroupEconomicsSnapshot(
        position_group_id=group_id, session_date=resolved, settled_cash_usd=settled_cash,
        unsettled_cash_usd=unsettled_cash, receivable_balance_usd=receivable_balance,
        net_economic_value_usd=settled_cash + unsettled_cash + receivable_balance,
        diagnostics={"matching_event_count": matching_event_count},
    )

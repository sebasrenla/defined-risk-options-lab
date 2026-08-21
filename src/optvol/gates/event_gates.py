"""Event gates: block new entries around earnings, corporate actions, and macro
events.

The income sleeves (covered calls, bull put spreads) are premium-selling
strategies whose worst outcomes cluster around *known scheduled events*, an
earnings release can gap the underlying straight through a short strike, a
corporate action (split, special dividend, M&A) can distort the option chain,
and a market-wide macro print (FOMC, CPI, NFP) injects portfolio-level jump
risk. Rather than price those tails, the program simply refuses to *open* new
risk into them.

``evaluate_program1_entry_gates`` returns a structured decision with granular
reason codes and diagnostics, so a blocked entry is fully explainable and
auditable. Missing/invalid context fields can themselves block an entry
(``strict_context_fields``), a fail-closed posture: if we cannot confirm the
event context, we do not open.

Provenance
----------
Ported from the program's ``event_gates.py`` (validators import path updated).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping

from ..backtest.validators import normalize_corporate_action_flag


REJECT_EARNINGS_BLACKOUT = "earnings_blackout"
REJECT_CORPORATE_ACTION_ACTIVE = "corporate_action_active"
REJECT_MACRO_EVENT_SESSION_DAY = "macro_event_session_day"
REJECT_MISSING_DAYS_TO_EARNINGS = "missing_days_to_earnings"
REJECT_INVALID_DAYS_TO_EARNINGS = "invalid_days_to_earnings"
REJECT_MISSING_CORPORATE_ACTION_FLAG = "missing_corporate_action_flag"
REJECT_INVALID_CORPORATE_ACTION_FLAG = "invalid_corporate_action_flag"
REJECT_MISSING_MACRO_EVENT_CALENDAR = "missing_macro_event_calendar"

DEFAULT_BLOCKED_MACRO_EVENTS = ("fomc", "cpi", "nfp")


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _normalize_symbol(value: str | None) -> str:
    return (value or "").upper().strip()


def _coerce_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Unsupported date value: {value!r}")
    text = value.strip()
    if not text:
        raise ValueError("Blank date value")
    try:
        return date.fromisoformat(text)
    except ValueError:
        normalized = text.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).date()


def _parse_days_to_earnings(value: Any) -> tuple[int | None, str | None]:
    if _is_blank(value):
        return None, REJECT_MISSING_DAYS_TO_EARNINGS
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None, REJECT_INVALID_DAYS_TO_EARNINGS
    if not parsed.is_integer():
        return None, REJECT_INVALID_DAYS_TO_EARNINGS
    return int(parsed), None


def _parse_corporate_action_flag(value: Any) -> tuple[str | None, str | None]:
    if _is_blank(value):
        return None, REJECT_MISSING_CORPORATE_ACTION_FLAG
    normalized = normalize_corporate_action_flag(str(value))
    if normalized is None:
        return None, REJECT_INVALID_CORPORATE_ACTION_FLAG
    return normalized, None


def _normalize_macro_event_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _iter_macro_event_names(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        items = [raw_value]
    elif isinstance(raw_value, Iterable):
        items = list(raw_value)
    else:
        items = [str(raw_value)]
    normalized: list[str] = []
    for item in items:
        if _is_blank(item):
            continue
        normalized.append(_normalize_macro_event_name(str(item)))
    return normalized


def _dedupe_in_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


@dataclass(frozen=True)
class Program1EventGateConfig:
    earnings_blackout_days_before: int = 5
    earnings_blackout_days_after: int = 1
    blocked_macro_events: tuple[str, ...] = DEFAULT_BLOCKED_MACRO_EVENTS
    strict_context_fields: bool = True
    require_macro_event_calendar: bool = False

    def __post_init__(self) -> None:
        if self.earnings_blackout_days_before < 0 or self.earnings_blackout_days_after < 0:
            raise ValueError("Earnings blackout configuration must be non-negative")
        normalized = _dedupe_in_order(
            _normalize_macro_event_name(name) for name in self.blocked_macro_events if name
        )
        object.__setattr__(self, "blocked_macro_events", normalized)


@dataclass
class Program1GateDecision:
    symbol: str
    as_of_date: date
    allow_new_entries: bool
    reason_codes: tuple[str, ...] = ()
    blocked_macro_events: tuple[str, ...] = ()
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return not self.allow_new_entries

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["as_of_date"] = self.as_of_date.isoformat()
        return payload


DEFAULT_PROGRAM1_EVENT_GATE_CONFIG = Program1EventGateConfig()


def is_earnings_blackout(
    days_to_earnings: int,
    *,
    days_before: int = DEFAULT_PROGRAM1_EVENT_GATE_CONFIG.earnings_blackout_days_before,
    days_after: int = DEFAULT_PROGRAM1_EVENT_GATE_CONFIG.earnings_blackout_days_after,
) -> bool:
    """True when ``days_to_earnings`` falls inside the blackout window.

    ``days_to_earnings`` is signed: positive before earnings, negative after. The
    window is ``[-days_after, +days_before]``.
    """
    if days_before < 0 or days_after < 0:
        raise ValueError("Earnings blackout window must be non-negative")
    return (-days_after) <= days_to_earnings <= days_before


def get_macro_events_for_session(
    as_of_date: date | datetime | str,
    macro_event_calendar: Mapping[date | str, Any] | None,
) -> tuple[str, ...]:
    if macro_event_calendar is None:
        return ()
    target_date = _coerce_date(as_of_date)
    collected: list[str] = []
    if target_date in macro_event_calendar:
        collected.extend(_iter_macro_event_names(macro_event_calendar[target_date]))
    if target_date.isoformat() in macro_event_calendar:
        collected.extend(_iter_macro_event_names(macro_event_calendar[target_date.isoformat()]))
    if not collected:
        for key, value in macro_event_calendar.items():
            try:
                if _coerce_date(key) == target_date:
                    collected.extend(_iter_macro_event_names(value))
            except (TypeError, ValueError):
                continue
    return _dedupe_in_order(collected)


def get_blocked_macro_events_for_session(
    as_of_date: date | datetime | str,
    macro_event_calendar: Mapping[date | str, Any] | None,
    *,
    blocked_macro_events: Iterable[str] = DEFAULT_BLOCKED_MACRO_EVENTS,
) -> tuple[str, ...]:
    session_events = get_macro_events_for_session(as_of_date, macro_event_calendar)
    blocked = set(_normalize_macro_event_name(name) for name in blocked_macro_events)
    return tuple(event for event in session_events if event in blocked)


def evaluate_program1_entry_gates(
    *,
    as_of_date: date | datetime | str,
    symbol: str,
    days_to_earnings: Any,
    corporate_action_flag: Any,
    macro_event_calendar: Mapping[date | str, Any] | None = None,
    config: Program1EventGateConfig = DEFAULT_PROGRAM1_EVENT_GATE_CONFIG,
) -> Program1GateDecision:
    """Evaluate all entry gates and return a structured, explainable decision."""
    session_date = _coerce_date(as_of_date)
    symbol_code = _normalize_symbol(symbol)
    reason_codes: list[str] = []
    diagnostics: dict[str, Any] = {
        "days_to_earnings_raw": days_to_earnings,
        "corporate_action_flag_raw": corporate_action_flag,
    }

    parsed_days_to_earnings, days_error = _parse_days_to_earnings(days_to_earnings)
    if parsed_days_to_earnings is not None:
        diagnostics["days_to_earnings"] = parsed_days_to_earnings
        if is_earnings_blackout(
            parsed_days_to_earnings,
            days_before=config.earnings_blackout_days_before,
            days_after=config.earnings_blackout_days_after,
        ):
            reason_codes.append(REJECT_EARNINGS_BLACKOUT)
    elif config.strict_context_fields and days_error is not None:
        reason_codes.append(days_error)

    normalized_flag, flag_error = _parse_corporate_action_flag(corporate_action_flag)
    if normalized_flag is not None:
        diagnostics["corporate_action_flag"] = normalized_flag
        if normalized_flag == "1":
            reason_codes.append(REJECT_CORPORATE_ACTION_ACTIVE)
    elif config.strict_context_fields and flag_error is not None:
        reason_codes.append(flag_error)

    blocked_macro_events: tuple[str, ...] = ()
    if macro_event_calendar is None:
        if config.require_macro_event_calendar:
            reason_codes.append(REJECT_MISSING_MACRO_EVENT_CALENDAR)
    else:
        blocked_macro_events = get_blocked_macro_events_for_session(
            session_date, macro_event_calendar, blocked_macro_events=config.blocked_macro_events
        )
        if blocked_macro_events:
            reason_codes.append(REJECT_MACRO_EVENT_SESSION_DAY)

    diagnostics["blocked_macro_events"] = list(blocked_macro_events)

    deduped_reasons = _dedupe_in_order(reason_codes)
    return Program1GateDecision(
        symbol=symbol_code,
        as_of_date=session_date,
        allow_new_entries=not deduped_reasons,
        reason_codes=deduped_reasons,
        blocked_macro_events=blocked_macro_events,
        diagnostics=diagnostics,
    )


def evaluate_program1_entry_from_context_row(
    *,
    as_of_date: date | datetime | str,
    context_row: Mapping[str, Any],
    macro_event_calendar: Mapping[date | str, Any] | None = None,
    config: Program1EventGateConfig = DEFAULT_PROGRAM1_EVENT_GATE_CONFIG,
) -> Program1GateDecision:
    return evaluate_program1_entry_gates(
        as_of_date=as_of_date,
        symbol=str(context_row.get("symbol") or ""),
        days_to_earnings=context_row.get("days_to_earnings"),
        corporate_action_flag=context_row.get("corporate_action_flag"),
        macro_event_calendar=macro_event_calendar,
        config=config,
    )

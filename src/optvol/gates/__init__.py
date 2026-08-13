"""Event-driven entry gates."""

from .event_gates import (
    DEFAULT_BLOCKED_MACRO_EVENTS,
    DEFAULT_PROGRAM1_EVENT_GATE_CONFIG,
    Program1EventGateConfig,
    Program1GateDecision,
    evaluate_program1_entry_from_context_row,
    evaluate_program1_entry_gates,
    get_blocked_macro_events_for_session,
    get_macro_events_for_session,
    is_earnings_blackout,
)

__all__ = [
    "DEFAULT_BLOCKED_MACRO_EVENTS",
    "DEFAULT_PROGRAM1_EVENT_GATE_CONFIG",
    "Program1EventGateConfig",
    "Program1GateDecision",
    "is_earnings_blackout",
    "get_macro_events_for_session",
    "get_blocked_macro_events_for_session",
    "evaluate_program1_entry_gates",
    "evaluate_program1_entry_from_context_row",
]

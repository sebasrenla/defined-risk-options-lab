"""Event gates: earnings, corporate-action, and macro blackouts."""

from datetime import date

from optvol.gates import (
    Program1EventGateConfig,
    evaluate_program1_entry_gates,
    is_earnings_blackout,
)
from optvol.gates.event_gates import (
    REJECT_CORPORATE_ACTION_ACTIVE,
    REJECT_EARNINGS_BLACKOUT,
    REJECT_INVALID_DAYS_TO_EARNINGS,
    REJECT_MACRO_EVENT_SESSION_DAY,
)


def test_earnings_blackout_window():
    # Default window: 5 days before through 1 day after.
    assert is_earnings_blackout(5) is True
    assert is_earnings_blackout(0) is True
    assert is_earnings_blackout(-1) is True
    assert is_earnings_blackout(6) is False
    assert is_earnings_blackout(-2) is False


def test_clean_entry_allowed():
    d = evaluate_program1_entry_gates(
        as_of_date="2026-06-01", symbol="AAA", days_to_earnings=20,
        corporate_action_flag="0")
    assert d.allow_new_entries is True
    assert d.reason_codes == ()


def test_earnings_blackout_blocks_entry():
    d = evaluate_program1_entry_gates(
        as_of_date="2026-06-01", symbol="AAA", days_to_earnings=3, corporate_action_flag="0")
    assert d.blocked is True
    assert REJECT_EARNINGS_BLACKOUT in d.reason_codes


def test_corporate_action_blocks_entry():
    d = evaluate_program1_entry_gates(
        as_of_date="2026-06-01", symbol="AAA", days_to_earnings=30, corporate_action_flag="1")
    assert REJECT_CORPORATE_ACTION_ACTIVE in d.reason_codes


def test_macro_event_blocks_entry():
    cal = {date(2026, 6, 1): ["FOMC"]}
    d = evaluate_program1_entry_gates(
        as_of_date="2026-06-01", symbol="AAA", days_to_earnings=30,
        corporate_action_flag="0", macro_event_calendar=cal)
    assert REJECT_MACRO_EVENT_SESSION_DAY in d.reason_codes
    assert "fomc" in d.blocked_macro_events


def test_strict_context_rejects_invalid_field():
    d = evaluate_program1_entry_gates(
        as_of_date="2026-06-01", symbol="AAA", days_to_earnings="not-a-number",
        corporate_action_flag="0")
    assert REJECT_INVALID_DAYS_TO_EARNINGS in d.reason_codes


def test_non_strict_config_tolerates_missing_field():
    cfg = Program1EventGateConfig(strict_context_fields=False)
    d = evaluate_program1_entry_gates(
        as_of_date="2026-06-01", symbol="AAA", days_to_earnings="", corporate_action_flag="0",
        config=cfg)
    assert d.allow_new_entries is True

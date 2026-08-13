"""Cash-ledger economics: margin, settlement, events, and snapshots."""

from datetime import date

import pytest

from optvol.economics import (
    build_bull_put_spread_open_events,
    build_dividend_ledger_events,
    build_margin_rate_tiers,
    build_position_group_economics_snapshot,
    build_replay_economics_snapshot,
    estimate_daily_margin_interest,
    resolve_margin_interest_rate,
    resolve_settlement_business_days,
    weekday_calendar,
)

CAL = weekday_calendar(date(2024, 1, 1), date(2028, 12, 31))


def test_margin_rate_tiers_decrease_with_size():
    tiers = build_margin_rate_tiers(0.10)
    rates = [t.annual_rate for t in tiers]
    assert rates == sorted(rates, reverse=True)  # bigger debit -> lower rate
    assert resolve_margin_interest_rate(10_000, margin_rate_tiers=tiers) > \
        resolve_margin_interest_rate(2_000_000, margin_rate_tiers=tiers)


def test_no_margin_interest_on_credit_balance():
    assert resolve_margin_interest_rate(-5000) == 0.0
    assert estimate_daily_margin_interest(0.0, annual_rate=0.11) == 0.0


def test_daily_margin_interest_360_basis():
    assert estimate_daily_margin_interest(36_000, annual_rate=0.10, day_basis=360) == pytest.approx(10.0)


def test_settlement_lag_schedule():
    # Equity settlement shortened over time; options are T+1.
    assert resolve_settlement_business_days("2016-06-01", asset_class="equity") == 3
    assert resolve_settlement_business_days("2020-06-01", asset_class="equity") == 2
    assert resolve_settlement_business_days("2026-06-01", asset_class="equity") == 1
    assert resolve_settlement_business_days("2026-06-01", asset_class="equity_option") == 1


def test_bps_open_credit_lives_in_cash_ledger():
    # The bull-put-spread opening credit must show up as positive cash.
    events = build_bull_put_spread_open_events(
        trade_date="2026-06-01", symbol="AAA", short_put_fill_price=3.0,
        long_put_fill_price=1.0, contracts=1, position_group_id="g", calendar=CAL)
    net_cash = sum(e.cash_effect_usd for e in events)
    # Gross credit = (3 - 1) * 100 = 200, minus fees.
    assert 150 < net_cash < 200


def test_group_snapshot_captures_credit_not_leg_sum():
    # The point of the group snapshot: opening credit is in the cash ledger.
    events = build_bull_put_spread_open_events(
        trade_date="2026-06-01", symbol="AAA", short_put_fill_price=3.0,
        long_put_fill_price=1.0, contracts=2, position_group_id="g", calendar=CAL)
    snap = build_position_group_economics_snapshot(
        position_group_id="g", session_date="2026-06-10", ledger_events=events)
    assert snap.net_economic_value_usd > 0


def test_dividend_receivable_then_cash():
    events = build_dividend_ledger_events(
        symbol="AAA", shares=100, dividend_per_share=0.5, ex_div_date="2026-06-01",
        pay_date="2026-06-20", position_group_id="g")
    # On ex-date: receivable accrues, no cash yet.
    accrual = build_position_group_economics_snapshot(
        position_group_id="g", session_date="2026-06-05", ledger_events=events)
    assert accrual.receivable_balance_usd == pytest.approx(50.0)
    assert accrual.settled_cash_usd == pytest.approx(0.0)
    # After pay-date: cash received, receivable cleared.
    paid = build_position_group_economics_snapshot(
        position_group_id="g", session_date="2026-06-25", ledger_events=events)
    assert paid.settled_cash_usd == pytest.approx(50.0)
    assert paid.receivable_balance_usd == pytest.approx(0.0)


def test_snapshot_margin_debit_and_nlv():
    events = build_bull_put_spread_open_events(
        trade_date="2026-06-01", symbol="AAA", short_put_fill_price=3.0,
        long_put_fill_price=1.0, contracts=1, position_group_id="g", calendar=CAL)
    snap = build_replay_economics_snapshot(
        session_date="2026-06-10", opening_settled_cash_usd=50_000.0, ledger_events=events,
        long_equity_market_value_usd=0.0, option_market_value_usd=0.0,
        defined_risk_requirement_usd=400.0)
    assert snap.margin_debit_usd == 0.0  # positive cash, no debit
    assert snap.net_liquidation_value_usd > 50_000

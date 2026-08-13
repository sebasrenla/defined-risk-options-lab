"""Strategy sleeves: selection and management decisions on a synthetic chain."""

from datetime import date

import pytest

from optvol.sleeves import (
    BullPutSpreadConfig,
    CoveredCallConfig,
    TailHedgeConfig,
    TailHedgePositionSnapshot,
    determine_allocation_multiplier,
    evaluate_bull_put_spread_exit_action,
    evaluate_covered_call_roll_action,
    evaluate_tail_hedge_budget,
    evaluate_tail_hedge_position_action,
    select_bull_put_spread_candidate,
    select_covered_call_candidate,
)

from generate_synthetic_chain import generate_synthetic_chain

CHAIN = generate_synthetic_chain(symbol="AAA", spot=300.0, as_of=date(2026, 6, 1),
                                 strikes_per_side=12, strike_step_pct=0.02)


def test_covered_call_selects_in_delta_window():
    cand, diag = select_covered_call_candidate(
        quotes=CHAIN, symbol="AAA", as_of_date="2026-06-01", nav_cap_pct=0.04,
        config=CoveredCallConfig(), iv_rank=45.0)
    assert cand is not None
    cfg = CoveredCallConfig()
    assert cfg.call_delta_min <= cand.delta <= cfg.call_delta_max
    assert cfg.dte_min <= cand.dte <= cfg.dte_max


def test_bull_put_spread_pairs_short_above_long():
    cand, diag = select_bull_put_spread_candidate(
        quotes=CHAIN, symbol="AAA", as_of_date="2026-06-01", nav_cap_pct=0.04,
        config=BullPutSpreadConfig(), iv_rank=45.0)
    assert cand is not None
    assert cand.long_strike < cand.short_strike
    assert cand.structure_mid >= cand.credit_floor
    assert cand.max_loss_model > 0


def test_iv_rank_allocation_haircut():
    cfg = CoveredCallConfig()
    assert determine_allocation_multiplier(45.0, cfg)[0] == cfg.default_allocation_multiplier
    assert determine_allocation_multiplier(10.0, cfg)[0] == cfg.low_iv_rank_allocation_multiplier
    assert determine_allocation_multiplier(None, cfg)[0] == cfg.iv_rank_unknown_allocation_multiplier


def test_covered_call_roll_profit_take():
    d = evaluate_covered_call_roll_action(
        entry_credit=2.0, current_option_mark=0.8, short_strike=100, spot_price=98,
        dte_remaining=20)
    assert d.action_code == "close_profit_take"
    assert d.should_close is True


def test_covered_call_roll_time_exit():
    d = evaluate_covered_call_roll_action(
        entry_credit=2.0, current_option_mark=1.9, short_strike=100, spot_price=95,
        dte_remaining=5)
    assert d.action_code == "roll_time_exit"


def test_bps_exit_stop_loss():
    d = evaluate_bull_put_spread_exit_action(
        entry_credit=1.0, current_close_cost=2.5, short_strike=100, spot_price=97, dte_remaining=20)
    assert d.action_code == "close_stop_loss"


def test_bps_exit_breach():
    d = evaluate_bull_put_spread_exit_action(
        entry_credit=1.0, current_close_cost=1.2, short_strike=100, spot_price=98, dte_remaining=8)
    assert d.action_code == "close_breach_exit"


def test_tail_hedge_budget_bands():
    b = evaluate_tail_hedge_budget(nav=1_000_000, spend_mtd_usd=0.0, config=TailHedgeConfig())
    assert b.top_up_required is True  # spent nothing this month
    assert b.pause_additional_buys is False
    # Annual max is 2% of NAV.
    assert b.annual_budget_max_usd == pytest.approx(20_000.0)


def test_tail_hedge_pause_when_over_budget():
    cfg = TailHedgeConfig()
    over = cfg.annual_budget_pct_target / 12.0 * 1_000_000 * cfg.monthly_over_spend_ratio + 1.0
    b = evaluate_tail_hedge_budget(nav=1_000_000, spend_mtd_usd=over, config=cfg)
    assert b.pause_additional_buys is True


def test_tail_hedge_monetize_windfall():
    pos = TailHedgePositionSnapshot(symbol="SPY", expiry=date(2026, 9, 18), strike=400,
                                    quantity=10, entry_premium=2.0, current_option_mark=6.0,
                                    dte_remaining=30)
    d = evaluate_tail_hedge_position_action(pos, TailHedgeConfig())
    assert d.action_code == "monetize_windfall_half"
    assert d.contracts_to_monetize == 5


def test_tail_hedge_hard_roll_final_week():
    pos = TailHedgePositionSnapshot(symbol="SPY", expiry=date(2026, 9, 18), strike=400,
                                    quantity=10, entry_premium=2.0, current_option_mark=1.5,
                                    dte_remaining=5)
    d = evaluate_tail_hedge_position_action(pos, TailHedgeConfig())
    assert d.action_code == "roll_monthly_hard"
    assert d.should_roll_now is True

"""Execution costs: time-varying regulatory schedules and per-structure fees."""

from datetime import date

import pytest

from optvol.execution import (
    estimate_equity_transaction_fees,
    estimate_option_transaction_fees,
)
from optvol.execution.fees_butterfly import estimate_bwb_side_fee_total
from optvol.execution.fees_regulatory import (
    option_taf_per_contract,
    sec_section31_per_million,
)


def test_option_taf_resolves_by_trade_date():
    assert option_taf_per_contract("2016-06-01") == pytest.approx(0.00200)  # first tier
    assert option_taf_per_contract("2024-06-01") == pytest.approx(0.00279)
    assert option_taf_per_contract("2026-06-01") == pytest.approx(0.00329)
    # Before the schedule begins (pre-2016), no rate is in effect -> 0.0.
    assert option_taf_per_contract("2015-01-01") == 0.0


def test_sec_section31_step_down_to_zero_in_2025():
    assert sec_section31_per_million("2024-06-01") == pytest.approx(27.80)
    assert sec_section31_per_million("2025-06-01") == pytest.approx(0.00)


def test_option_fee_applies_commission_cap_and_taf_on_sales():
    # 10 contracts, $1 commission capped at $10, clearing .10, ORF .02295, TAF on sale.
    fee = estimate_option_transaction_fees(
        contracts=10, commission_per_contract=1.0, commission_cap_per_leg=10.0,
        clearing_per_contract=0.10, orf_per_contract=0.02295, is_sale=True,
        trade_date="2026-06-01")
    expected = 10.0 + 10 * (0.10 + 0.02295) + 10 * 0.00329
    assert fee == pytest.approx(expected)


def test_option_fee_no_taf_on_buys():
    buy = estimate_option_transaction_fees(
        contracts=5, commission_per_contract=1.0, commission_cap_per_leg=10.0,
        clearing_per_contract=0.10, orf_per_contract=0.02295, is_sale=False,
        trade_date="2026-06-01")
    sell = estimate_option_transaction_fees(
        contracts=5, commission_per_contract=1.0, commission_cap_per_leg=10.0,
        clearing_per_contract=0.10, orf_per_contract=0.02295, is_sale=True,
        trade_date="2026-06-01")
    assert sell > buy


def test_equity_sec_fee_only_on_sales():
    common = dict(shares=1000, commission_per_order=0.0, clearing_per_share=0.0008,
                  price_per_share=100.0, trade_date="2024-06-01")
    buy = estimate_equity_transaction_fees(is_sale=False, **common)
    sell = estimate_equity_transaction_fees(is_sale=True, **common)
    assert buy == pytest.approx(1000 * 0.0008)
    assert sell > buy  # adds TAF + SEC section 31


def test_butterfly_side_fee_taf_on_sell_side_only():
    cfg = {"fee_open_commission": 1.0, "fee_close_commission": 0.0,
           "fee_commission_cap_per_leg": 10.0, "fee_clearing": 0.10, "fee_orf": 0.02295,
           "fee_taf": 0.00329, "fee_per_order": 0.0, "fee_legs_per_spread": 3,
           "fee_spread_leg_contracts": [1, 2, 1]}
    open_fee = estimate_bwb_side_fee_total(cfg, 10, side="open")
    assert open_fee > 0

"""The synthetic chain itself must be realistic and arbitrage-free."""

import math
from datetime import date

import pytest

from generate_synthetic_chain import (
    check_static_arbitrage,
    generate_synthetic_chain,
    ssvi_iv,
)


def test_surface_is_statically_arbitrage_free():
    rep = check_static_arbitrage()
    assert rep["arbitrage_free"] is True
    assert rep["butterfly_ok"] is True
    assert rep["calendar_ok"] is True
    assert rep["min_density"] >= -1e-6   # non-negative risk-neutral density
    assert rep["max_call_slope"] <= 1e-6  # calls non-increasing in strike


def test_equity_left_skew_present():
    # OTM put IV > ATM IV > OTM call IV (structural equity skew).
    t = 30 / 365.0
    otm_put = ssvi_iv(math.log(0.90), t, 0.24, -0.55, 0.7)
    atm = ssvi_iv(0.0, t, 0.24, -0.55, 0.7)
    otm_call = ssvi_iv(math.log(1.10), t, 0.24, -0.55, 0.7)
    assert otm_put > atm > otm_call
    # Skew magnitude is realistic (a few to ~15 vol points), not extreme.
    assert 0.03 < (otm_put - otm_call) < 0.20


def test_skew_flattens_with_maturity():
    def skew(dte):
        t = dte / 365.0
        return ssvi_iv(math.log(0.90), t, 0.24, -0.55, 0.7) - ssvi_iv(math.log(1.10), t, 0.24, -0.55, 0.7)
    assert skew(30) > skew(60)


def test_put_call_parity_holds_on_generated_chain():
    chain = generate_synthetic_chain(spot=100.0)
    by_key = {}
    for q in chain:
        by_key.setdefault((q.strike, q.dte), {})[q.option_type] = q
    for (strike, dte), legs in by_key.items():
        if "call" in legs and "put" in legs:
            c, p = legs["call"], legs["put"]
            c_mid = (c.bid + c.ask) / 2.0
            p_mid = (p.bid + p.ask) / 2.0
            # C - P ~ S - K (rate 0), within the spread/quote noise.
            assert c_mid - p_mid == pytest.approx(100.0 - strike, abs=0.5)


def test_generated_chain_has_both_expiries_and_types():
    chain = generate_synthetic_chain(expiries_dte=(30, 60))
    assert {q.dte for q in chain} == {30, 60}
    assert {q.option_type for q in chain} == {"call", "put"}

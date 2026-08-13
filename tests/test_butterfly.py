"""Broken-wing butterfly: payoff geometry, break-evens, and POP vs PHT."""

import pytest

from optvol.structures import BrokenWingButterfly, break_evens, payoff_metrics


def test_call_bwb_metrics():
    # 95/100/110 call, credit 1.0: w1=5, w2=10.
    mp, mr, free = payoff_metrics("call", 1.0, 5.0, 10.0)
    assert mp == pytest.approx(6.0)   # credit + w1
    assert mr == pytest.approx(4.0)   # -(credit + w1 - w2)
    assert free is False


def test_free_risk_flag():
    # Credit structure with no downside (w2 <= w1 + credit path) -> free risk.
    mp, mr, free = payoff_metrics("call", 2.0, 5.0, 3.0)
    assert mr == pytest.approx(0.0)
    assert free is True


def test_break_evens_call_and_put():
    lo, hi = break_evens("call", 95, 100, 110, 1.0)
    assert lo == pytest.approx(94.0)
    assert hi == pytest.approx(106.0)
    lo2, hi2 = break_evens("put", 90, 100, 105, 0.5)
    assert lo2 == pytest.approx(94.5)
    assert hi2 == pytest.approx(105.5)


def test_pop_exceeds_pht_for_wide_credit_structure():
    # The core thesis: terminal probability of profit overstates the path
    # probability of holding to target without breaching a break-even.
    bwb = BrokenWingButterfly("call", 95, 100, 110, net_credit=1.0)
    pop = bwb.probability_of_profit(100, 0.30, 30)
    pht = bwb.hold_probability(100, 0.30, 30, exit_days=10)
    assert pop is not None and pht is not None
    assert pop > pht


def test_ev_uses_path_probability():
    bwb = BrokenWingButterfly("put", 90, 100, 105, net_credit=0.5)
    ev = bwb.expected_value(100, 0.30, 30, exit_days=10, profit_target_pct=0.5, stop_loss_pct=1.0)
    pht = bwb.hold_probability(100, 0.30, 30, exit_days=10)
    mp, mr, _ = bwb.metrics()
    assert ev == pytest.approx(pht * (mp * 0.5) - (1 - pht) * (mr * 1.0))


def test_probabilities_none_on_bad_inputs():
    bwb = BrokenWingButterfly("call", 95, 100, 110, net_credit=1.0)
    assert bwb.probability_of_profit(None, 0.3, 30) is None
    assert bwb.hold_probability(100, 0.0, 30, exit_days=10) is None

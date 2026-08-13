"""Composite scoring: bounded, shaped sub-scores and the weighted blend."""

import pytest

from optvol.scoring import (
    DEFAULT_SCORING_CONFIG,
    composite_score,
    gamma_penalty,
    score_band,
    score_expected_value,
    score_iv_regime,
    score_kink,
    score_time,
)


def test_ev_score_absolute_normalization():
    scale = DEFAULT_SCORING_CONFIG.ev_score_abs_scale
    assert score_expected_value(scale, scale) == pytest.approx(1.0)  # saturates at 1
    assert score_expected_value(scale / 2, scale) == pytest.approx(0.5)
    assert score_expected_value(-1.0, scale) == 0.0  # clamped at 0
    assert score_expected_value(None, scale) == 0.0


def test_iv_regime_buckets():
    assert score_iv_regime(80, None, None) == 1.0
    assert score_iv_regime(55, None, None) == pytest.approx(0.7)
    assert score_iv_regime(35, None, None) == pytest.approx(0.3)
    assert score_iv_regime(10, None, None) == 0.0


def test_band_is_one_inside_target_and_ramps_out():
    assert score_band(3.0, 1.5, 2.0, 4.0, 5.0) == 1.0        # inside [2,4]
    assert score_band(4.5, 1.5, 2.0, 4.0, 5.0) == pytest.approx(0.5)  # halfway to soft max
    assert score_band(6.0, 1.5, 2.0, 4.0, 5.0) == 0.0        # beyond soft max
    assert score_band(None, 1.5, 2.0, 4.0, 5.0) == 0.5       # neutral on missing


def test_kink_score_is_bounded_sigmoid():
    assert score_kink(0.0) == pytest.approx(0.5)
    assert score_kink(5.0) == pytest.approx(1.0, abs=1e-3)
    assert score_kink(-5.0) == pytest.approx(0.0, abs=1e-3)
    assert score_kink(None) == 0.0


def test_time_score_prefers_short_dated():
    assert score_time(10) == 1.0
    assert score_time(25) == pytest.approx(0.7)
    assert score_time(45) == pytest.approx(0.4)
    assert score_time(2) == 0.0


def test_gamma_penalty_higher_is_less_risk():
    low_gamma = gamma_penalty(0.05, max_risk=400, wing_width=5)
    high_gamma = gamma_penalty(0.18, max_risk=400, wing_width=5)
    assert low_gamma > high_gamma


def test_composite_weighted_blend():
    comps = {"ev": 1.0, "liquidity": 1.0, "iv": 1.0, "gamma": 1.0, "time": 1.0}
    # All sub-scores 1.0 -> composite equals the sum of active weights (~1.0).
    assert composite_score(comps) == pytest.approx(sum(
        DEFAULT_SCORING_CONFIG.weights[k] for k in comps))

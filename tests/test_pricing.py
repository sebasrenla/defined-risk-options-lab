"""Pricing and probability: Black-Scholes, IV inversion, and first-passage."""

import math

import pytest

from optvol.pricing import (
    black_scholes_price,
    expected_value,
    first_passage_hit_probability,
    implied_volatility,
    lognormal_cdf,
    prob_stay_within_barriers,
)


def test_put_call_parity_rate_zero():
    # With rate = 0: C - P = S - K.
    S, K, t, vol = 100.0, 95.0, 0.25, 0.30
    c = black_scholes_price(S, K, t, vol, is_call=True)
    p = black_scholes_price(S, K, t, vol, is_call=False)
    assert c - p == pytest.approx(S - K, abs=1e-9)


def test_price_monotonic_in_vol():
    S, K, t = 100.0, 100.0, 0.25
    prices = [black_scholes_price(S, K, t, v, is_call=True) for v in (0.1, 0.2, 0.4, 0.8)]
    assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))


def test_degenerate_price_is_intrinsic():
    assert black_scholes_price(110, 100, 0.0, 0.2, is_call=True) == pytest.approx(10.0)
    assert black_scholes_price(90, 100, 0.5, 0.0, is_call=False) == pytest.approx(10.0)


@pytest.mark.parametrize("vol", [0.12, 0.25, 0.5, 0.9])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_round_trip(vol, is_call):
    S, K, t = 100.0, 105.0, 0.3
    price = black_scholes_price(S, K, t, vol, is_call=is_call)
    recovered = implied_volatility(price, S, K, t, is_call=is_call)
    assert recovered == pytest.approx(vol, abs=1e-4)


def test_implied_vol_rejects_out_of_bounds_price():
    # Price above the underlying is not invertible for a call.
    assert implied_volatility(150.0, 100.0, 100.0, 0.3, is_call=True) is None
    # Non-positive price -> None.
    assert implied_volatility(0.0, 100.0, 100.0, 0.3) is None


def test_lognormal_cdf_at_driftless_median_is_half():
    S, sig, t = 100.0, 0.3, 0.25
    median = S * math.exp(-0.5 * sig * sig * t)
    assert lognormal_cdf(S, sig, t, median) == pytest.approx(0.5, abs=1e-9)


def test_lognormal_cdf_degenerate_returns_none():
    assert lognormal_cdf(0, 0.3, 0.25, 100) is None
    assert lognormal_cdf(100, 0.3, 0.25, -1) is None


def test_first_passage_nearer_barrier_more_likely():
    near = first_passage_hit_probability(math.log(1.02), 0.3, 0.1)
    far = first_passage_hit_probability(math.log(1.20), 0.3, 0.1)
    assert near > far
    assert 0.0 <= far <= near <= 1.0


def test_first_passage_at_barrier_is_certain():
    assert first_passage_hit_probability(0.0, 0.3, 0.1) == 1.0


def test_stay_within_narrower_band_less_likely():
    wide = prob_stay_within_barriers(100, 90, 110, 0.3, 0.1)
    narrow = prob_stay_within_barriers(100, 97, 103, 0.3, 0.1)
    assert narrow < wide


def test_stay_within_degenerate_returns_none():
    assert prob_stay_within_barriers(100, 110, 90, 0.3, 0.1) is None  # upper <= lower


def test_expected_value_formula():
    assert expected_value(0.6, 100.0, 50.0) == pytest.approx(0.6 * 100 - 0.4 * 50)

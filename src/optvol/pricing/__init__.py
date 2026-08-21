"""Pricing and probability primitives.

``black_scholes``: European pricing and implied-volatility inversion.
``probability``: terminal (POP) and first-passage (path/PHT) probabilities.
"""

from .black_scholes import black_scholes_price, implied_volatility, norm_cdf
from .probability import (
    expected_value,
    first_passage_hit_probability,
    lognormal_cdf,
    prob_stay_within_barriers,
    prob_touch_level,
    terminal_prob_in_range,
)

__all__ = [
    "norm_cdf",
    "black_scholes_price",
    "implied_volatility",
    "lognormal_cdf",
    "terminal_prob_in_range",
    "first_passage_hit_probability",
    "prob_touch_level",
    "prob_stay_within_barriers",
    "expected_value",
]

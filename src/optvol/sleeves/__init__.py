"""Strategy sleeves: covered call, bull put spread, tail hedge (decision logic)."""

from .bull_put_spread import (
    BullPutSpreadCandidate,
    BullPutSpreadConfig,
    BullPutSpreadExitDecision,
    estimate_bull_put_spread_roundtrip_fees,
    evaluate_bull_put_spread_exit_action,
    select_bull_put_spread_candidate,
)
from .covered_call import (
    CoveredCallCandidate,
    CoveredCallConfig,
    CoveredCallRollDecision,
    OptionQuote,
    compute_near_ex_div,
    determine_allocation_multiplier,
    estimate_covered_call_roundtrip_fees,
    estimate_option_slippage,
    evaluate_covered_call_roll_action,
    select_covered_call_candidate,
)
from .tail_hedge import (
    TailHedgeBudgetDecision,
    TailHedgeConfig,
    TailHedgePositionDecision,
    TailHedgePositionSnapshot,
    estimate_tail_hedge_fees,
    evaluate_tail_hedge_budget,
    evaluate_tail_hedge_position_action,
)

__all__ = [
    # shared
    "OptionQuote",
    # covered call
    "CoveredCallConfig",
    "CoveredCallCandidate",
    "CoveredCallRollDecision",
    "estimate_option_slippage",
    "determine_allocation_multiplier",
    "compute_near_ex_div",
    "estimate_covered_call_roundtrip_fees",
    "select_covered_call_candidate",
    "evaluate_covered_call_roll_action",
    # bull put spread
    "BullPutSpreadConfig",
    "BullPutSpreadCandidate",
    "BullPutSpreadExitDecision",
    "estimate_bull_put_spread_roundtrip_fees",
    "select_bull_put_spread_candidate",
    "evaluate_bull_put_spread_exit_action",
    # tail hedge
    "TailHedgeConfig",
    "TailHedgeBudgetDecision",
    "TailHedgePositionSnapshot",
    "TailHedgePositionDecision",
    "evaluate_tail_hedge_budget",
    "estimate_tail_hedge_fees",
    "evaluate_tail_hedge_position_action",
]

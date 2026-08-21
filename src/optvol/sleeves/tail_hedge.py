"""Tail-hedge sleeve: cost-budget governance and position management.

A tail hedge holds long, far-OTM index puts (SPY/QQQ) as portfolio insurance.
Because it is a *cost* (it bleeds premium in calm markets), the discipline is
almost entirely about **spending control** and **management**, which is exactly
what this module captures:

* :func:`evaluate_tail_hedge_budget`: translate an **annualized** cost budget
  (a band of ~1.5% / 1.75% / 2.0% of NAV per year) into a **monthly** target and
  under-/over-spend bands, then flag ``top_up_required`` (under-hedged this
  month) or ``pause_additional_buys`` (over budget). This is the hard risk cap
  that keeps insurance from quietly eating returns.
* :func:`evaluate_tail_hedge_position_action`: the management rule set:
  **monetize** half of a position that has appreciated past a windfall multiple
  (take some crisis-alpha off the table), **hard-roll** in the final week, roll
  as the roll window opens, otherwise hold.
* :func:`estimate_tail_hedge_fees`: per-position transaction cost (open, or
  round-trip).

The contract-selection and budget-constrained sizing steps (choosing the strike
within the target OTM / delta / liquidity windows and sizing the buy to the
remaining monthly budget) are part of the full engine; here we publish the
governance and management decisions, which carry the sleeve's distinctive risk
logic.

Provenance
----------
Ported from the program's ``tail_hedge_engine.py`` (decision functions only).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Mapping

from ..execution.fees_regulatory import (
    CURRENT_OPTION_CLEARING_PER_CONTRACT,
    CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT,
    CURRENT_OPTION_COMMISSION_CAP_PER_LEG,
    CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT,
    CURRENT_OPTION_ORF_PER_CONTRACT,
    CURRENT_OPTION_TAF_PER_CONTRACT,
    estimate_option_transaction_fees,
)


@dataclass(frozen=True)
class TailHedgeConfig:
    """Tail-hedge sleeve parameters (illustrative defaults)."""

    primary_symbol: str = "SPY"
    secondary_symbol: str = "QQQ"
    dte_min: int = 25
    dte_max: int = 45
    target_dte: int = 35
    otm_pct_min: float = 0.03
    otm_pct_max: float = 0.07
    target_otm_pct: float = 0.05
    target_abs_delta_hint: float = 0.05
    max_abs_delta_hint: float = 0.12
    min_open_interest: int = 500
    min_volume: int = 50
    max_spread_pct_of_mid: float = 0.12
    annual_budget_pct_min: float = 0.015
    annual_budget_pct_target: float = 0.0175
    annual_budget_pct_max: float = 0.020
    monthly_under_spend_ratio: float = 0.80
    monthly_over_spend_ratio: float = 1.20
    windfall_premium_multiple: float = 2.5
    windfall_monetize_fraction: float = 0.50
    roll_window_start_dte: int = 10
    roll_window_hard_dte: int = 7
    contract_multiplier: int = 100
    fee_open_commission: float = CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT
    fee_close_commission: float = CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT
    fee_commission_cap_per_leg: float = CURRENT_OPTION_COMMISSION_CAP_PER_LEG
    fee_clearing: float = CURRENT_OPTION_CLEARING_PER_CONTRACT
    fee_orf: float = CURRENT_OPTION_ORF_PER_CONTRACT
    fee_taf: float = CURRENT_OPTION_TAF_PER_CONTRACT
    fee_symbol_addons: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TailHedgeBudgetDecision:
    nav: float
    spend_mtd_usd: float
    annual_budget_min_usd: float
    annual_budget_target_usd: float
    annual_budget_max_usd: float
    monthly_target_spend_usd: float
    monthly_lower_band_usd: float
    monthly_upper_band_usd: float
    spend_pct_of_target: float | None
    top_up_required: bool
    pause_additional_buys: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TailHedgePositionSnapshot:
    symbol: str
    expiry: date
    strike: float
    quantity: int
    entry_premium: float
    current_option_mark: float
    dte_remaining: int
    monetized_quantity: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["expiry"] = self.expiry.isoformat()
        return payload


@dataclass(frozen=True)
class TailHedgePositionDecision:
    symbol: str
    action_code: str
    should_roll_now: bool
    contracts_to_roll: int
    contracts_to_monetize: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_tail_hedge_budget(
    *, nav: float, spend_mtd_usd: float, config: TailHedgeConfig = TailHedgeConfig()
) -> TailHedgeBudgetDecision:
    """Turn an annualized NAV cost budget into monthly bands and spend flags."""
    if nav <= 0:
        raise ValueError("nav must be positive")
    if spend_mtd_usd < 0:
        raise ValueError("spend_mtd_usd must be non-negative")

    annual_budget_min_usd = nav * config.annual_budget_pct_min
    annual_budget_target_usd = nav * config.annual_budget_pct_target
    annual_budget_max_usd = nav * config.annual_budget_pct_max
    monthly_target_spend_usd = annual_budget_target_usd / 12.0
    monthly_lower_band_usd = monthly_target_spend_usd * config.monthly_under_spend_ratio
    monthly_upper_band_usd = monthly_target_spend_usd * config.monthly_over_spend_ratio
    spend_pct_of_target = spend_mtd_usd / monthly_target_spend_usd if monthly_target_spend_usd > 0 else None

    return TailHedgeBudgetDecision(
        nav=float(nav), spend_mtd_usd=float(spend_mtd_usd),
        annual_budget_min_usd=float(annual_budget_min_usd),
        annual_budget_target_usd=float(annual_budget_target_usd),
        annual_budget_max_usd=float(annual_budget_max_usd),
        monthly_target_spend_usd=float(monthly_target_spend_usd),
        monthly_lower_band_usd=float(monthly_lower_band_usd),
        monthly_upper_band_usd=float(monthly_upper_band_usd),
        spend_pct_of_target=(float(spend_pct_of_target) if spend_pct_of_target is not None else None),
        top_up_required=spend_mtd_usd < monthly_lower_band_usd,
        pause_additional_buys=spend_mtd_usd > monthly_upper_band_usd,
        diagnostics={
            "annual_budget_pct_target": config.annual_budget_pct_target,
            "monthly_under_spend_ratio": config.monthly_under_spend_ratio,
            "monthly_over_spend_ratio": config.monthly_over_spend_ratio,
        },
    )


def estimate_tail_hedge_fees(
    config: TailHedgeConfig, *, quantity: int = 1, symbol: str | None = None, roundtrip: bool = True
) -> float:
    """Per-position transaction fees (open only, or round-trip)."""
    contracts = max(0, int(quantity))
    if contracts == 0:
        return 0.0
    symbol_addon = 0.0
    if symbol:
        raw = config.fee_symbol_addons.get(symbol.upper().strip())
        if raw is not None:
            try:
                symbol_addon = float(raw)
            except (TypeError, ValueError):
                symbol_addon = 0.0
    open_fees = estimate_option_transaction_fees(
        contracts=contracts, commission_per_contract=config.fee_open_commission,
        commission_cap_per_leg=config.fee_commission_cap_per_leg, clearing_per_contract=config.fee_clearing,
        orf_per_contract=config.fee_orf, taf_per_contract=config.fee_taf,
        symbol_addon_per_contract=symbol_addon, is_sale=False)
    if not roundtrip:
        return open_fees
    close_fees = estimate_option_transaction_fees(
        contracts=contracts, commission_per_contract=config.fee_close_commission,
        commission_cap_per_leg=config.fee_commission_cap_per_leg, clearing_per_contract=config.fee_clearing,
        orf_per_contract=config.fee_orf, taf_per_contract=config.fee_taf,
        symbol_addon_per_contract=symbol_addon, is_sale=True)
    return open_fees + close_fees


def evaluate_tail_hedge_position_action(
    position: TailHedgePositionSnapshot, config: TailHedgeConfig = TailHedgeConfig()
) -> TailHedgePositionDecision:
    """Manage an open tail hedge: monetize a windfall, hard-roll, roll, or hold.

    Priority: monetize half if the mark has risen past the windfall multiple and
    there is still time -> hard roll inside the final week -> roll once the roll
    window opens -> otherwise hold.
    """
    if position.quantity <= 0:
        raise ValueError("position.quantity must be positive")
    if position.entry_premium <= 0:
        raise ValueError("position.entry_premium must be positive")

    current_multiple = position.current_option_mark / position.entry_premium
    remaining_quantity = max(position.quantity - position.monetized_quantity, 0)
    symbol = position.symbol.upper().strip()
    diagnostics = {
        "expiry": position.expiry.isoformat(), "strike": position.strike, "quantity": position.quantity,
        "monetized_quantity": position.monetized_quantity, "current_multiple": current_multiple,
        "dte_remaining": position.dte_remaining,
    }

    if current_multiple >= config.windfall_premium_multiple and position.dte_remaining > config.roll_window_start_dte:
        contracts_to_monetize = int(remaining_quantity * config.windfall_monetize_fraction)
        if contracts_to_monetize >= 1:
            diagnostics["windfall_threshold"] = config.windfall_premium_multiple
            return TailHedgePositionDecision(symbol, "monetize_windfall_half", False, 0, contracts_to_monetize, diagnostics)
        diagnostics["fractional_windfall_unavailable"] = True

    if position.dte_remaining <= config.roll_window_hard_dte:
        return TailHedgePositionDecision(symbol, "roll_monthly_hard", True, position.quantity, 0, diagnostics)
    if position.dte_remaining <= config.roll_window_start_dte:
        return TailHedgePositionDecision(symbol, "roll_monthly", True, position.quantity, 0, diagnostics)
    return TailHedgePositionDecision(symbol, "hold", False, 0, 0, diagnostics)

"""Portfolio risk overlay: caps, regime cut, and entry throttle."""

from optvol.risk import (
    BullPutSpreadRiskInput,
    CoveredCallRiskInput,
    PortfolioRiskConfig,
    evaluate_portfolio_risk_layer,
)
from optvol.risk.portfolio_risk import (
    REJECT_RUN_ENTRY_LIMIT_REACHED,
    REJECT_SECTOR_CAP,
)

CTX = {"AAA": {"sector": "Tech"}, "BBB": {"sector": "Tech"}, "CCC": {"sector": "Energy"}}


def test_covered_call_sized_to_symbol_cap():
    # 4% of $1M = $40k symbol cap; per-lot notional = 100 * 100 = $10k -> 4 lots.
    cc = CoveredCallRiskInput("AAA", nav_cap_pct=0.04, allocation_multiplier=1.0, underlying_price=100.0)
    ev = evaluate_portfolio_risk_layer(
        as_of_date="2026-06-01", nav=1_000_000, vix_level=15.0,
        program1_candidates=[cc], symbol_context=CTX)
    assert len(ev.accepted_program1) == 1
    assert ev.accepted_program1[0].approved_quantity == 4


def test_entry_limit_throttle():
    cc = CoveredCallRiskInput("AAA", 0.04, 1.0, 100.0)
    cfg = PortfolioRiskConfig(max_new_program1_entries_per_run=1)
    ev = evaluate_portfolio_risk_layer(
        as_of_date="2026-06-01", nav=1_000_000, vix_level=15.0,
        program1_candidates=[cc, cc], symbol_context=CTX, config=cfg)
    assert len(ev.accepted_program1) == 1
    assert len(ev.rejected_program1) == 1
    assert REJECT_RUN_ENTRY_LIMIT_REACHED in ev.rejected_program1[0].reason_codes


def test_vix_regime_cut_reduces_size():
    cc = CoveredCallRiskInput("AAA", 0.04, 1.0, 100.0)
    base = PortfolioRiskConfig(vix_regime_cut_threshold=None)
    cut = PortfolioRiskConfig(vix_regime_cut_threshold=20.0, vix_regime_cut_multiplier=0.5)
    q_base = evaluate_portfolio_risk_layer(as_of_date="2026-06-01", nav=1_000_000, vix_level=30.0,
             program1_candidates=[cc], symbol_context=CTX, config=base).accepted_program1[0].approved_quantity
    q_cut = evaluate_portfolio_risk_layer(as_of_date="2026-06-01", nav=1_000_000, vix_level=30.0,
             program1_candidates=[cc], symbol_context=CTX, config=cut).accepted_program1[0].approved_quantity
    assert q_cut < q_base


def test_bps_consumes_aggregate_and_sleeve_risk():
    bps = BullPutSpreadRiskInput("CCC", 0.04, 1.0, max_loss_model=500.0, estimated_roundtrip_fees=5.0)
    ev = evaluate_portfolio_risk_layer(
        as_of_date="2026-06-01", nav=1_000_000, vix_level=15.0,
        program1_candidates=[bps], symbol_context=CTX)
    assert ev.accepted_program1[0].incremental_risk_usd > 0
    assert ev.ending_state.bps_risk_used_usd == ev.accepted_program1[0].incremental_risk_usd


def test_sector_cap_shared_across_symbols():
    # Two Tech names competing for the 20% sector cap.
    ccs = [CoveredCallRiskInput("AAA", 0.20, 1.0, 100.0),
           CoveredCallRiskInput("BBB", 0.20, 1.0, 100.0)]
    ev = evaluate_portfolio_risk_layer(
        as_of_date="2026-06-01", nav=1_000_000, vix_level=15.0,
        program1_candidates=ccs, symbol_context=CTX)
    tech_used = ev.ending_state.sector_risk_used_usd.get("Tech", 0.0)
    assert tech_used <= 1_000_000 * 0.20 + 1e-6

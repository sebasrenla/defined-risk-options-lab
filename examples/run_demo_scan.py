"""End-to-end demo: drive the whole library on a synthetic chain (no vendor data).

Walks through every layer of ``optvol`` on a reproducible synthetic option chain:

1. Pricing / probability — price an option and round-trip its implied vol.
2. Signal — compute the IV-curvature ("kink") across the smile.
3. Structure — build a broken-wing butterfly and show its terminal (POP) vs
   path (PHT) probabilities and expected value.
4. Sleeves — select a covered-call and a bull-put-spread candidate.
5. Risk — pass the sleeve candidates through the portfolio risk overlay.
6. Economics — book a buy-write to the cash ledger and take a snapshot.

Run:
    python examples/run_demo_scan.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from optvol.economics import build_buy_write_open_events, build_replay_economics_snapshot  # noqa: E402
from optvol.pricing import black_scholes_price, implied_volatility  # noqa: E402
from optvol.risk import (  # noqa: E402
    BullPutSpreadRiskInput,
    CoveredCallRiskInput,
    evaluate_portfolio_risk_layer,
)
from optvol.signals import StrikeIV, apply_history_z, compute_kinks, effective_kink_z  # noqa: E402
from optvol.sleeves import (  # noqa: E402
    BullPutSpreadConfig,
    CoveredCallConfig,
    select_bull_put_spread_candidate,
    select_covered_call_candidate,
)
from optvol.structures import BrokenWingButterfly  # noqa: E402

from generate_synthetic_chain import check_static_arbitrage, generate_synthetic_chain  # noqa: E402


def hr(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> None:
    SPOT = 300.0
    AS_OF = date(2026, 6, 1)
    chain = generate_synthetic_chain(spot=SPOT, as_of=AS_OF, expiries_dte=(30, 60),
                                     strikes_per_side=12, strike_step_pct=0.02,
                                     inject_kink_at_offset=3)
    print(f"Generated synthetic chain: {len(chain)} quotes, spot={SPOT}, as_of={AS_OF}")
    arb = check_static_arbitrage(spot=SPOT)
    print(f"SSVI surface verified: arbitrage_free={arb['arbitrage_free']} "
          f"(min risk-neutral density={arb['min_density']:.2e} >= 0)")

    # 1) Pricing / probability -------------------------------------------------
    hr("1. Pricing & implied volatility")
    atm = min((q for q in chain if q.option_type == "call" and q.dte == 30),
              key=lambda q: abs(q.strike - SPOT))
    mid = (atm.bid + atm.ask) / 2.0
    t = atm.dte / 365.0
    recovered = implied_volatility(mid, SPOT, atm.strike, t, is_call=True)
    price_at_true = black_scholes_price(SPOT, atm.strike, t, atm.iv, is_call=True)
    print(f"ATM call K={atm.strike}: mid={mid:.3f}, true IV={atm.iv:.4f}")
    print(f"  BS price at true IV = {price_at_true:.3f}")
    print(f"  IV recovered from mid via bisection = {recovered:.4f}")

    # 2) Signal ----------------------------------------------------------------
    hr("2. IV-curvature (kink) signal")
    calls30 = sorted((q for q in chain if q.option_type == "call" and q.dte == 30),
                     key=lambda q: q.strike)
    kinks = compute_kinks([StrikeIV(q.strike, q.iv) for q in calls30])
    # (A rolling per-strike history would populate kink_z; here we show the
    #  cross-sectional z only.)
    apply_history_z(kinks, history={}, min_history=5)
    top = max(kinks, key=lambda k: (k.kink_z_cross if k.kink_z_cross is not None else -9))
    print("strike   kink      z_cross")
    for k in kinks:
        flag = "  <-- richest (injected kink)" if k is top else ""
        z = f"{k.kink_z_cross:+.2f}" if k.kink_z_cross is not None else "  n/a"
        print(f"{k.strike:7.2f}  {k.kink:+.4f}   {z}{flag}")

    # 3) Structure -------------------------------------------------------------
    hr("3. Broken-wing butterfly: terminal POP vs path PHT vs EV")
    # A put BWB centered at the money (body 300) with unequal wings (270 / 315).
    bwb = BrokenWingButterfly("put", lower_strike=270, body_strike=300, upper_strike=315,
                              net_credit=1.50)
    mp, mr, free = bwb.metrics()
    lo_be, hi_be = bwb.break_evens()
    iv_body = next(q.iv for q in chain if q.option_type == "put" and q.dte == 30 and q.strike == 300)
    pop = bwb.probability_of_profit(SPOT, iv_body, 30)
    pht = bwb.hold_probability(SPOT, iv_body, 30, exit_days=10)
    ev = bwb.expected_value(SPOT, iv_body, 30, exit_days=10, profit_target_pct=0.5, stop_loss_pct=1.0)
    print(f"structure: put BWB 270/300/315, net credit 1.50, body IV={iv_body:.3f}")
    print(f"  max_profit={mp:.2f}  max_risk={mr:.2f}  break-evens=({lo_be:.2f}, {hi_be:.2f})")
    print(f"  terminal POP = {pop:.4f}   (probability profitable AT expiry)")
    print(f"  path PHT     = {pht:.4f}   (probability of never breaching a break-even)")
    print(f"  --> POP {'>' if pop > pht else '<='} PHT: terminal probability overstates edge; "
          f"EV is scored off PHT = {ev:+.4f}")

    # 4) Sleeves ---------------------------------------------------------------
    hr("4. Sleeve selection (covered call + bull put spread)")
    cc_cand, cc_diag = select_covered_call_candidate(
        quotes=chain, symbol="DEMO", as_of_date=AS_OF, nav_cap_pct=0.04,
        config=CoveredCallConfig(), iv_rank=45.0)
    bps_cand, bps_diag = select_bull_put_spread_candidate(
        quotes=chain, symbol="DEMO", as_of_date=AS_OF, nav_cap_pct=0.04,
        config=BullPutSpreadConfig(), iv_rank=45.0)
    if cc_cand:
        print(f"covered call: sell {cc_cand.strike}C @ mid {cc_cand.mid:.2f}, delta {cc_cand.delta:.3f}, "
              f"DTE {cc_cand.dte}, alloc x{cc_cand.allocation_multiplier} ({cc_cand.allocation_reason})")
    else:
        print(f"covered call: no candidate ({cc_diag})")
    if bps_cand:
        print(f"bull put spread: {bps_cand.short_strike}/{bps_cand.long_strike}P, "
              f"credit {bps_cand.structure_mid:.2f}, width {bps_cand.width:.1f}, "
              f"max_loss ${bps_cand.max_loss_model:.0f}, alloc x{bps_cand.allocation_multiplier}")
    else:
        print(f"bull put spread: no candidate ({bps_diag})")

    # 5) Risk ------------------------------------------------------------------
    hr("5. Portfolio risk overlay")
    candidates = []
    if cc_cand:
        candidates.append(CoveredCallRiskInput("DEMO", cc_cand.nav_cap_pct,
                          cc_cand.allocation_multiplier, cc_cand.underlying_price))
    if bps_cand:
        candidates.append(BullPutSpreadRiskInput("DEMO", bps_cand.nav_cap_pct,
                          bps_cand.allocation_multiplier, bps_cand.max_loss_model,
                          bps_cand.estimated_roundtrip_fees))
    evaluation = evaluate_portfolio_risk_layer(
        as_of_date=AS_OF, nav=1_000_000.0, vix_level=18.0,
        program1_candidates=candidates, symbol_context={"DEMO": {"sector": "Demo"}})
    for d in evaluation.accepted_program1:
        print(f"  ACCEPT {d.sub_sleeve} {d.symbol}: qty {d.approved_quantity}, "
              f"incremental risk ${d.incremental_risk_usd:,.0f}")
    for d in evaluation.rejected_program1:
        print(f"  REJECT {d.sub_sleeve} {d.symbol}: {', '.join(d.reason_codes)}")
    print(f"  ending aggregate defined-risk used: ${evaluation.ending_state.aggregate_defined_risk_used_usd:,.0f}"
          f" / cap ${1_000_000 * 0.35:,.0f}")

    # 6) Economics -------------------------------------------------------------
    hr("6. Cash-ledger economics (book a buy-write)")
    events = build_buy_write_open_events(trade_date=AS_OF, symbol="DEMO",
             stock_fill_price=SPOT, short_call_fill_price=(cc_cand.mid if cc_cand else 2.0),
             position_group_id="demo-cc-1")
    snap = build_replay_economics_snapshot(
        session_date=AS_OF, opening_settled_cash_usd=1_000_000.0, ledger_events=events,
        long_equity_market_value_usd=SPOT * 100, option_market_value_usd=0.0,
        defined_risk_requirement_usd=0.0)
    print(f"  {len(events)} ledger events booked (stock buy, call sell, and their fees)")
    print(f"  settled cash after open = ${snap.settled_cash_usd:,.2f}")
    print(f"  net liquidation value   = ${snap.net_liquidation_value_usd:,.2f}")
    print("\nDemo complete. All figures above were produced from synthetic data only.")


if __name__ == "__main__":
    main()

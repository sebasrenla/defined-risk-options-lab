"""Time-varying regulatory & exchange transaction fees.

Realistic backtests must charge the fees that were *actually in effect on each
trade date*, not today's rates. US options and equities carry several small
per-contract / per-share charges that change on published effective dates:

* **Options Regulatory Fee (ORF)** and **clearing**: per contract.
* **FINRA Trading Activity Fee (TAF)**: per contract (options) / per share
  (equity sales), with a per-order cap on equity sales.
* **SEC Section 31 fee**: per dollar of equity *sale* proceeds (quoted per
  $1,000,000), which the SEC re-sets periodically.

All schedules below are **public** (SEC / FINRA / exchange fee filings), so they
are safe to publish. Broker commissions are configured separately (they are
account-specific), see :mod:`optvol.execution.fees_butterfly`.

Rates are resolved by looking up the most recent effective date on or before the
trade date. Values past the current year are the published forward schedule and
should be re-verified against the source before relying on them.

Provenance
----------
Ported from the program's ``broker_fees.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime


# --- Current live fee-schedule constants -------------------------------------
# Commission values are account-specific (illustrative of a $1-open / $0-close /
# $10-cap options schedule); clearing / ORF / TAF are the current published
# regulatory rates. The dated schedules below supersede these for historical
# backtests.
CURRENT_LIVE_FEE_SCHEDULE_AS_OF = date(2026, 3, 14)

CURRENT_OPTION_OPEN_COMMISSION_PER_CONTRACT = 1.00
CURRENT_OPTION_CLOSE_COMMISSION_PER_CONTRACT = 0.00
CURRENT_OPTION_COMMISSION_CAP_PER_LEG = 10.00
CURRENT_OPTION_CLEARING_PER_CONTRACT = 0.10
CURRENT_OPTION_ORF_PER_CONTRACT = 0.02295
CURRENT_OPTION_TAF_PER_CONTRACT = 0.00329

CURRENT_EQUITY_OPEN_COMMISSION_PER_ORDER = 0.00
CURRENT_EQUITY_CLOSE_COMMISSION_PER_ORDER = 0.00
CURRENT_EQUITY_CLEARING_PER_SHARE = 0.0008
CURRENT_EQUITY_SELL_TAF_PER_SHARE = 0.000195
CURRENT_EQUITY_SELL_TAF_CAP_PER_ORDER = 9.79


# --- Public fee schedules (effective_date, rate), ascending by date ----------

OPTION_TAF_PER_CONTRACT: tuple[tuple[date, float], ...] = (
    (date(2016, 1, 1), 0.00200),
    (date(2022, 1, 1), 0.00218),
    (date(2023, 1, 1), 0.00244),
    (date(2024, 1, 1), 0.00279),
    (date(2026, 1, 1), 0.00329),
    (date(2027, 1, 1), 0.00390),
    (date(2028, 1, 1), 0.00466),
    (date(2029, 1, 1), 0.00550),
)

EQUITY_SELL_TAF_PER_SHARE: tuple[tuple[date, float], ...] = (
    (date(2016, 1, 1), 0.000119),
    (date(2022, 1, 1), 0.000130),
    (date(2023, 1, 1), 0.000145),
    (date(2024, 1, 1), 0.000166),
    (date(2026, 1, 1), 0.000195),
    (date(2027, 1, 1), 0.000232),
    (date(2028, 1, 1), 0.000277),
    (date(2029, 1, 1), 0.000327),
)

EQUITY_SELL_TAF_CAP_PER_ORDER: tuple[tuple[date, float], ...] = (
    (date(2016, 1, 1), 5.95),
    (date(2022, 1, 1), 6.49),
    (date(2023, 1, 1), 7.27),
    (date(2024, 1, 1), 8.30),
    (date(2026, 1, 1), 9.79),
    (date(2027, 1, 1), 11.61),
    (date(2028, 1, 1), 13.88),
    (date(2029, 1, 1), 16.35),
)

SEC_SECTION31_PER_MILLION: tuple[tuple[date, float], ...] = (
    (date(2016, 1, 1), 18.40),
    (date(2016, 2, 16), 21.80),
    (date(2017, 7, 4), 23.10),
    (date(2018, 5, 22), 13.00),
    (date(2019, 4, 16), 20.70),
    (date(2020, 2, 18), 22.10),
    (date(2021, 2, 25), 5.10),
    (date(2022, 5, 14), 22.90),
    (date(2023, 2, 27), 8.00),
    (date(2024, 5, 22), 27.80),
    (date(2025, 5, 14), 0.00),
    (date(2026, 4, 4), 20.60),
)


@dataclass(frozen=True)
class RegulatoryFeeSnapshot:
    """The resolved regulatory rates for a given trade date."""

    trade_date: date
    option_taf_per_contract: float
    equity_sell_taf_per_share: float
    equity_sell_taf_cap_per_order: float
    sec_section31_per_million: float

    def to_dict(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["trade_date"] = self.trade_date.isoformat()
        return payload


def _coerce_trade_date(value: date | datetime | str | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value).strip())


def _resolve_rate(schedule: tuple[tuple[date, float], ...], trade_date: date | datetime | str | None) -> float:
    """Return the rate from the latest effective date on or before ``trade_date``."""
    resolved = _coerce_trade_date(trade_date)
    active = 0.0
    for effective_date, rate in schedule:
        if resolved >= effective_date:
            active = rate
    return active


def option_taf_per_contract(trade_date=None) -> float:
    return _resolve_rate(OPTION_TAF_PER_CONTRACT, trade_date)


def equity_sell_taf_per_share(trade_date=None) -> float:
    return _resolve_rate(EQUITY_SELL_TAF_PER_SHARE, trade_date)


def equity_sell_taf_cap_per_order(trade_date=None) -> float:
    return _resolve_rate(EQUITY_SELL_TAF_CAP_PER_ORDER, trade_date)


def sec_section31_per_million(trade_date=None) -> float:
    return _resolve_rate(SEC_SECTION31_PER_MILLION, trade_date)


def regulatory_fee_snapshot(trade_date=None) -> RegulatoryFeeSnapshot:
    resolved = _coerce_trade_date(trade_date)
    return RegulatoryFeeSnapshot(
        trade_date=resolved,
        option_taf_per_contract=option_taf_per_contract(resolved),
        equity_sell_taf_per_share=equity_sell_taf_per_share(resolved),
        equity_sell_taf_cap_per_order=equity_sell_taf_cap_per_order(resolved),
        sec_section31_per_million=sec_section31_per_million(resolved),
    )


def estimate_option_transaction_fees(
    *,
    contracts: int,
    commission_per_contract: float,
    commission_cap_per_leg: float,
    clearing_per_contract: float,
    orf_per_contract: float,
    is_sale: bool,
    taf_per_contract: float | None = None,
    symbol_addon_per_contract: float = 0.0,
    trade_date=None,
) -> float:
    """Total option transaction fees for one leg (commission + clearing + ORF +
    TAF on sales + optional symbol add-on), with the per-leg commission cap
    applied."""
    n = max(0, int(contracts))
    if n == 0:
        return 0.0
    taf = option_taf_per_contract(trade_date) if taf_per_contract is None else float(taf_per_contract)
    commission = min(commission_per_contract * n, commission_cap_per_leg)
    fees = commission + n * (clearing_per_contract + orf_per_contract + symbol_addon_per_contract)
    if is_sale:
        fees += n * taf
    return fees


def estimate_equity_transaction_fees(
    *,
    shares: int,
    commission_per_order: float,
    clearing_per_share: float,
    is_sale: bool,
    price_per_share: float | None = None,
    sell_taf_per_share: float | None = None,
    sell_taf_cap_per_order: float | None = None,
    trade_date=None,
) -> float:
    """Total equity transaction fees. On sales, adds capped TAF and the SEC
    Section 31 fee (on proceeds)."""
    n = max(0, int(shares))
    if n == 0:
        return 0.0
    fees = commission_per_order + n * clearing_per_share
    if not is_sale:
        return fees
    taf = equity_sell_taf_per_share(trade_date) if sell_taf_per_share is None else float(sell_taf_per_share)
    cap = equity_sell_taf_cap_per_order(trade_date) if sell_taf_cap_per_order is None else float(sell_taf_cap_per_order)
    fees += min(n * taf, cap)
    if price_per_share is not None and price_per_share > 0:
        fees += (n * price_per_share / 1_000_000.0) * sec_section31_per_million(trade_date)
    return fees

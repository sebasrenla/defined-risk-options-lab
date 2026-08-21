"""Contract-driven dataset validation and liquidity-snapshot evaluation.

Two responsibilities:

1. **Validation**: check a CSV dataset against the versioned data contract
   (:mod:`optvol.backtest.data_contract`): required columns present, non-nullable
   values populated, numeric/integer parsing, value bounds, enum membership, and
   cross-row structural invariants (e.g. a bull-put-spread position group must
   contain exactly one short put and one long put of matching size). Errors are
   collected by code with capped samples, so a bad file yields a structured
   report rather than a stack trace.

2. **Liquidity-snapshot evaluation**: scan a chain snapshot for
   covered-call / bull-put-spread candidacy within target delta/DTE windows and
   the contract's liquidity gates, then aggregate candidacy and pass-rates across
   many snapshots into a per-symbol opportunity summary.

Provenance
----------
Ported from the program's ``validators.py`` (import path updated to the local
``data_contract`` module).
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from .data_contract import (
    LiquidityGateThresholds,
    get_dataset_schema,
    get_liquidity_thresholds,
    load_contract_schema,
)


@dataclass
class ValidationSummary:
    dataset_name: str
    file_path: str
    rows_checked: int = 0
    errors_by_code: dict[str, int] = field(default_factory=dict)
    sample_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.errors_by_code

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["is_valid"] = self.is_valid
        return payload


@dataclass
class SnapshotLiquidityFlags:
    seen: bool = False
    cc_candidate: bool = False
    cc_pass: bool = False
    bps_candidate: bool = False
    bps_pass: bool = False
    best_cc_spread_pct: float | None = None
    best_bps_spread_pct: float | None = None


@dataclass
class SymbolLiquidityAggregate:
    snapshots_seen: int = 0
    cc_candidate_snapshots: int = 0
    cc_pass_snapshots: int = 0
    bps_candidate_snapshots: int = 0
    bps_pass_snapshots: int = 0
    both_pass_snapshots: int = 0
    cc_spread_samples: list[float] = field(default_factory=list)
    bps_spread_samples: list[float] = field(default_factory=list)


def _is_blank(value: str | None) -> bool:
    return value is None or value.strip() == ""


def _to_float(value: str | None) -> float | None:
    if _is_blank(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: str | None) -> int | None:
    if _is_blank(value):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not parsed.is_integer():
        return None
    return int(parsed)


def _spread_pct_of_mid(bid: float | None, ask: float | None) -> float | None:
    if bid is None or ask is None:
        return None
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return (ask - bid) / mid


def _safe_div(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def normalize_corporate_action_flag(value: str | None) -> str | None:
    """Map assorted truthy/falsey spellings to canonical ``"1"`` / ``"0"``."""
    if _is_blank(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return "1"
    if normalized in {"0", "false", "no", "n"}:
        return "0"
    return None


def _record_error(
    errors: "Counter[str]", samples: list[str], code: str, message: str, max_samples: int
) -> None:
    errors[code] += 1
    if len(samples) < max_samples:
        samples.append(message)


def _validate_position_group_invariants(
    position_group_rows: list[dict[str, Any]],
    errors: "Counter[str]",
    sample_errors: list[str],
    max_error_samples: int,
) -> None:
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in position_group_rows:
        grouped_rows[row["position_group_id"]].append(row)

    for position_group_id, rows in grouped_rows.items():
        short_rows = [row for row in rows if row["position_leg_role"] == "short_put"]
        long_rows = [row for row in rows if row["position_leg_role"] == "long_put"]

        if len(short_rows) != 1 or len(long_rows) != 1:
            _record_error(
                errors, sample_errors, "bps_group_role_invariant_violation",
                (f"group={position_group_id} expected exactly one short_put and one long_put "
                 f"got short_put={len(short_rows)} long_put={len(long_rows)}"),
                max_error_samples,
            )

        sizes = {row["size"] for row in rows if row["size"] is not None}
        if len(sizes) > 1:
            _record_error(
                errors, sample_errors, "bps_group_size_mismatch",
                f"group={position_group_id} has mismatched leg sizes: {sorted(sizes)}",
                max_error_samples,
            )

        for row in [r for r in rows if r["option_type"] != "put"]:
            _record_error(
                errors, sample_errors, "bps_group_option_type_violation",
                (f"group={position_group_id} row={row['row_number']} uses "
                 f"position_leg_role={row['position_leg_role']} with option_type={row['option_type']}"),
                max_error_samples,
            )


def validate_dataset_csv(
    csv_path: str | Path,
    dataset_name: str,
    schema: dict[str, Any] | None = None,
    max_error_samples: int = 30,
) -> ValidationSummary:
    """Validate a CSV file against the named dataset in the contract."""
    if schema is None:
        schema = load_contract_schema()
    path = Path(csv_path)
    dataset_schema = get_dataset_schema(schema, dataset_name)

    required_columns = set(dataset_schema.get("required_columns", []))
    nullable_columns = set(dataset_schema.get("nullable_columns", []))
    numeric_columns = set(dataset_schema.get("numeric_columns", []))
    integer_columns = set(dataset_schema.get("integer_columns", []))
    enum_constraints = dataset_schema.get("enum_constraints", {})
    bounds = dataset_schema.get("bounds", {})

    errors: "Counter[str]" = Counter()
    sample_errors: list[str] = []
    summary = ValidationSummary(dataset_name=dataset_name, file_path=str(path))
    position_group_rows: list[dict[str, Any]] = []

    if not path.exists():
        summary.errors_by_code = {"missing_file": 1}
        summary.sample_errors = [f"{path} does not exist"]
        return summary

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - actual_columns)
        if missing_columns:
            for missing in missing_columns:
                _record_error(errors, sample_errors, "missing_required_column",
                              f"Missing required column: {missing}", max_error_samples)
            summary.errors_by_code = dict(errors)
            summary.sample_errors = sample_errors
            return summary

        for row_number, row in enumerate(reader, start=2):
            summary.rows_checked += 1

            for column in required_columns:
                value = row.get(column)
                if column not in nullable_columns and _is_blank(value):
                    _record_error(errors, sample_errors, "missing_required_value",
                                  f"row={row_number} col={column} value is blank", max_error_samples)

            numeric_values: dict[str, float | None] = {}
            for column in numeric_columns:
                raw = row.get(column)
                if _is_blank(raw):
                    if column not in nullable_columns:
                        _record_error(errors, sample_errors, "missing_numeric_value",
                                      f"row={row_number} col={column} missing numeric value", max_error_samples)
                    numeric_values[column] = None
                    continue

                parsed_float = _to_float(raw)
                if parsed_float is None:
                    _record_error(errors, sample_errors, "numeric_parse_error",
                                  f"row={row_number} col={column} cannot parse '{raw}'", max_error_samples)
                    numeric_values[column] = None
                    continue

                numeric_values[column] = parsed_float

                if column in integer_columns and _to_int(raw) is None:
                    _record_error(errors, sample_errors, "integer_parse_error",
                                  f"row={row_number} col={column} expected integer got '{raw}'", max_error_samples)

                bound = bounds.get(column)
                if not bound:
                    continue
                if "min" in bound and parsed_float < float(bound["min"]):
                    _record_error(errors, sample_errors, "bound_min_violation",
                                  f"row={row_number} col={column} value={parsed_float} < min={bound['min']}", max_error_samples)
                if "max" in bound and parsed_float > float(bound["max"]):
                    _record_error(errors, sample_errors, "bound_max_violation",
                                  f"row={row_number} col={column} value={parsed_float} > max={bound['max']}", max_error_samples)
                if "min_exclusive" in bound and parsed_float <= float(bound["min_exclusive"]):
                    _record_error(errors, sample_errors, "bound_min_exclusive_violation",
                                  f"row={row_number} col={column} value={parsed_float} <= min_exclusive={bound['min_exclusive']}", max_error_samples)
                if "max_exclusive" in bound and parsed_float >= float(bound["max_exclusive"]):
                    _record_error(errors, sample_errors, "bound_max_exclusive_violation",
                                  f"row={row_number} col={column} value={parsed_float} >= max_exclusive={bound['max_exclusive']}", max_error_samples)

            for column, allowed_values in enum_constraints.items():
                raw = row.get(column)
                if _is_blank(raw):
                    continue
                value_to_check = str(raw)
                if dataset_name == "context_snapshot" and column == "corporate_action_flag":
                    normalized = normalize_corporate_action_flag(raw)
                    if normalized is None:
                        _record_error(errors, sample_errors, "enum_violation",
                                      f"row={row_number} col={column} value='{raw}' not in {allowed_values}", max_error_samples)
                        continue
                    value_to_check = normalized
                elif dataset_name == "position_snapshot" and column == "position_leg_role":
                    value_to_check = str(raw).strip().lower()

                if value_to_check not in set(allowed_values):
                    _record_error(errors, sample_errors, "enum_violation",
                                  f"row={row_number} col={column} value='{raw}' not in {allowed_values}", max_error_samples)

            if dataset_name == "chain_snapshot":
                bid = numeric_values.get("bid")
                ask = numeric_values.get("ask")
                if bid is not None and ask is not None and ask < bid:
                    _record_error(errors, sample_errors, "ask_less_than_bid",
                                  f"row={row_number} ask={ask} < bid={bid}", max_error_samples)
            elif dataset_name == "position_snapshot":
                position_leg_role = str(row.get("position_leg_role", "") or "").strip().lower()
                position_group_id = str(row.get("position_group_id", "") or "").strip()
                if position_leg_role:
                    if not position_group_id:
                        _record_error(errors, sample_errors, "position_leg_role_requires_group_id",
                                      f"row={row_number} has position_leg_role={position_leg_role} but blank position_group_id", max_error_samples)
                    position_group_rows.append({
                        "row_number": row_number,
                        "position_group_id": position_group_id,
                        "position_leg_role": position_leg_role,
                        "option_type": str(row.get("option_type", "") or "").strip().lower(),
                        "size": _to_int(row.get("size")),
                    })

        if dataset_name == "position_snapshot":
            _validate_position_group_invariants(position_group_rows, errors, sample_errors, max_error_samples)

    summary.errors_by_code = dict(errors)
    summary.sample_errors = sample_errors
    return summary


def validate_chain_snapshot_csv(csv_path, schema=None, max_error_samples=30) -> ValidationSummary:
    return validate_dataset_csv(csv_path, "chain_snapshot", schema, max_error_samples)


def validate_context_snapshot_csv(csv_path, schema=None, max_error_samples=30) -> ValidationSummary:
    return validate_dataset_csv(csv_path, "context_snapshot", schema, max_error_samples)


def validate_position_snapshot_csv(csv_path, schema=None, max_error_samples=30) -> ValidationSummary:
    return validate_dataset_csv(csv_path, "position_snapshot", schema, max_error_samples)


def evaluate_liquidity_snapshot(
    csv_path: str | Path,
    symbols: set[str] | None = None,
    dte_min: int = 20,
    dte_max: int = 45,
    call_delta_min: float = 0.20,
    call_delta_max: float = 0.35,
    put_abs_delta_min: float = 0.20,
    put_abs_delta_max: float = 0.30,
    thresholds: LiquidityGateThresholds | None = None,
) -> dict[str, SnapshotLiquidityFlags]:
    """Flag covered-call / bull-put-spread candidacy and gate-pass per symbol in
    a single chain snapshot."""
    if thresholds is None:
        thresholds = get_liquidity_thresholds(load_contract_schema())

    path = Path(csv_path)
    symbol_filter = {s.upper().strip() for s in symbols} if symbols else None
    flags_by_symbol: dict[str, SnapshotLiquidityFlags] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            symbol = (row.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            if symbol_filter is not None and symbol not in symbol_filter:
                continue

            flags = flags_by_symbol.setdefault(symbol, SnapshotLiquidityFlags())
            flags.seen = True

            dte = _to_int(row.get("dte"))
            delta = _to_float(row.get("delta"))
            bid = _to_float(row.get("bid"))
            ask = _to_float(row.get("ask"))
            oi = _to_int(row.get("open_interest"))
            volume = _to_int(row.get("volume"))
            option_type = (row.get("option_type") or "").strip().lower()

            if dte is None or delta is None or bid is None or ask is None:
                continue
            if dte < dte_min or dte > dte_max:
                continue

            spread_pct = _spread_pct_of_mid(bid=bid, ask=ask)
            if spread_pct is None:
                continue

            gate_pass = (
                oi is not None and oi >= thresholds.open_interest_min
                and volume is not None and volume >= thresholds.volume_min
                and spread_pct <= thresholds.max_spread_pct_of_mid
            )

            if option_type == "call" and call_delta_min <= delta <= call_delta_max:
                flags.cc_candidate = True
                if flags.best_cc_spread_pct is None or spread_pct < flags.best_cc_spread_pct:
                    flags.best_cc_spread_pct = spread_pct
                if gate_pass:
                    flags.cc_pass = True
            elif option_type == "put" and (-put_abs_delta_max <= delta <= -put_abs_delta_min):
                flags.bps_candidate = True
                if flags.best_bps_spread_pct is None or spread_pct < flags.best_bps_spread_pct:
                    flags.best_bps_spread_pct = spread_pct
                if gate_pass:
                    flags.bps_pass = True

    if symbol_filter is not None:
        for symbol in symbol_filter:
            flags_by_symbol.setdefault(symbol, SnapshotLiquidityFlags())

    return flags_by_symbol


def update_liquidity_aggregate(
    aggregate: dict[str, SymbolLiquidityAggregate],
    snapshot_flags: dict[str, SnapshotLiquidityFlags],
) -> None:
    """Fold one snapshot's flags into a running per-symbol aggregate."""
    for symbol, flags in snapshot_flags.items():
        entry = aggregate.setdefault(symbol, SymbolLiquidityAggregate())
        if flags.seen:
            entry.snapshots_seen += 1
        if flags.cc_candidate:
            entry.cc_candidate_snapshots += 1
            if flags.best_cc_spread_pct is not None:
                entry.cc_spread_samples.append(flags.best_cc_spread_pct)
        if flags.cc_pass:
            entry.cc_pass_snapshots += 1
        if flags.bps_candidate:
            entry.bps_candidate_snapshots += 1
            if flags.best_bps_spread_pct is not None:
                entry.bps_spread_samples.append(flags.best_bps_spread_pct)
        if flags.bps_pass:
            entry.bps_pass_snapshots += 1
        if flags.cc_pass and flags.bps_pass:
            entry.both_pass_snapshots += 1


def summarize_symbol_liquidity(
    aggregate: dict[str, SymbolLiquidityAggregate],
    total_snapshots_considered: int,
) -> list[dict[str, Any]]:
    """Rank symbols by a blended liquidity-opportunity score."""
    rows: list[dict[str, Any]] = []
    for symbol, entry in aggregate.items():
        cc_pass_rate = _safe_div(entry.cc_pass_snapshots, entry.cc_candidate_snapshots)
        bps_pass_rate = _safe_div(entry.bps_pass_snapshots, entry.bps_candidate_snapshots)
        both_pass_rate_seen = _safe_div(entry.both_pass_snapshots, entry.snapshots_seen)
        cc_candidate_coverage = _safe_div(entry.cc_candidate_snapshots, entry.snapshots_seen)
        bps_candidate_coverage = _safe_div(entry.bps_candidate_snapshots, entry.snapshots_seen)
        snapshot_coverage = _safe_div(entry.snapshots_seen, total_snapshots_considered)

        composite_score = (
            0.40 * both_pass_rate_seen
            + 0.25 * cc_pass_rate
            + 0.25 * bps_pass_rate
            + 0.10 * min(cc_candidate_coverage, bps_candidate_coverage)
        )

        rows.append({
            "symbol": symbol,
            "snapshots_seen": entry.snapshots_seen,
            "total_snapshots_considered": total_snapshots_considered,
            "snapshot_coverage": round(snapshot_coverage, 6),
            "cc_candidate_snapshots": entry.cc_candidate_snapshots,
            "cc_pass_snapshots": entry.cc_pass_snapshots,
            "cc_candidate_coverage": round(cc_candidate_coverage, 6),
            "cc_pass_rate_given_candidate": round(cc_pass_rate, 6),
            "bps_candidate_snapshots": entry.bps_candidate_snapshots,
            "bps_pass_snapshots": entry.bps_pass_snapshots,
            "bps_candidate_coverage": round(bps_candidate_coverage, 6),
            "bps_pass_rate_given_candidate": round(bps_pass_rate, 6),
            "both_pass_snapshots": entry.both_pass_snapshots,
            "both_pass_rate_by_seen": round(both_pass_rate_seen, 6),
            "median_cc_best_spread_pct": (
                round(float(median(entry.cc_spread_samples)), 6) if entry.cc_spread_samples else None
            ),
            "median_bps_best_spread_pct": (
                round(float(median(entry.bps_spread_samples)), 6) if entry.bps_spread_samples else None
            ),
            "composite_score": round(composite_score, 6),
        })

    return sorted(
        rows,
        key=lambda item: (item["composite_score"], item["both_pass_rate_by_seen"], item["snapshots_seen"]),
        reverse=True,
    )

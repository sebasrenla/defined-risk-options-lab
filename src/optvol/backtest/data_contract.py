"""Versioned data contract loader.

A backtest is only as trustworthy as its inputs, so the datasets that feed it
(option-chain snapshots, per-symbol context, and position snapshots) are
governed by an explicit, versioned **data contract**: a JSON document that
declares each dataset's required columns, nullability, numeric/integer typing,
enum constraints, and value bounds. Validation is driven entirely by that
contract (see :mod:`optvol.backtest.validators`), so tightening a rule is a data
change, not a code change.

A self-contained example contract ships alongside this module
(``example_data_contract.json``); point ``load_contract_schema`` at your own
file to use a different one.

Provenance
----------
Ported from the program's ``schema_v1.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEFAULT_SCHEMA_PATH = Path(__file__).resolve().parent / "example_data_contract.json"


@dataclass(frozen=True)
class LiquidityGateThresholds:
    """Minimum tradability thresholds for a short option leg."""

    open_interest_min: int
    volume_min: int
    max_spread_pct_of_mid: float


def load_contract_schema(schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    """Load and parse a data-contract JSON document."""
    path = Path(schema_path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_dataset_schema(schema: Mapping[str, Any], dataset_name: str) -> dict[str, Any]:
    """Return one dataset's sub-schema, or raise ``KeyError`` if unknown."""
    datasets = schema.get("datasets", {})
    if dataset_name not in datasets:
        raise KeyError(f"Unknown dataset '{dataset_name}' in schema")
    return datasets[dataset_name]


def get_liquidity_thresholds(schema: Mapping[str, Any]) -> LiquidityGateThresholds:
    """Extract the short-leg liquidity thresholds from a contract."""
    payload = schema["liquidity_gates"]
    return LiquidityGateThresholds(
        open_interest_min=int(payload["short_leg_open_interest_min"]),
        volume_min=int(payload["short_leg_volume_min"]),
        max_spread_pct_of_mid=float(payload["short_leg_max_spread_pct_of_mid"]),
    )

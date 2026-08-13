"""Backtest support: versioned data contract and dataset validation."""

from .data_contract import (
    DEFAULT_SCHEMA_PATH,
    LiquidityGateThresholds,
    get_dataset_schema,
    get_liquidity_thresholds,
    load_contract_schema,
)
from .validators import (
    ValidationSummary,
    evaluate_liquidity_snapshot,
    normalize_corporate_action_flag,
    summarize_symbol_liquidity,
    update_liquidity_aggregate,
    validate_chain_snapshot_csv,
    validate_context_snapshot_csv,
    validate_dataset_csv,
    validate_position_snapshot_csv,
)

__all__ = [
    "DEFAULT_SCHEMA_PATH",
    "LiquidityGateThresholds",
    "load_contract_schema",
    "get_dataset_schema",
    "get_liquidity_thresholds",
    "ValidationSummary",
    "normalize_corporate_action_flag",
    "validate_dataset_csv",
    "validate_chain_snapshot_csv",
    "validate_context_snapshot_csv",
    "validate_position_snapshot_csv",
    "evaluate_liquidity_snapshot",
    "update_liquidity_aggregate",
    "summarize_symbol_liquidity",
]

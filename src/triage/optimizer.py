"""Optimizer: greedy budget allocation by expected loss."""

from dataclasses import dataclass
from typing import Any

import polars as pl

from triage.calibration import Calibrator
from triage.costs import CostModel


@dataclass(frozen=True)
class Policy:
    """Routing policy: per-segment p_error thresholds and expected outcomes."""

    budget: dict[str, Any]
    config_snapshot: dict[str, Any]
    created_at: str
    input_sha256s: dict[str, str]
    policy_hash: str
    segments: dict[str, dict[str, Any]]
    tool_version: str
    totals: dict[str, Any]


def plan(
    batch: pl.DataFrame,
    calibrator: Calibrator,
    cost_model: CostModel,
    *,
    input_sha256s: dict[str, str],
    tool_version: str,
) -> Policy:
    """Plan a routing policy for the batch given calibrator and cost model.

    Computes:
    1. p_error per item via calibrator.predict(score)
    2. expected_loss = p_error * false_accept_cost(segment)
    3. net_benefit = expected_loss - review_cost(segment)
    4. Sort by net_benefit DESC, tie-break item_id ASC
    5. Allocate budget greedily while net_benefit > 0
    6. Per-segment threshold = min p_error of routed items
    7. Compute expected_captured_loss and expected_residual_loss
    """
    raise NotImplementedError


def check_residual_gate(policy: Policy, cost_model: CostModel) -> None:
    """Check if policy's residual risk exceeds gate; raise ResidualRiskExceeded if so."""
    raise NotImplementedError

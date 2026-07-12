"""Optimizer: greedy budget allocation by expected loss."""

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import polars as pl

from triage import contract
from triage.artifacts import content_hash
from triage.costs import CostModel
from triage.exceptions import ResidualRiskExceeded


class PredictsErrors(Protocol):
    """Structural type for anything that maps raw scores to calibrated p_error.

    The real `triage.calibration.Calibrator` satisfies this. Tests may pass in
    a lightweight stub instead, since calibration.py is developed in parallel.
    """

    def predict(self, scores: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Policy:
    """Routing policy: per-segment p_error thresholds and expected outcomes."""

    budget: dict[str, Any]
    config_snapshot: dict[str, Any]
    input_sha256s: dict[str, str]
    tool_version: str
    segments: dict[str, dict[str, Any]]
    totals: dict[str, Any]
    policy_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return the §7 policy.json dict, without `created_at` (CLI adds it)."""
        return {
            "budget": self.budget,
            "config_snapshot": self.config_snapshot,
            "input_sha256s": self.input_sha256s,
            "policy_hash": self.policy_hash,
            "segments": self.segments,
            "tool_version": self.tool_version,
            "totals": self.totals,
        }


def plan(
    batch: pl.DataFrame,
    calibrator: PredictsErrors,
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
    validated = contract.validate(batch, require_truth=False)

    item_ids = validated["item_id"].to_list()
    segments = validated["segment"].to_list()
    scores = validated["score"].to_numpy()

    p_error = np.asarray(calibrator.predict(scores), dtype=np.float64)

    false_accept_costs = np.empty(len(item_ids), dtype=np.float64)
    review_costs = np.empty(len(item_ids), dtype=np.float64)
    for i, seg in enumerate(segments):
        seg_costs = cost_model.costs_for(seg)
        false_accept_costs[i] = seg_costs.false_accept_cost
        review_costs[i] = seg_costs.review_cost

    expected_loss = p_error * false_accept_costs
    net_benefit = expected_loss - review_costs

    # Single stable sort: net_benefit DESC, tie-break item_id ASC.
    # Achieved with two stable sorts: first ascending by item_id, then
    # descending by net_benefit. Python's sort is stable, so ties on
    # net_benefit retain the item_id-ascending order from the first pass.
    order = sorted(range(len(item_ids)), key=lambda i: item_ids[i])
    order.sort(key=lambda i: -net_benefit[i])

    n_batch = len(item_ids)
    budget = cost_model.budget
    n_capacity: int
    if budget.type == "fraction":
        n_capacity = math.floor(budget.value * n_batch)
    elif budget.type == "count":
        n_capacity = min(int(budget.value), n_batch)
    elif budget.type == "money":
        # Walk the sorted list, admit while cumulative review_cost <= value.
        # Pinned rule: stop at the FIRST item that would exceed the remaining
        # money budget (do not skip-and-continue past it).
        n_capacity = 0
        cumulative = 0.0
        for i in order:
            candidate = cumulative + review_costs[i]
            if candidate <= budget.value:
                cumulative = candidate
                n_capacity += 1
            else:
                break
    else:  # pragma: no cover - pydantic Literal guards this
        raise ValueError(f"Unknown budget type: {budget.type}")

    routed = np.zeros(n_batch, dtype=bool)
    n_routed = 0
    for i in order:
        if n_routed >= n_capacity:
            break
        if net_benefit[i] <= 0:
            break
        routed[i] = True
        n_routed += 1

    all_segments = sorted(set(segments))
    segment_stats: dict[str, dict[str, Any]] = {}
    for seg in all_segments:
        seg_indices = [i for i in range(n_batch) if segments[i] == seg]
        seg_routed = [i for i in seg_indices if routed[i]]
        n_review = len(seg_routed)
        n_total = len(seg_indices)
        captured = math.fsum(float(expected_loss[i]) for i in seg_routed)
        residual = math.fsum(float(expected_loss[i]) for i in seg_indices if not routed[i])
        threshold = min((float(p_error[i]) for i in seg_routed), default=None)
        segment_stats[seg] = {
            "expected_captured_loss": captured,
            "expected_residual_loss": residual,
            "n_review": n_review,
            "n_total": n_total,
            "p_error_threshold": threshold,
        }

    # Totals are defined as the plain sequential sum of the per-segment values
    # in sorted-segment order (not an independent fsum over the interleaved
    # batch): the accounting table in policy.json must tie out exactly, and
    # separately-computed float sums differ at ULP level.
    total_captured = sum(s["expected_captured_loss"] for s in segment_stats.values())
    total_residual = sum(s["expected_residual_loss"] for s in segment_stats.values())
    total_review_spend = math.fsum(float(review_costs[i]) for i in range(n_batch) if routed[i])
    totals = {
        "expected_captured_loss": total_captured,
        "expected_residual_loss": total_residual,
        "n_review": int(n_routed),
        "n_total": int(n_batch),
        "review_spend": total_review_spend,
    }

    budget_dict = {"type": budget.type, "value": float(budget.value)}
    config_snapshot = cost_model.model_dump()

    policy_dict: dict[str, Any] = {
        "budget": budget_dict,
        "config_snapshot": config_snapshot,
        "input_sha256s": input_sha256s,
        "segments": segment_stats,
        "tool_version": tool_version,
        "totals": totals,
    }
    policy_hash = content_hash(policy_dict)

    return Policy(
        budget=budget_dict,
        config_snapshot=config_snapshot,
        input_sha256s=input_sha256s,
        tool_version=tool_version,
        segments=segment_stats,
        totals=totals,
        policy_hash=policy_hash,
    )


def check_residual_gate(policy: Policy, cost_model: CostModel) -> None:
    """Check if policy's residual risk exceeds gate; raise ResidualRiskExceeded if so."""
    max_residual_risk = cost_model.gates.max_residual_risk
    if max_residual_risk is None:
        return
    residual = policy.totals["expected_residual_loss"]
    if residual > max_residual_risk:
        raise ResidualRiskExceeded(
            f"Expected residual loss {residual} exceeds gates.max_residual_risk {max_residual_risk}"
        )

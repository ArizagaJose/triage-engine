"""Optimizer: greedy budget allocation by expected net benefit."""

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import polars as pl

from triage import contract
from triage.artifacts import content_hash
from triage.costs import DEFAULT_BUDGET_STRATEGY, CostModel
from triage.exceptions import ResidualRiskExceeded


class PredictsErrors(Protocol):
    """Anything that maps raw scores to calibrated p_error.

    `triage.calibration.Calibrator` satisfies it; tests may pass a stub.
    """

    def predict(self, scores: np.ndarray) -> np.ndarray: ...


# --- review queue ---------------------------------------------------------------
# The ordered, per-item review list the optimizer actually selected. policy.json
# publishes per-segment thresholds; re-deriving the review set from those
# thresholds is NOT equivalent, because items tied exactly AT a threshold are
# admitted by the threshold but may be cut by capacity in the ranking (tie
# inflation, docs/interfaces.md §9.4: 1183 vs 1000 planned at seed 42,
# overconfident). The queue therefore carries the optimizer's own selection.
#
# Column order and dtypes are pinned: they are the exported CSV/Parquet schema.
REVIEW_QUEUE_SCHEMA: dict[str, pl.DataType] = {
    # 1-based position in the optimizer's ranking (net_benefit DESC, item_id
    # ASC). Priorities enumerate the selected subset in this ranking; fill may
    # skip non-fitting items. Priorities stay contiguous 1..n_review.
    "review_priority": pl.Int64(),
    "item_id": pl.String(),
    "segment": pl.String(),
    "p_error": pl.Float64(),
    "false_accept_cost": pl.Float64(),
    "expected_loss": pl.Float64(),
    "review_cost": pl.Float64(),
    "net_benefit": pl.Float64(),
    # Traceability, repeated per row so a queue stays linked to its policy and
    # source batch even when rows are copied into a review tool.
    "policy_hash": pl.String(),
    "batch_sha256": pl.String(),
}

REVIEW_QUEUE_COLUMNS: tuple[str, ...] = tuple(REVIEW_QUEUE_SCHEMA)


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
    # "fill" strategy only; None keeps the key out of default policy.json.
    allocation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the §7 policy.json dict, without `created_at` (CLI adds it)."""
        out: dict[str, Any] = {
            "budget": self.budget,
            "config_snapshot": self.config_snapshot,
            "input_sha256s": self.input_sha256s,
            "policy_hash": self.policy_hash,
            "segments": self.segments,
            "tool_version": self.tool_version,
            "totals": self.totals,
        }
        if self.allocation is not None:
            out["allocation"] = self.allocation
        return out


# "fill" candidate names, written to policy.json as
# `allocation.selected_candidate` (artifact contract).
CANDIDATE_NET_BENEFIT = "net_benefit"
CANDIDATE_NET_BENEFIT_PER_COST = "net_benefit_per_cost"


def _fill_walk(order: list[int], review_costs: np.ndarray, budget_value: float) -> np.ndarray:
    """Admit each item of `order` whose review_cost still fits; skip the rest.

    Accumulates spend exactly like the prefix walk, so over the same ranking
    the result is a superset of it. Items of one segment share a review_cost
    and spend never decreases, so each segment's selection stays a prefix of
    the ranking and `p_error_threshold` replays it exactly.
    """
    routed = np.zeros(len(review_costs), dtype=bool)
    cumulative = 0.0
    for i in order:
        candidate = cumulative + review_costs[i]
        if candidate <= budget_value:
            cumulative = candidate
            routed[i] = True
    return routed


def _allocate_money_fill(
    order: list[int],
    net_benefit: np.ndarray,
    review_costs: np.ndarray,
    budget_value: float,
) -> tuple[np.ndarray, str]:
    """Opt-in "fill" allocation of a money budget; returns (routed, candidate).

    Walks the eligible items (net_benefit > 0) of two rankings with
    `_fill_walk`: the default `order` ("net_benefit") and net_benefit per unit
    of review_cost ("net_benefit_per_cost"). Keeps the larger total expected
    net benefit; an exact tie keeps "net_benefit". Guarantees (never worse
    than "prefix", at least half of the optimum): docs/interfaces.md §5.
    """
    eligible = [i for i in order if net_benefit[i] > 0]

    def per_cost(i: int) -> float:
        cost = float(review_costs[i])
        return math.inf if cost == 0.0 else float(net_benefit[i]) / cost

    # Stable over the default ranking, so equal ratios keep its order and each
    # segment (constant review_cost) is ordered exactly as by default.
    per_cost_order = list(eligible)
    per_cost_order.sort(key=lambda i: -per_cost(i))

    by_net_benefit = _fill_walk(eligible, review_costs, budget_value)
    by_per_cost = _fill_walk(per_cost_order, review_costs, budget_value)

    def total_net_benefit(routed: np.ndarray) -> float:
        return math.fsum(float(net_benefit[i]) for i in np.flatnonzero(routed))

    if total_net_benefit(by_per_cost) > total_net_benefit(by_net_benefit):
        return by_per_cost, CANDIDATE_NET_BENEFIT_PER_COST
    return by_net_benefit, CANDIDATE_NET_BENEFIT


@dataclass(frozen=True)
class PlanResult:
    """One allocation, two views of it: the published `policy` and the
    per-item `review_queue` that produced its numbers.

    Both come from the SAME `plan_with_queue` call, so the queue is the
    optimizer's actual selection, not a reconstruction from the policy's
    thresholds (which ties make inexact, see REVIEW_QUEUE_SCHEMA).
    """

    policy: Policy
    review_queue: pl.DataFrame


def plan(
    batch: pl.DataFrame,
    calibrator: PredictsErrors,
    cost_model: CostModel,
    *,
    input_sha256s: dict[str, str],
    tool_version: str,
) -> Policy:
    """Plan a routing policy for the batch given calibrator and cost model.

    Thin wrapper over `plan_with_queue` that keeps only the policy; identical
    inputs therefore give an identical `policy_hash` whether or not a caller
    asks for the review queue.
    """
    return plan_with_queue(
        batch,
        calibrator,
        cost_model,
        input_sha256s=input_sha256s,
        tool_version=tool_version,
    ).policy


def plan_with_queue(
    batch: pl.DataFrame,
    calibrator: PredictsErrors,
    cost_model: CostModel,
    *,
    input_sha256s: dict[str, str],
    tool_version: str,
) -> PlanResult:
    """Plan a routing policy AND return the review queue behind it.

    Computes:
    1. p_error per item via calibrator.predict(score)
    2. expected_loss = p_error * false_accept_cost(segment)
    3. net_benefit = expected_loss - review_cost(segment)
    4. Sort by net_benefit DESC, tie-break item_id ASC
    5. Allocate budget greedily while net_benefit > 0 (money budgets with the
       opt-in `budget.strategy: fill` use `_allocate_money_fill` instead)
    6. Per-segment threshold = min p_error of routed items
    7. Compute expected_captured_loss and expected_residual_loss
    8. Emit the routed items, in ranking order, as the review queue
       (REVIEW_QUEUE_SCHEMA)
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

    # net_benefit DESC, ties by item_id ASC: sort by item_id first, then a
    # stable sort by net_benefit keeps that order within ties.
    order = sorted(range(len(item_ids)), key=lambda i: item_ids[i])
    order.sort(key=lambda i: -net_benefit[i])

    n_batch = len(item_ids)
    budget = cost_model.budget
    allocation: dict[str, Any] | None = None
    if budget.type == "money" and budget.strategy == "fill":
        routed, selected_candidate = _allocate_money_fill(
            order, net_benefit, review_costs, float(budget.value)
        )
        n_routed = int(routed.sum())
        allocation = {"selected_candidate": selected_candidate}
    else:
        n_capacity: int
        if budget.type == "fraction":
            n_capacity = math.floor(budget.value * n_batch)
        elif budget.type == "count":
            n_capacity = min(int(budget.value), n_batch)
        elif budget.type == "money":
            # Default "prefix" strategy (pinned): admit while cumulative
            # review_cost <= value and stop at the FIRST item that does not fit.
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

    budget_dict: dict[str, Any] = {"type": budget.type, "value": float(budget.value)}
    if budget.strategy != DEFAULT_BUDGET_STRATEGY:  # default omitted: hashes unchanged
        budget_dict["strategy"] = budget.strategy
    config_snapshot = cost_model.model_dump()

    policy_dict: dict[str, Any] = {
        "budget": budget_dict,
        "config_snapshot": config_snapshot,
        "input_sha256s": input_sha256s,
        "segments": segment_stats,
        "tool_version": tool_version,
        "totals": totals,
    }
    if allocation is not None:
        policy_dict["allocation"] = allocation
    policy_hash = content_hash(policy_dict)

    policy = Policy(
        budget=budget_dict,
        config_snapshot=config_snapshot,
        input_sha256s=input_sha256s,
        tool_version=tool_version,
        segments=segment_stats,
        totals=totals,
        policy_hash=policy_hash,
        allocation=allocation,
    )

    # The queue is built from `routed`/`order` -- the allocation above -- so it
    # is the selection the policy's numbers were computed from, by
    # construction. The selected subset is ordered by `order`; priorities enumerate it
    # as 1..n_routed without gaps, including when fill skips items.
    queue_indices = [i for i in order if routed[i]]
    review_queue = _review_queue_frame(
        queue_indices,
        item_ids=item_ids,
        segments=segments,
        p_error=p_error,
        false_accept_costs=false_accept_costs,
        expected_loss=expected_loss,
        review_costs=review_costs,
        net_benefit=net_benefit,
        policy_hash=policy_hash,
        batch_sha256=input_sha256s.get("batch", ""),
    )

    return PlanResult(policy=policy, review_queue=review_queue)


def _review_queue_frame(
    queue_indices: list[int],
    *,
    item_ids: list[str],
    segments: list[str],
    p_error: np.ndarray,
    false_accept_costs: np.ndarray,
    expected_loss: np.ndarray,
    review_costs: np.ndarray,
    net_benefit: np.ndarray,
    policy_hash: str,
    batch_sha256: str,
) -> pl.DataFrame:
    """Build the REVIEW_QUEUE_SCHEMA frame for `queue_indices` (batch row
    indices in review-priority order).

    Floats are passed through unrounded: the queue must reconcile with
    policy.json exactly, not approximately. An empty selection yields a
    zero-row frame with the same columns and dtypes, so an exported file is
    always readable and self-describing.
    """
    n = len(queue_indices)
    columns: dict[str, list[Any]] = {
        "review_priority": list(range(1, n + 1)),
        "item_id": [item_ids[i] for i in queue_indices],
        "segment": [segments[i] for i in queue_indices],
        "p_error": [float(p_error[i]) for i in queue_indices],
        "false_accept_cost": [float(false_accept_costs[i]) for i in queue_indices],
        "expected_loss": [float(expected_loss[i]) for i in queue_indices],
        "review_cost": [float(review_costs[i]) for i in queue_indices],
        "net_benefit": [float(net_benefit[i]) for i in queue_indices],
        "policy_hash": [policy_hash] * n,
        "batch_sha256": [batch_sha256] * n,
    }
    return pl.DataFrame(columns, schema=REVIEW_QUEUE_SCHEMA)


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

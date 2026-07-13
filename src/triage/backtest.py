"""Backtester: replay a policy against labeled history vs random/raw-confidence baselines.

All "loss" quantities computed here are REALIZED (ground-truth) losses:
``true_loss(item) = (predicted_value != true_value) * false_accept_cost(segment)`` —
not the calibrated expectation the optimizer used to plan the policy.

Accounting convention (see docs/interfaces.md §9, point 9): for the "policy"
strategy the per-segment `captured_loss`/`residual_loss` values are independent
`math.fsum`s, and the strategy-level `captured_loss` is the sequential sum of
those per-segment values in sorted-segment order (matching the optimizer's
tie-out convention). The strategy-level `residual_loss`, however, is NOT the
sequential sum of per-segment residuals — it is defined as
`total_true_loss - captured_loss`, because two independently-computed fsums
are not guaranteed to satisfy `captured + residual == total` at ULP level.
The same `total_true_loss - captured_loss` definition is used for
`raw_confidence` and `random` (mean-based), which have no per-segment table.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from pydantic import ValidationError

from triage import contract
from triage.artifacts import content_hash
from triage.costs import CostModel
from triage.exceptions import ContractViolation, ResidualRiskExceeded
from triage.optimizer import PredictsErrors

__all__ = ["BacktestResult", "run", "check_realized_residual_gate"]


def _require(d: dict[str, Any], key: str, *, context: str) -> Any:
    """Fetch `key` from dict `d`, raising ContractViolation (never KeyError) if absent."""
    if key not in d:
        raise ContractViolation(f"{context}: missing required key {key!r}")
    return d[key]


@dataclass(frozen=True)
class BacktestResult:
    """Backtest outcome: policy replay vs baselines. Mirrors backtest.json minus created_at."""

    backtest_hash: str
    expected: dict[str, Any]
    input_sha256s: dict[str, str]
    n_errors: int
    n_history: int
    n_random_draws: int
    seed: int
    strategies: dict[str, Any]
    tool_version: str
    total_true_loss: float

    def to_dict(self) -> dict[str, Any]:
        """Return the §7 backtest.json dict, without `created_at` (CLI adds it)."""
        return {
            "backtest_hash": self.backtest_hash,
            "expected": self.expected,
            "input_sha256s": self.input_sha256s,
            "n_errors": self.n_errors,
            "n_history": self.n_history,
            "n_random_draws": self.n_random_draws,
            "seed": self.seed,
            "strategies": self.strategies,
            "tool_version": self.tool_version,
            "total_true_loss": self.total_true_loss,
        }


def run(
    policy_dict: dict[str, Any],
    calibrator: PredictsErrors,
    history: pl.DataFrame,
    *,
    seed: int,
    n_random_draws: int = 200,
    input_sha256s: dict[str, str],
    tool_version: str,
) -> BacktestResult:
    """Replay `policy_dict` on labeled `history` and compare it to two baselines.

    See docs/interfaces.md §9 for the pinned semantics.
    """
    # 1. Policy self-check: the artifact must hash to its own declared policy_hash.
    declared_hash = _require(policy_dict, "policy_hash", context="policy")
    computed_hash = content_hash(policy_dict)
    if computed_hash != declared_hash:
        raise ContractViolation(
            f"Policy self-hash mismatch: declared policy_hash={declared_hash!r}, "
            f"computed content_hash={computed_hash!r}"
        )

    # 2. Cost model from the policy's embedded config snapshot.
    config_snapshot = _require(policy_dict, "config_snapshot", context="policy")
    if not isinstance(config_snapshot, dict):
        raise ContractViolation("policy.config_snapshot must be an object")
    try:
        cost_model = CostModel.model_validate(config_snapshot)
    except ValidationError as e:
        raise ContractViolation(f"policy.config_snapshot failed validation: {e}") from e

    policy_segments = _require(policy_dict, "segments", context="policy")
    if not isinstance(policy_segments, dict):
        raise ContractViolation("policy.segments must be an object")
    policy_totals = _require(policy_dict, "totals", context="policy")
    if not isinstance(policy_totals, dict):
        raise ContractViolation("policy.totals must be an object")
    for key in ("expected_captured_loss", "expected_residual_loss"):
        if key not in policy_totals:
            raise ContractViolation(f"policy.totals: missing required key {key!r}")
    expected = {
        "expected_captured_loss": policy_totals["expected_captured_loss"],
        "expected_residual_loss": policy_totals["expected_residual_loss"],
    }

    # 3. Validate history (labels required — that is the point of a backtest).
    validated = contract.validate(history, require_truth=True)

    item_ids = validated["item_id"].to_list()
    segments = validated["segment"].to_list()
    scores = validated["score"].to_numpy()
    predicted = validated["predicted_value"].to_list()
    truth = validated["true_value"].to_list()

    n_history = len(item_ids)

    is_error = np.array(
        [1.0 if pv != tv else 0.0 for pv, tv in zip(predicted, truth)], dtype=np.float64
    )

    false_accept_costs = np.empty(n_history, dtype=np.float64)
    review_costs = np.empty(n_history, dtype=np.float64)
    for i, seg in enumerate(segments):
        seg_costs = cost_model.costs_for(seg)
        false_accept_costs[i] = seg_costs.false_accept_cost
        review_costs[i] = seg_costs.review_cost

    true_loss = is_error * false_accept_costs
    total_true_loss = math.fsum(true_loss.tolist())
    n_errors = int(is_error.sum())

    p_error = np.asarray(calibrator.predict(scores), dtype=np.float64)

    # 4. Policy strategy: threshold replay. Unseen segments (absent from
    # policy_dict["segments"]) are never reviewed, same as a null threshold.
    reviewed_policy = np.zeros(n_history, dtype=bool)
    threshold_used: dict[str, float | None] = {}
    for i, seg in enumerate(segments):
        seg_stats = policy_segments.get(seg)
        threshold = seg_stats.get("p_error_threshold") if isinstance(seg_stats, dict) else None
        if seg not in threshold_used:
            threshold_used[seg] = threshold
        if threshold is not None and p_error[i] >= threshold:
            reviewed_policy[i] = True

    n_review = int(reviewed_policy.sum())

    all_segments = sorted(set(segments))
    policy_seg_stats: dict[str, dict[str, Any]] = {}
    for seg in all_segments:
        seg_indices = [i for i in range(n_history) if segments[i] == seg]
        seg_reviewed = [i for i in seg_indices if reviewed_policy[i]]
        seg_not_reviewed = [i for i in seg_indices if not reviewed_policy[i]]
        captured = math.fsum(float(true_loss[i]) for i in seg_reviewed)
        residual = math.fsum(float(true_loss[i]) for i in seg_not_reviewed)
        policy_seg_stats[seg] = {
            "captured_loss": captured,
            "residual_loss": residual,
            "n_review": len(seg_reviewed),
            "n_total": len(seg_indices),
            "p_error_threshold": threshold_used.get(seg),
        }

    # Strategy-level captured_loss is the sequential sum of per-segment values
    # in sorted-segment order (tie-out convention); residual_loss is defined
    # as total_true_loss - captured_loss (see module docstring, point 9).
    policy_captured = sum(s["captured_loss"] for s in policy_seg_stats.values())
    policy_residual = total_true_loss - policy_captured
    policy_review_spend = math.fsum(
        float(review_costs[i]) for i in range(n_history) if reviewed_policy[i]
    )
    policy_efficiency = policy_captured / n_review if n_review > 0 else None
    policy_capture_rate = policy_captured / total_true_loss if total_true_loss != 0 else None

    strategies: dict[str, Any] = {
        "policy": {
            "captured_loss": policy_captured,
            "residual_loss": policy_residual,
            "n_review": n_review,
            "review_spend": policy_review_spend,
            "efficiency": policy_efficiency,
            "capture_rate": policy_capture_rate,
            "segments": policy_seg_stats,
        }
    }

    # 6. Raw-confidence baseline: rank by score ASC, tie-break item_id ASC
    # (two stable sorts, same idiom as the optimizer); review the first
    # n_review items. No calibration, no per-segment breakdown.
    order_by_id = sorted(range(n_history), key=lambda i: item_ids[i])
    order_by_id.sort(key=lambda i: scores[i])
    raw_reviewed = set(order_by_id[:n_review])

    raw_captured = math.fsum(float(true_loss[i]) for i in raw_reviewed)
    raw_residual = total_true_loss - raw_captured
    raw_review_spend = math.fsum(float(review_costs[i]) for i in raw_reviewed)
    raw_efficiency = raw_captured / n_review if n_review > 0 else None
    raw_capture_rate = raw_captured / total_true_loss if total_true_loss != 0 else None

    strategies["raw_confidence"] = {
        "captured_loss": raw_captured,
        "residual_loss": raw_residual,
        "n_review": n_review,
        "review_spend": raw_review_spend,
        "efficiency": raw_efficiency,
        "capture_rate": raw_capture_rate,
    }

    # 7. Random baseline: n_random_draws independent samples of size n_review,
    # without replacement, from a single rng seeded by `seed`.
    rng = np.random.default_rng(seed)
    captured_draws = np.empty(n_random_draws, dtype=np.float64)
    spend_draws = np.empty(n_random_draws, dtype=np.float64)
    for d in range(n_random_draws):
        draw = rng.choice(n_history, size=n_review, replace=False)
        captured_draws[d] = math.fsum(float(true_loss[i]) for i in draw)
        spend_draws[d] = math.fsum(float(review_costs[i]) for i in draw)

    captured_mean = float(np.mean(captured_draws)) if n_random_draws > 0 else 0.0
    captured_std = float(np.std(captured_draws, ddof=0)) if n_random_draws > 0 else 0.0
    spend_mean = float(np.mean(spend_draws)) if n_random_draws > 0 else 0.0
    residual_mean = total_true_loss - captured_mean
    random_efficiency = captured_mean / n_review if n_review > 0 else None
    random_capture_rate = captured_mean / total_true_loss if total_true_loss != 0 else None

    strategies["random"] = {
        "captured_loss_mean": captured_mean,
        "captured_loss_std": captured_std,
        "residual_loss_mean": residual_mean,
        "review_spend_mean": spend_mean,
        "n_review": n_review,
        "efficiency": random_efficiency,
        "capture_rate": random_capture_rate,
    }

    backtest_dict: dict[str, Any] = {
        "expected": expected,
        "input_sha256s": input_sha256s,
        "n_errors": n_errors,
        "n_history": n_history,
        "n_random_draws": n_random_draws,
        "seed": seed,
        "strategies": strategies,
        "tool_version": tool_version,
        "total_true_loss": total_true_loss,
    }
    backtest_hash = content_hash(backtest_dict)

    return BacktestResult(
        backtest_hash=backtest_hash,
        expected=expected,
        input_sha256s=input_sha256s,
        n_errors=n_errors,
        n_history=n_history,
        n_random_draws=n_random_draws,
        seed=seed,
        strategies=strategies,
        tool_version=tool_version,
        total_true_loss=total_true_loss,
    )


def check_realized_residual_gate(result: BacktestResult, cost_model: CostModel) -> None:
    """Raise ResidualRiskExceeded if the policy's realized residual loss exceeds the gate."""
    max_residual_risk = cost_model.gates.max_residual_risk
    if max_residual_risk is None:
        return
    residual = result.strategies["policy"]["residual_loss"]
    if residual > max_residual_risk:
        raise ResidualRiskExceeded(
            f"Realized residual loss {residual} exceeds gates.max_residual_risk {max_residual_risk}"
        )

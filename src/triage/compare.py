"""Policy comparison: replay a CURRENT and a CANDIDATE policy on the same
labeled history, under one explicit cost basis, and report the difference.

See docs/interfaces.md §14 for the pinned contract. The question this module
answers is practical: "if we switch to the candidate, how much more (or less)
loss do we catch, and what happens to the human review workload?"

Four rules make the answer trustworthy:

1. **Each policy is replayed with its own calibrator**, verified by hash.
   Thresholds live in calibrated-`p_error` space, so replaying a policy
   with any other calibrator is meaningless. The policy must hash to its
   own `policy_hash`, and the calibrator must hash to the
   `input_sha256s.calibrator` the policy recorded. Two policies may share
   one calibrator (e.g. a budget-only change).
2. **The same replay rule as `backtest`** (`backtest.replay_thresholds`,
   §9.4) and the same accounting (`backtest.policy_outcome`, §9.8), so a
   comparison under a policy's own costs reproduces that policy's
   backtest numbers exactly.
3. **One evaluation cost basis for both policies.** Realized loss is
   `error * false_accept_cost(segment)` and spend is `review_cost(segment)`.
   Comparing two policies priced with different costs compares different
   units, and the verdict can flip with the basis. So:
   - without `eval_costs`, each history segment's costs are resolved
     (exact key, else `default`) under BOTH policies' `config_snapshot`.
     They must agree exactly on every segment present in the history, or
     the comparison is rejected (`ContractViolation`, listing every
     conflicting or unpriceable segment). Segments absent from the history
     cannot affect the evaluation and are not compared. Budgets and gates
     may differ freely: they are policy choices, not prices.
   - with `eval_costs` (a cost model supplied explicitly), its
     `segments` price both policies. Its `budget` and `gates` are ignored
     (recorded in the artifact). Each policy is flagged when it was
     planned under costs that differ from the basis on any history segment.
4. **Verdict by dominance on (captured loss, review count, review spend).**
   `no_change` when no direction differs; `improvement` when the candidate
   is better in at least one and worse in none; `regression` the mirror
   image; `trade_off` otherwise. `total_cost = residual_loss +
   review_spend` (the cost model's own objective: the optimizer ranks by
   expected loss minus review cost, so both are in the same units) is
   always reported and tells whether a trade-off pays off under the basis.
   Directions compare correctly-rounded `math.fsum`s over the items where
   the two review sets DIFFER, so swapping equal-loss items can never
   produce a spurious +-1 ULP verdict.

5. **Uncertainty: paired bootstrap over history items** (§14.7). A
   verdict from ONE history can be noise (e.g. 3 fewer reviews out of
   ~1 180). Items are resampled with replacement (`rng.integers`, seeded,
   same idiom as the calibration CI); each item's review decision is fixed
   by the policies' thresholds, so only items where the two review sets
   differ move a delta, and each resampled delta is a correctly-rounded
   `math.fsum` over those items weighted by their resample counts.
   Percentile CIs at `ci_level`. A direction is `better`/`worse` only when
   its CI excludes 0, `same` when no item differs or the CI is exactly
   [0, 0] (the differing items never move that delta), else `inconclusive`;
   the same dominance rule on those directions gives `verdict.ci.label`
   (plus `inconclusive` when nothing is significant). The opt-in
   regression gate (exit 6) reads only the CI verdict.

Accounting identities (same history, same basis, so `total_true_loss` is
shared): `delta.residual_loss == -delta.captured_loss` up to float
rounding, and `delta.captured_loss == overlap.candidate_only.captured_loss
- overlap.current_only.captured_loss` up to float rounding.
"""

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import ValidationError

from triage import contract
from triage.artifacts import content_hash
from triage.backtest import Replay, policy_outcome, replay_thresholds
from triage.calibration import Calibrator
from triage.costs import CostModel
from triage.exceptions import CandidateRegression, ContractViolation

__all__ = [
    "DEFAULT_CI_LEVEL",
    "DEFAULT_N_BOOTSTRAP",
    "SIDES",
    "ComparisonResult",
    "check_regression_gate",
    "run",
    "validate_uncertainty_options",
]

Side = Literal["current", "candidate"]
SIDES: tuple[Side, Side] = ("current", "candidate")

# Per-side status of a history segment (§14.5). All three statuses other
# than "planned" mean the policy reviews nothing in that segment; the report
# says WHY.
SegmentStatus = Literal["planned", "no_threshold", "not_in_plan"]

Direction = Literal["better", "worse", "same"]
# A CI direction adds "inconclusive": the delta's CI contains 0 although the
# review sets differ.
CiDirection = Literal["better", "worse", "same", "inconclusive"]
VerdictLabel = Literal["no_change", "improvement", "regression", "trade_off"]
CiVerdictLabel = Literal["no_change", "improvement", "regression", "trade_off", "inconclusive"]

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_CI_LEVEL = 0.95

_DELTA_KEYS = ("captured_loss", "residual_loss", "n_review", "review_spend", "total_cost")


@dataclass(frozen=True)
class ComparisonResult:
    """Mirrors comparison.json minus `created_at` (the CLI adds it)."""

    basis: dict[str, Any]
    comparison_hash: str
    delta: dict[str, Any]
    history: dict[str, Any]
    input_sha256s: dict[str, str | None]
    overlap: dict[str, Any]
    policies: dict[str, Any]
    tool_version: str
    uncertainty: dict[str, Any] | None
    verdict: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": self.basis,
            "comparison_hash": self.comparison_hash,
            "delta": self.delta,
            "history": self.history,
            "input_sha256s": self.input_sha256s,
            "overlap": self.overlap,
            "policies": self.policies,
            "tool_version": self.tool_version,
            "uncertainty": self.uncertainty,
            "verdict": self.verdict,
        }


# ---------------------------------------------------------------------------
# 1. Artifact relationships
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _VerifiedSide:
    side: Side
    policy: dict[str, Any]
    policy_hash: str
    calibrator: Calibrator
    calibrator_hash: str
    cost_model: CostModel


def _check_policy(side: Side, policy: Any) -> tuple[str, CostModel]:
    """Structural checks + self-hash; returns (policy_hash, cost model)."""
    if not isinstance(policy, dict):
        raise ContractViolation(f"{side} policy: expected a JSON object")
    for key in ("policy_hash", "config_snapshot", "segments", "totals", "budget"):
        if key not in policy:
            raise ContractViolation(f"{side} policy: missing required key {key!r}")
    for key in ("config_snapshot", "segments", "totals", "budget"):
        if not isinstance(policy[key], dict):
            raise ContractViolation(f"{side} policy: {key} must be an object")
    for key in ("n_review", "n_total"):
        if key not in policy["totals"]:
            raise ContractViolation(f"{side} policy: totals is missing required key {key!r}")
    declared = policy["policy_hash"]
    computed = content_hash(policy)
    if computed != declared:
        raise ContractViolation(
            f"{side} policy: self-hash mismatch, declared policy_hash={declared!r} but "
            f"content_hash(policy)={computed!r} (the policy was edited after planning)"
        )
    try:
        cost_model = CostModel.model_validate(policy["config_snapshot"])
    except ValidationError as e:
        raise ContractViolation(f"{side} policy: config_snapshot failed validation: {e}") from e
    return computed, cost_model


def _load_calibrator(side: Side, calibrator: Any) -> Calibrator:
    if not isinstance(calibrator, dict):
        raise ContractViolation(f"{side} calibrator: expected a JSON object")
    try:
        return Calibrator.from_dict(calibrator)
    except KeyError as e:
        raise ContractViolation(f"{side} calibrator: missing required key {e}") from e
    except (TypeError, ValueError) as e:
        raise ContractViolation(f"{side} calibrator: malformed artifact: {e}") from e


def _verify_sides(
    policies: dict[Side, Any], calibrators: dict[Side, Any]
) -> dict[Side, _VerifiedSide]:
    """Every relationship is RECOMPUTED from content, never trusted as
    declared (same rule as report §11.2). Detects swapped calibrators and
    says so, instead of reporting two unrelated mismatches."""
    checked = {side: _check_policy(side, policies[side]) for side in SIDES}
    cal_hashes: dict[Side, str] = {}
    for side in SIDES:
        if not isinstance(calibrators[side], dict):
            raise ContractViolation(f"{side} calibrator: expected a JSON object")
        cal_hashes[side] = content_hash(calibrators[side])
    declared: dict[Side, Any] = {
        side: (policies[side].get("input_sha256s") or {}).get("calibrator") for side in SIDES
    }

    mismatched = [side for side in SIDES if cal_hashes[side] != declared[side]]
    if mismatched == list(SIDES) and (
        cal_hashes["current"] == declared["candidate"]
        and cal_hashes["candidate"] == declared["current"]
    ):
        raise ContractViolation(
            "Calibrators are swapped: the current calibrator belongs to the candidate policy "
            "and the candidate calibrator belongs to the current policy "
            f"(current policy expects {declared['current']!r}, candidate policy expects "
            f"{declared['candidate']!r})"
        )
    for side in mismatched:
        other: Side = "candidate" if side == "current" else "current"
        hint = (
            f"; it is the {other} policy's calibrator"
            if cal_hashes[side] == declared[other]
            else ""
        )
        raise ContractViolation(
            f"{side} calibrator mismatch: it hashes to {cal_hashes[side]!r}, but the {side} "
            f"policy was planned with calibrator {declared[side]!r}{hint}. Each policy must be "
            "replayed with the calibrator that produced its thresholds"
        )

    return {
        side: _VerifiedSide(
            side=side,
            policy=policies[side],
            policy_hash=checked[side][0],
            calibrator=_load_calibrator(side, calibrators[side]),
            calibrator_hash=cal_hashes[side],
            cost_model=checked[side][1],
        )
        for side in SIDES
    }


# ---------------------------------------------------------------------------
# 2. Evaluation cost basis
# ---------------------------------------------------------------------------


def _resolve(cost_model: CostModel, segment: str) -> dict[str, float] | None:
    """`CostModel.costs_for` semantics (exact key, else `default`), but
    returning None instead of raising for an unpriceable segment."""
    costs = cost_model.segments.get(segment) or cost_model.segments.get("default")
    if costs is None:
        return None
    return {"false_accept_cost": costs.false_accept_cost, "review_cost": costs.review_cost}


def _describe(costs: dict[str, float] | None) -> str:
    if costs is None:
        return "cannot price it (no entry and no 'default')"
    return f"false_accept_cost={costs['false_accept_cost']!r} review_cost={costs['review_cost']!r}"


def _resolve_basis(
    sides: dict[Side, _VerifiedSide],
    history_segments: list[str],
    eval_costs: CostModel | None,
) -> dict[str, Any]:
    own: dict[Side, dict[str, dict[str, float] | None]] = {
        side: {seg: _resolve(sides[side].cost_model, seg) for seg in history_segments}
        for side in SIDES
    }

    if eval_costs is None:
        conflicts = [
            f"  {seg}: current {_describe(own['current'][seg])}; "
            f"candidate {_describe(own['candidate'][seg])}"
            for seg in history_segments
            if own["current"][seg] is None
            or own["candidate"][seg] is None
            or own["current"][seg] != own["candidate"][seg]
        ]
        if conflicts:
            raise ContractViolation(
                "The two policies price the history differently, so their losses are not "
                "comparable. Conflicting segment costs (resolved: exact entry, else "
                "'default'):\n" + "\n".join(conflicts) + "\nSupply one explicit evaluation "
                "cost model for both policies (compare --eval-costs costs.yaml)."
            )
        basis_segments = {seg: own["current"][seg] for seg in history_segments}
        source = "policies"
        ignored: list[str] = []
    else:
        unpriced = [seg for seg in history_segments if _resolve(eval_costs, seg) is None]
        if unpriced:
            raise ContractViolation(
                "The evaluation cost model cannot price history segment(s) "
                f"{unpriced} (no entry and no 'default')"
            )
        basis_segments = {seg: _resolve(eval_costs, seg) for seg in history_segments}
        source = "eval_costs"
        ignored = ["budget", "gates"]

    differences: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        differences[side] = {
            seg: own[side][seg] for seg in history_segments if own[side][seg] != basis_segments[seg]
        }
    return {
        "source": source,
        "ignored_eval_costs_fields": ignored,
        "segments": basis_segments,
        "planned_under_basis": {side: not differences[side] for side in SIDES},
        # Per side: history segments whose OWN (planning) costs differ from the
        # basis -> the side's own resolved costs there (null = unpriceable).
        "differences": differences,
    }


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------


def _segment_status(policy_segments: dict[str, Any], segment: str) -> SegmentStatus:
    stats = policy_segments.get(segment)
    if not isinstance(stats, dict):
        return "not_in_plan"
    return "no_threshold" if stats.get("p_error_threshold") is None else "planned"


def _side_result(
    side: _VerifiedSide,
    replay: Replay,
    segments: list[str],
    history_segments: list[str],
    true_loss: np.ndarray,
    review_costs: np.ndarray,
    total_true_loss: float,
) -> dict[str, Any]:
    outcome = policy_outcome(
        replay, segments, true_loss, review_costs, total_true_loss, segment_review_spend=True
    )
    outcome["total_cost"] = outcome["residual_loss"] + outcome["review_spend"]
    policy_segments = side.policy["segments"]
    for seg, stats in outcome["segments"].items():
        stats["total_cost"] = stats["residual_loss"] + stats["review_spend"]
        stats["status"] = _segment_status(policy_segments, seg)
    totals = side.policy["totals"]
    return {
        "budget": side.policy["budget"],
        "calibrator_hash": side.calibrator_hash,
        "calibrator_method": side.calibrator.method,
        "outcome": outcome,
        "planned": {"n_review": totals["n_review"], "n_total": totals["n_total"]},
        "planned_segments_not_in_history": sorted(
            seg for seg in policy_segments if seg not in set(history_segments)
        ),
        "policy_hash": side.policy_hash,
        "policy_input_sha256s": side.policy.get("input_sha256s") or {},
    }


def _group(mask: np.ndarray, true_loss: np.ndarray, review_costs: np.ndarray) -> dict[str, Any]:
    return {
        "n_review": int(mask.sum()),
        "captured_loss": math.fsum(true_loss[mask].tolist()),
        "review_spend": math.fsum(review_costs[mask].tolist()),
    }


def _direction(candidate: float, current: float, *, higher_is_better: bool) -> Direction:
    if candidate == current:
        return "same"
    better = candidate > current if higher_is_better else candidate < current
    return "better" if better else "worse"


def _verdict(
    current: np.ndarray, candidate: np.ndarray, true_loss: np.ndarray, review_costs: np.ndarray
) -> dict[str, Any]:
    """Directions from correctly-rounded fsums over the DIFFERING items only
    (module docstring, rule 4)."""
    only_cur = current & ~candidate
    only_cand = candidate & ~current
    loss_cur, loss_cand = true_loss[only_cur].tolist(), true_loss[only_cand].tolist()
    spend_cur, spend_cand = review_costs[only_cur].tolist(), review_costs[only_cand].tolist()

    directions: dict[str, Direction] = {
        "captured_loss": _direction(
            math.fsum(loss_cand), math.fsum(loss_cur), higher_is_better=True
        ),
        "n_review": _direction(int(candidate.sum()), int(current.sum()), higher_is_better=False),
        "review_spend": _direction(
            math.fsum(spend_cand), math.fsum(spend_cur), higher_is_better=False
        ),
    }
    # total_cost = residual + spend; its delta is
    # (loss only current reviews + spend only candidate reviews)
    # - (loss only candidate reviews + spend only current reviews).
    total_cost: Direction = _direction(
        math.fsum(loss_cur + spend_cand), math.fsum(loss_cand + spend_cur), higher_is_better=False
    )

    values = set(directions.values())
    label: VerdictLabel
    if values == {"same"}:
        label = "no_change"
    elif "worse" not in values:
        label = "improvement"
    elif "better" not in values:
        label = "regression"
    else:
        label = "trade_off"
    return {"label": label, "directions": directions, "total_cost": total_cost}


# ---------------------------------------------------------------------------
# 4. Uncertainty (paired bootstrap) and the CI verdict
# ---------------------------------------------------------------------------


def validate_uncertainty_options(seed: int, n_bootstrap: int, ci_level: float) -> None:
    """Seed as everywhere else; `n_bootstrap` an int >= 0 (0 = no CI, no CI
    verdict); `ci_level` finite in (0, 1). ContractViolation otherwise."""
    contract.validate_seed(seed)
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, int) or n_bootstrap < 0:
        raise ContractViolation(f"n_bootstrap must be an integer >= 0, got {n_bootstrap!r}")
    if isinstance(ci_level, bool) or not isinstance(ci_level, (int, float)):
        raise ContractViolation(f"ci_level must be a number in (0, 1), got {ci_level!r}")
    # Positive form so NaN is rejected (every NaN comparison is False).
    if not (0.0 < ci_level < 1.0):
        raise ContractViolation(f"ci_level must be in (0, 1), got {ci_level!r}")


def _bootstrap(
    current: np.ndarray,
    candidate: np.ndarray,
    true_loss: np.ndarray,
    review_costs: np.ndarray,
    *,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    n = len(current)
    rng = np.random.default_rng(seed)
    differ = np.flatnonzero(current != candidate)
    # +1: only the candidate reviews the item; -1: only the current policy does.
    sign = np.where(candidate[differ], 1.0, -1.0)
    loss = (sign * true_loss[differ]).tolist()
    spend = (sign * review_costs[differ]).tolist()
    total = (sign * (review_costs[differ] - true_loss[differ])).tolist()
    reviews = [int(v) for v in sign.tolist()]

    draws: dict[str, np.ndarray] = {
        key: np.empty(n_bootstrap, dtype=np.float64)
        for key in ("captured_loss", "n_review", "review_spend", "total_cost")
    }
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        counts = np.bincount(idx, minlength=n)[differ].tolist()
        draws["captured_loss"][b] = math.fsum(c * v for c, v in zip(counts, loss))
        draws["review_spend"][b] = math.fsum(c * v for c, v in zip(counts, spend))
        draws["total_cost"][b] = math.fsum(c * v for c, v in zip(counts, total))
        draws["n_review"][b] = float(sum(c * v for c, v in zip(counts, reviews)))
    draws["residual_loss"] = -draws["captured_loss"]

    alpha = (1.0 - ci_level) / 2.0
    delta_ci = {
        key: [
            float(np.percentile(values, 100 * alpha)),
            float(np.percentile(values, 100 * (1.0 - alpha))),
        ]
        for key, values in draws.items()
    }
    return {
        "ci_level": ci_level,
        "delta_ci": {key: delta_ci[key] for key in _DELTA_KEYS},
        "method": "paired_bootstrap",
        "n_bootstrap": n_bootstrap,
        "n_differing_items": int(differ.size),
        "seed": seed,
    }


def _ci_direction(ci: list[float], n_differing: int, *, higher_is_better: bool) -> CiDirection:
    lo, hi = ci
    # No differing item, or differing items that never move this delta (e.g.
    # swapped items that are all error-free: the loss delta is 0 in every
    # resample): the quantity does not change, it is not "uncertain".
    if n_differing == 0 or lo == hi == 0:
        return "same"
    if lo > 0:
        return "better" if higher_is_better else "worse"
    if hi < 0:
        return "worse" if higher_is_better else "better"
    return "inconclusive"


def _ci_verdict(uncertainty: dict[str, Any]) -> dict[str, Any]:
    ci, n_diff = uncertainty["delta_ci"], uncertainty["n_differing_items"]
    directions: dict[str, CiDirection] = {
        "captured_loss": _ci_direction(ci["captured_loss"], n_diff, higher_is_better=True),
        "n_review": _ci_direction(ci["n_review"], n_diff, higher_is_better=False),
        "review_spend": _ci_direction(ci["review_spend"], n_diff, higher_is_better=False),
    }
    values = set(directions.values())
    label: CiVerdictLabel
    if values == {"same"}:
        label = "no_change"
    elif not values & {"better", "worse"}:
        label = "inconclusive"
    elif "worse" not in values:
        label = "improvement"
    elif "better" not in values:
        label = "regression"
    else:
        label = "trade_off"
    total_cost = _ci_direction(ci["total_cost"], n_diff, higher_is_better=False)
    return {"label": label, "directions": directions, "total_cost": total_cost}


def check_regression_gate(result: "ComparisonResult") -> None:
    """`compare --fail-on-regression` (exit 6). Fails iff the candidate is
    SIGNIFICANTLY worse: its total cost (residual + spend) CI lies entirely
    above 0, or the CI verdict is `regression` (worse on at least one of
    captured loss / reviews / spend, better on none). Needs the bootstrap."""
    ci_verdict = result.verdict.get("ci")
    if ci_verdict is None or result.uncertainty is None:
        raise ContractViolation(
            "the regression gate needs the bootstrap CI; n_bootstrap must be > 0"
        )
    reasons = []
    ci = result.uncertainty["delta_ci"]
    level = f"{result.uncertainty['ci_level'] * 100:g}%"
    if ci_verdict["total_cost"] == "worse":
        reasons.append(
            f"total cost (residual + review spend) rises: {level} CI of the delta "
            f"[{ci['total_cost'][0]!r}, {ci['total_cost'][1]!r}] is entirely above 0"
        )
    if ci_verdict["label"] == "regression":
        worse = sorted(k for k, v in ci_verdict["directions"].items() if v == "worse")
        reasons.append(f"significantly worse on {', '.join(worse)} and better on nothing")
    if reasons:
        raise CandidateRegression(
            "Candidate policy is significantly worse than the current policy "
            f"(basis: {result.basis['source']}): " + "; ".join(reasons)
        )


def _delta(candidate: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    return {key: candidate[key] - current[key] for key in _DELTA_KEYS}


def run(
    *,
    current_policy: dict[str, Any],
    current_calibrator: dict[str, Any],
    candidate_policy: dict[str, Any],
    candidate_calibrator: dict[str, Any],
    history: pl.DataFrame,
    history_sha256: str,
    eval_costs: CostModel | None = None,
    eval_costs_sha256: str | None = None,
    seed: int = 42,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = DEFAULT_CI_LEVEL,
    tool_version: str,
) -> ComparisonResult:
    """Compare two policies on one labeled history. See docs/interfaces.md §14.

    `*_policy` / `*_calibrator` are PARSED artifacts (dicts); relationships
    are verified here, before anything is evaluated. `history_sha256` and
    `eval_costs_sha256` are raw-file hashes supplied by the caller (the CLI
    reads the files), recorded in `input_sha256s`.
    """
    if (eval_costs is None) != (eval_costs_sha256 is None):
        raise ValueError("eval_costs and eval_costs_sha256 must be given together")
    validate_uncertainty_options(seed, n_bootstrap, ci_level)

    sides = _verify_sides(
        {"current": current_policy, "candidate": candidate_policy},
        {"current": current_calibrator, "candidate": candidate_calibrator},
    )

    validated = contract.validate(history, require_truth=True)
    segments: list[str] = validated["segment"].to_list()
    scores = validated["score"].to_numpy()
    predicted = validated["predicted_value"].to_list()
    truth = validated["true_value"].to_list()
    history_segments = sorted(set(segments))

    basis = _resolve_basis(sides, history_segments, eval_costs)
    basis_segments: dict[str, dict[str, float]] = basis["segments"]

    is_error = np.array(
        [1.0 if pv != tv else 0.0 for pv, tv in zip(predicted, truth)], dtype=np.float64
    )
    false_accept_costs = np.array(
        [basis_segments[seg]["false_accept_cost"] for seg in segments], dtype=np.float64
    )
    review_costs = np.array(
        [basis_segments[seg]["review_cost"] for seg in segments], dtype=np.float64
    )
    true_loss = is_error * false_accept_costs
    total_true_loss = math.fsum(true_loss.tolist())

    replays: dict[Side, Replay] = {}
    policies: dict[str, Any] = {}
    for side in SIDES:
        p_error = np.asarray(sides[side].calibrator.predict(scores), dtype=np.float64)
        replays[side] = replay_thresholds(sides[side].policy["segments"], p_error, segments)
        policies[side] = _side_result(
            sides[side],
            replays[side],
            segments,
            history_segments,
            true_loss,
            review_costs,
            total_true_loss,
        )

    cur_out, cand_out = policies["current"]["outcome"], policies["candidate"]["outcome"]
    delta = _delta(cand_out, cur_out)
    delta["segments"] = {
        seg: _delta(cand_out["segments"][seg], cur_out["segments"][seg]) for seg in history_segments
    }

    rev_cur, rev_cand = replays["current"].reviewed, replays["candidate"].reviewed
    overlap = {
        "both": _group(rev_cur & rev_cand, true_loss, review_costs),
        "current_only": _group(rev_cur & ~rev_cand, true_loss, review_costs),
        "candidate_only": _group(~rev_cur & rev_cand, true_loss, review_costs),
        "neither": {"n_items": int((~rev_cur & ~rev_cand).sum())},
    }

    history_summary = {
        "n_history": len(segments),
        "n_errors": int(is_error.sum()),
        "total_true_loss": total_true_loss,
    }
    input_sha256s: dict[str, str | None] = {
        "candidate_calibrator": sides["candidate"].calibrator_hash,
        "candidate_policy": sides["candidate"].policy_hash,
        "current_calibrator": sides["current"].calibrator_hash,
        "current_policy": sides["current"].policy_hash,
        "eval_costs": eval_costs_sha256,
        "history": history_sha256,
    }
    verdict = _verdict(rev_cur, rev_cand, true_loss, review_costs)
    uncertainty: dict[str, Any] | None = None
    verdict["ci"] = None
    if n_bootstrap > 0:
        uncertainty = _bootstrap(
            rev_cur,
            rev_cand,
            true_loss,
            review_costs,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            seed=seed,
        )
        verdict["ci"] = _ci_verdict(uncertainty)

    comparison: dict[str, Any] = {
        "basis": basis,
        "delta": delta,
        "history": history_summary,
        "input_sha256s": input_sha256s,
        "overlap": overlap,
        "policies": policies,
        "tool_version": tool_version,
        "uncertainty": uncertainty,
        "verdict": verdict,
    }
    return ComparisonResult(comparison_hash=content_hash(comparison), **comparison)

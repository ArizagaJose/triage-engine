"""Unit tests for triage.compare (docs/interfaces.md §14), on the hand-checkable
history in tests/support.py (`compare_history`): every expected number below
is derived from that table, not from running the code.

Policies used throughout (p_error = 1 - score unless "inverted"):

    CUR       A >= 0.45, B null      -> a1 a2 a3     captured 200  n 3  spend 6
    IMPROVED  A >= 0.60, B >= 0.75   -> a1 a2 b1     captured 210  n 3  spend 5
    WIDER     A >= 0.45, B >= 0.35   -> a1 a2 a3 b1 b2 b3   220    n 6  spend 9
    INVERTED  (p_error = score) A >= 0.45, B >= 0.35
                                     -> a3 a4 b2 b3 b4      10     n 5  spend 7
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from tests.support import (
    COMPARE_COSTS,
    calibrator_dict,
    compare_history,
    cost_model_dict,
    history_frame,
    policy_dict,
)
from triage import backtest, compare
from triage.artifacts import content_hash
from triage.calibration import Calibrator
from triage.costs import CostModel
from triage.exceptions import CandidateRegression, ContractViolation

CAL = calibrator_dict()
INV_CAL = calibrator_dict(inverted=True)


def cur(**kw: Any) -> dict[str, Any]:
    return policy_dict(CAL, {"A": 0.45, "B": None}, **kw)


def improved(**kw: Any) -> dict[str, Any]:
    return policy_dict(CAL, {"A": 0.6, "B": 0.75}, **kw)


def wider(**kw: Any) -> dict[str, Any]:
    return policy_dict(CAL, {"A": 0.45, "B": 0.35}, planned_n_review=6, **kw)


def inverted(**kw: Any) -> dict[str, Any]:
    return policy_dict(INV_CAL, {"A": 0.45, "B": 0.35}, planned_n_review=5, **kw)


def run(
    current: dict[str, Any],
    candidate: dict[str, Any],
    *,
    current_cal: dict[str, Any] = CAL,
    candidate_cal: dict[str, Any] = CAL,
    copies: int = 1,
    eval_costs: dict[str, tuple[float, float]] | None = None,
    n_bootstrap: int = 0,
    **kw: Any,
) -> compare.ComparisonResult:
    return compare.run(
        current_policy=current,
        current_calibrator=current_cal,
        candidate_policy=candidate,
        candidate_calibrator=candidate_cal,
        history=kw.pop("history", compare_history(copies)),
        history_sha256="history-sha",
        eval_costs=(CostModel.model_validate(cost_model_dict(eval_costs)) if eval_costs else None),
        eval_costs_sha256="eval-sha" if eval_costs else None,
        n_bootstrap=n_bootstrap,
        tool_version="0.1.0",
        **kw,
    )


def outcome(result: compare.ComparisonResult, side: str) -> dict[str, Any]:
    out: dict[str, Any] = result.policies[side]["outcome"]
    return out


# ---------------------------------------------------------------------------
# Evaluation: exact numbers and verdicts
# ---------------------------------------------------------------------------


class TestKnownOutcomes:
    def test_current_policy_numbers(self) -> None:
        out = outcome(run(cur(), cur()), "current")
        assert out["captured_loss"] == 200.0
        assert out["residual_loss"] == 20.0
        assert out["n_review"] == 3
        assert out["review_spend"] == 6.0
        assert out["total_cost"] == 26.0
        assert out["capture_rate"] == 200.0 / 220.0
        assert out["efficiency"] == 200.0 / 3

    def test_identical_policies_no_change(self) -> None:
        result = run(cur(), cur())
        assert result.verdict["label"] == "no_change"
        assert result.verdict["total_cost"] == "same"
        assert set(result.verdict["directions"].values()) == {"same"}
        assert all(result.delta[k] == 0 for k in compare._DELTA_KEYS)
        assert result.overlap["current_only"]["n_review"] == 0
        assert result.overlap["candidate_only"]["n_review"] == 0
        assert result.overlap["both"]["n_review"] == 3

    def test_known_improvement(self) -> None:
        result = run(cur(), improved())
        assert result.delta["captured_loss"] == 10.0
        assert result.delta["n_review"] == 0
        assert result.delta["review_spend"] == -1.0
        assert result.delta["total_cost"] == -11.0
        assert result.verdict["label"] == "improvement"
        assert result.verdict["directions"] == {
            "captured_loss": "better",
            "n_review": "same",
            "review_spend": "better",
        }
        assert result.verdict["total_cost"] == "better"

    def test_known_regression_is_the_mirror_image(self) -> None:
        result = run(improved(), cur())
        assert result.verdict["label"] == "regression"
        assert result.delta["captured_loss"] == -10.0
        assert result.verdict["total_cost"] == "worse"

    def test_inverted_calibrator_is_a_regression(self) -> None:
        """Each policy is replayed with ITS OWN calibrator: the inverted one
        reviews the most confident items and misses nearly everything."""
        result = run(cur(), inverted(), candidate_cal=INV_CAL)
        out = outcome(result, "candidate")
        assert (out["captured_loss"], out["n_review"], out["review_spend"]) == (10.0, 5, 7.0)
        assert result.verdict["label"] == "regression"

    def test_bigger_budget_is_a_trade_off_with_lower_total_cost(self) -> None:
        result = run(cur(), wider())
        assert result.delta["captured_loss"] == 20.0
        assert result.delta["n_review"] == 3
        assert result.delta["review_spend"] == 3.0
        assert result.delta["total_cost"] == -17.0  # 26 -> 9
        assert result.verdict["label"] == "trade_off"
        assert result.verdict["total_cost"] == "better"

    def test_different_budget_types_are_comparable(self) -> None:
        """Budgets are policy choices, not prices: fraction vs count is fine
        and each side's declared budget is recorded."""
        result = run(cur(budget=("fraction", 0.5)), wider(budget=("count", 6)))
        assert result.policies["current"]["budget"] == {"type": "fraction", "value": 0.5}
        assert result.policies["candidate"]["budget"] == {"type": "count", "value": 6.0}
        assert result.verdict["label"] == "trade_off"

    def test_totals_scale_with_history_size(self) -> None:
        result = run(cur(), wider(), copies=5)
        assert result.delta["captured_loss"] == 100.0
        assert result.delta["n_review"] == 15
        assert result.history == {"n_history": 40, "n_errors": 20, "total_true_loss": 1100.0}


class TestAccountingIdentities:
    @pytest.mark.parametrize(
        "pair",
        [("cur", "cur"), ("cur", "improved"), ("improved", "cur"), ("cur", "wider")],
    )
    def test_identities(self, pair: tuple[str, str]) -> None:
        policies = {"cur": cur(), "improved": improved(), "wider": wider()}
        result = run(policies[pair[0]], policies[pair[1]], copies=3)
        d, o = result.delta, result.overlap
        total = result.history["total_true_loss"]
        for side in ("current", "candidate"):
            out = outcome(result, side)
            assert out["residual_loss"] == total - out["captured_loss"]
            assert out["total_cost"] == out["residual_loss"] + out["review_spend"]
            assert math.isclose(
                sum(s["captured_loss"] for s in out["segments"].values()), out["captured_loss"]
            )
        assert math.isclose(d["residual_loss"], -d["captured_loss"], abs_tol=1e-9)
        assert math.isclose(
            d["captured_loss"],
            o["candidate_only"]["captured_loss"] - o["current_only"]["captured_loss"],
            abs_tol=1e-9,
        )
        n_groups = sum(o[g]["n_review"] for g in ("both", "current_only", "candidate_only"))
        assert n_groups + o["neither"]["n_items"] == result.history["n_history"]
        for seg, seg_delta in d["segments"].items():
            cur_seg = outcome(result, "current")["segments"][seg]
            cand_seg = outcome(result, "candidate")["segments"][seg]
            expected = cand_seg["captured_loss"] - cur_seg["captured_loss"]
            assert seg_delta["captured_loss"] == expected

    def test_verdict_directions_agree_with_delta_signs(self) -> None:
        for current, candidate in [(cur(), improved()), (cur(), wider()), (wider(), cur())]:
            result = run(current, candidate)
            d, dirs = result.delta, result.verdict["directions"]
            assert dirs["captured_loss"] == _sign_word(d["captured_loss"], higher_better=True)
            assert dirs["n_review"] == _sign_word(d["n_review"], higher_better=False)
            assert dirs["review_spend"] == _sign_word(d["review_spend"], higher_better=False)
            assert result.verdict["total_cost"] == _sign_word(d["total_cost"], higher_better=False)

    def test_equal_loss_swap_is_no_change_despite_float_grouping(self) -> None:
        """Totals follow backtest's convention (sequential sum of per-segment
        fsums), which can differ by ULPs between two review sets whose exact
        sums are equal. The verdict compares fsums over the differing items,
        so such a swap is `no_change`, never a spurious +-ULP verdict.

        Current reviews A (1e16) + B (1) + C (1): sequential 1e16 + 1 + 1 rounds
        to 1e16. Candidate reviews A + D (2) + E (error-free): 1e16 + 2 is
        exact. Exact sums are equal (1e16 + 2); spend and count are equal too.
        """
        costs = {"A": (1e16, 1.0), "B": (1.0, 1.0), "C": (1.0, 1.0), "D": (2.0, 1.0)}
        costs["E"] = (5.0, 1.0)
        history = history_frame(
            [
                ("a", "A", 0.1, "x", "v"),
                ("b", "B", 0.1, "x", "v"),
                ("c", "C", 0.1, "x", "v"),
                ("d", "D", 0.1, "x", "v"),
                ("e", "E", 0.1, "v", "v"),
            ]
        )
        current = policy_dict(CAL, {"A": 0.5, "B": 0.5, "C": 0.5}, costs=costs)
        candidate = policy_dict(CAL, {"A": 0.5, "D": 0.5, "E": 0.5}, costs=costs)
        result = run(current, candidate, history=history)
        assert result.overlap["current_only"]["n_review"] == 2
        assert result.overlap["candidate_only"]["n_review"] == 2
        assert result.verdict["label"] == "no_change"
        assert result.verdict["total_cost"] == "same"


def _sign_word(value: float, *, higher_better: bool) -> str:
    if value == 0:
        return "same"
    return "better" if (value > 0) == higher_better else "worse"


# ---------------------------------------------------------------------------
# Segments: statuses, missing from one policy, absent from history
# ---------------------------------------------------------------------------


class TestSegments:
    def test_statuses(self) -> None:
        no_b = policy_dict(CAL, {"A": 0.45})
        result = run(cur(), no_b)
        cur_segs = outcome(result, "current")["segments"]
        cand_segs = outcome(result, "candidate")["segments"]
        assert cur_segs["A"]["status"] == "planned"
        assert cur_segs["B"]["status"] == "no_threshold"
        assert cand_segs["B"]["status"] == "not_in_plan"
        # Both review nothing in B: identical review sets.
        assert result.verdict["label"] == "no_change"

    def test_segment_missing_from_candidate_is_never_reviewed(self) -> None:
        only_a = policy_dict(CAL, {"A": 0.45})
        result = run(wider(), only_a)
        b = outcome(result, "candidate")["segments"]["B"]
        assert b["n_review"] == 0
        assert b["p_error_threshold"] is None
        assert result.delta["segments"]["B"]["captured_loss"] == -20.0
        assert result.delta["segments"]["A"]["captured_loss"] == 0.0

    def test_planned_segment_absent_from_history_is_listed(self) -> None:
        with_c = policy_dict(CAL, {"A": 0.45, "B": None, "C": 0.5}, costs={**COMPARE_COSTS})
        result = run(cur(), with_c)
        assert result.policies["candidate"]["planned_segments_not_in_history"] == ["C"]
        assert result.policies["current"]["planned_segments_not_in_history"] == []
        assert set(result.delta["segments"]) == {"A", "B"}

    def test_per_segment_spend_and_total_cost(self) -> None:
        seg = outcome(run(cur(), wider()), "candidate")["segments"]
        assert seg["A"]["review_spend"] == 6.0
        assert seg["B"]["review_spend"] == 3.0
        assert seg["B"]["total_cost"] == 0.0 + 3.0


# ---------------------------------------------------------------------------
# Evaluation cost basis
# ---------------------------------------------------------------------------


class TestBasis:
    def test_shared_costs(self) -> None:
        result = run(cur(), improved())
        assert result.basis["source"] == "policies"
        assert result.basis["segments"] == {
            "A": {"false_accept_cost": 100.0, "review_cost": 2.0},
            "B": {"false_accept_cost": 10.0, "review_cost": 1.0},
        }
        assert result.basis["planned_under_basis"] == {"current": True, "candidate": True}
        assert result.basis["ignored_eval_costs_fields"] == []
        assert result.input_sha256s["eval_costs"] is None

    def test_default_fallback_with_equal_costs_is_compatible(self) -> None:
        via_default = policy_dict(
            CAL, {"A": 0.6, "B": 0.75}, costs={"A": (100.0, 2.0), "default": (10.0, 1.0)}
        )
        assert run(cur(), via_default).basis["source"] == "policies"

    def test_budget_and_gates_may_differ(self) -> None:
        other = improved(budget=("money", 5.0), gates={"max_residual_risk": 1.0})
        assert run(cur(), other).basis["source"] == "policies"

    def test_differences_outside_the_history_do_not_matter(self) -> None:
        extra = improved(costs={**COMPARE_COSTS, "Z": (999.0, 9.0)})
        assert run(cur(), extra).basis["source"] == "policies"

    def test_conflicting_costs_are_rejected_naming_every_segment(self) -> None:
        conflicting = improved(costs={"A": (100.0, 3.0), "B": (40.0, 1.0)})
        with pytest.raises(ContractViolation) as exc:
            run(cur(), conflicting)
        message = str(exc.value)
        assert "A: current false_accept_cost=100.0 review_cost=2.0; candidate " in message
        assert "review_cost=3.0" in message
        assert "B: current false_accept_cost=10.0" in message
        assert "false_accept_cost=40.0" in message
        assert "--eval-costs" in message

    def test_unpriceable_segment_is_rejected(self) -> None:
        no_b = improved(costs={"A": (100.0, 2.0)})
        with pytest.raises(ContractViolation, match="B: current .*candidate cannot price it"):
            run(cur(), no_b)

    def test_eval_costs_resolve_a_conflict(self) -> None:
        conflicting = improved(costs={"A": (100.0, 2.0), "B": (40.0, 1.0)})
        result = run(cur(), conflicting, eval_costs=COMPARE_COSTS)
        assert result.basis["source"] == "eval_costs"
        assert result.basis["ignored_eval_costs_fields"] == ["budget", "gates"]
        assert result.basis["planned_under_basis"] == {"current": True, "candidate": False}
        assert result.basis["differences"]["candidate"] == {
            "B": {"false_accept_cost": 40.0, "review_cost": 1.0}
        }
        assert result.basis["differences"]["current"] == {}
        assert result.input_sha256s["eval_costs"] == "eval-sha"

    def test_eval_costs_price_both_policies(self) -> None:
        """Same review sets, different basis: every loss is re-priced."""
        result = run(cur(), improved(), eval_costs={"A": (1.0, 1.0), "B": (1000.0, 1.0)})
        assert outcome(result, "current")["captured_loss"] == 2.0
        assert outcome(result, "candidate")["captured_loss"] == 1002.0
        assert result.history["total_true_loss"] == 2002.0

    def test_verdict_can_flip_with_the_basis(self) -> None:
        """The reason the basis must be explicit (the analysis example, small)."""
        a_heavy = run(cur(), improved(), eval_costs={"A": (100.0, 2.0), "B": (10.0, 1.0)})
        b_heavy = run(cur(), improved(), eval_costs={"A": (100.0, 2.0), "B": (0.0, 50.0)})
        assert a_heavy.verdict["total_cost"] == "better"
        assert b_heavy.verdict["total_cost"] == "worse"

    def test_eval_costs_must_price_every_history_segment(self) -> None:
        with pytest.raises(ContractViolation, match=r"cannot price history segment\(s\) \['B'\]"):
            run(cur(), improved(), eval_costs={"A": (100.0, 2.0)})

    def test_eval_costs_and_sha_go_together(self) -> None:
        with pytest.raises(ValueError):
            compare.run(
                current_policy=cur(),
                current_calibrator=CAL,
                candidate_policy=cur(),
                candidate_calibrator=CAL,
                history=compare_history(),
                history_sha256="h",
                eval_costs=CostModel.model_validate(cost_model_dict()),
                tool_version="0.1.0",
            )


# ---------------------------------------------------------------------------
# Artifact relationships
# ---------------------------------------------------------------------------


class TestArtifactRelationships:
    def test_shared_calibrator_is_allowed(self) -> None:
        result = run(cur(), wider())
        assert result.input_sha256s["current_calibrator"] == content_hash(CAL)
        assert result.input_sha256s["candidate_calibrator"] == content_hash(CAL)
        assert result.input_sha256s["current_policy"] == cur()["policy_hash"]

    def test_tampered_policy_names_the_side(self) -> None:
        tampered = improved()
        tampered["segments"]["A"]["p_error_threshold"] = 0.1
        with pytest.raises(ContractViolation, match="candidate policy: self-hash mismatch"):
            run(cur(), tampered)

    def test_wrong_calibrator_names_the_side_and_the_owner(self) -> None:
        with pytest.raises(ContractViolation) as exc:
            run(cur(), inverted(), candidate_cal=CAL)
        message = str(exc.value)
        assert message.startswith("candidate calibrator mismatch")
        assert "it is the current policy's calibrator" in message

    def test_unrelated_calibrator_has_no_owner_hint(self) -> None:
        with pytest.raises(ContractViolation) as exc:
            run(cur(), cur(), candidate_cal=calibrator_dict(seed=99))
        assert "candidate calibrator mismatch" in str(exc.value)
        assert "it is the" not in str(exc.value)

    def test_swapped_calibrators_are_called_swapped(self) -> None:
        with pytest.raises(ContractViolation, match="Calibrators are swapped"):
            run(cur(), inverted(), current_cal=INV_CAL, candidate_cal=CAL)

    @pytest.mark.parametrize("key", ["policy_hash", "segments", "config_snapshot", "budget"])
    def test_policy_missing_key(self, key: str) -> None:
        broken = cur()
        del broken[key]
        message = f"current policy: missing required key '{key}'"
        with pytest.raises(ContractViolation, match=message):
            run(broken, cur())

    def test_policy_with_invalid_cost_snapshot(self) -> None:
        broken = cur()
        broken["config_snapshot"]["segments"]["A"]["review_cost"] = -1.0
        broken["policy_hash"] = content_hash(broken)
        with pytest.raises(ContractViolation, match="current policy: config_snapshot failed"):
            run(broken, cur())

    def test_malformed_calibrator(self) -> None:
        broken = dict(CAL)
        del broken["params"]
        policy = policy_dict(broken, {"A": 0.45, "B": None})
        with pytest.raises(ContractViolation, match="current calibrator: missing required key"):
            run(policy, cur(), current_cal=broken)

    def test_history_without_labels_is_rejected(self) -> None:
        unlabeled = compare_history().drop("true_value")
        with pytest.raises(ContractViolation, match="true_value"):
            run(cur(), cur(), history=unlabeled)


# ---------------------------------------------------------------------------
# Parity with backtest (one replay rule, one accounting)
# ---------------------------------------------------------------------------


def test_outcome_under_own_costs_equals_backtest_policy_strategy() -> None:
    """A comparison under a policy's own costs reproduces backtest.json's
    `strategies.policy` exactly (same replay helper, same accounting)."""
    policy = wider()
    bt = backtest.run(
        policy,
        Calibrator.from_dict(CAL),
        compare_history(3),
        seed=1,
        n_random_draws=0,
        input_sha256s={},
        tool_version="0.1.0",
    ).strategies["policy"]
    out = outcome(run(cur(), policy, copies=3), "candidate")
    for key in ("captured_loss", "residual_loss", "n_review", "review_spend", "efficiency"):
        assert out[key] == bt[key]
    assert out["capture_rate"] == bt["capture_rate"]
    for seg, stats in bt["segments"].items():
        for key, value in stats.items():
            assert out["segments"][seg][key] == value


def test_replay_thresholds_rule() -> None:
    p_error = np.array([0.9, 0.5, 0.5, 0.2, 0.7])
    segments = ["A", "A", "B", "B", "Z"]
    replay = backtest.replay_thresholds(
        {"A": {"p_error_threshold": 0.5}, "B": {"p_error_threshold": None}}, p_error, segments
    )
    # Ties AT the threshold are reviewed; null thresholds and unseen segments never.
    assert replay.reviewed.tolist() == [True, True, False, False, False]
    assert replay.thresholds == {"A": 0.5, "B": None, "Z": None}


# ---------------------------------------------------------------------------
# Uncertainty (paired bootstrap) and the regression gate
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_no_bootstrap_means_no_ci(self) -> None:
        result = run(cur(), wider(), n_bootstrap=0)
        assert result.uncertainty is None
        assert result.verdict["ci"] is None

    def test_record(self) -> None:
        result = run(cur(), wider(), copies=10, n_bootstrap=200, seed=3, ci_level=0.9)
        unc = result.uncertainty
        assert unc is not None
        assert unc["method"] == "paired_bootstrap"
        assert (unc["n_bootstrap"], unc["seed"], unc["ci_level"]) == (200, 3, 0.9)
        assert unc["n_differing_items"] == 30  # b1 b2 b3 x 10
        assert set(unc["delta_ci"]) == set(compare._DELTA_KEYS)
        for key in compare._DELTA_KEYS:
            lo, hi = unc["delta_ci"][key]
            assert lo <= hi

    def test_deterministic_and_seed_sensitive(self) -> None:
        a = run(cur(), wider(), copies=10, n_bootstrap=200, seed=1)
        b = run(cur(), wider(), copies=10, n_bootstrap=200, seed=1)
        c = run(cur(), wider(), copies=10, n_bootstrap=200, seed=2)
        assert a.to_dict() == b.to_dict()
        assert a.comparison_hash == b.comparison_hash
        assert a.uncertainty != c.uncertainty

    def test_identical_policies_have_a_zero_ci_and_same_directions(self) -> None:
        result = run(cur(), cur(), n_bootstrap=50)
        assert result.uncertainty is not None
        assert all(ci == [0.0, 0.0] for ci in result.uncertainty["delta_ci"].values())
        assert result.verdict["ci"]["label"] == "no_change"

    def test_residual_ci_mirrors_captured_ci(self) -> None:
        result = run(cur(), wider(), copies=10, n_bootstrap=100)
        ci = result.uncertainty["delta_ci"] if result.uncertainty else {}
        assert ci["residual_loss"] == [-ci["captured_loss"][1], -ci["captured_loss"][0]]

    def test_large_history_makes_the_trade_off_significant(self) -> None:
        result = run(cur(), wider(), copies=50, n_bootstrap=300)
        assert result.verdict["ci"] == {
            "label": "trade_off",
            "directions": {
                "captured_loss": "better",
                "n_review": "worse",
                "review_spend": "worse",
            },
            "total_cost": "better",
        }

    def test_small_history_is_inconclusive(self) -> None:
        """One copy of the table: a 1-item difference is noise."""
        single = policy_dict(CAL, {"A": 0.45, "B": 0.75})  # cur + b1
        result = run(cur(), single, n_bootstrap=300)
        assert result.verdict["label"] == "trade_off"
        assert result.verdict["ci"]["label"] == "inconclusive"

    def test_error_free_swap_is_same_not_inconclusive(self) -> None:
        """Differing items that never move a delta give a [0, 0] CI: `same`."""
        plus_a4 = policy_dict(CAL, {"A": 0.15, "B": None})  # cur + a4 (no error)
        result = run(cur(), plus_a4, copies=10, n_bootstrap=200)
        assert result.uncertainty is not None
        assert result.uncertainty["delta_ci"]["captured_loss"] == [0.0, 0.0]
        assert result.verdict["ci"]["directions"]["captured_loss"] == "same"
        assert result.verdict["ci"]["directions"]["n_review"] == "worse"

    @pytest.mark.parametrize(
        "options,message",
        [
            ({"seed": -1}, "seed must be a non-negative integer"),
            ({"n_bootstrap": -1}, "n_bootstrap must be an integer >= 0"),
            ({"ci_level": 1.0}, r"ci_level must be in \(0, 1\)"),
            ({"ci_level": 0.0}, r"ci_level must be in \(0, 1\)"),
            ({"ci_level": float("nan")}, r"ci_level must be in \(0, 1\)"),
        ],
    )
    def test_option_validation(self, options: dict[str, Any], message: str) -> None:
        with pytest.raises(ContractViolation, match=message):
            run(cur(), cur(), **options)


class TestRegressionGate:
    def test_passes_on_improvement_and_cheaper_trade_off(self) -> None:
        compare.check_regression_gate(run(cur(), improved(), copies=50, n_bootstrap=200))
        compare.check_regression_gate(run(cur(), wider(), copies=50, n_bootstrap=200))

    def test_fails_on_significant_regression(self) -> None:
        result = run(cur(), inverted(), candidate_cal=INV_CAL, copies=50, n_bootstrap=200)
        with pytest.raises(CandidateRegression) as exc:
            compare.check_regression_gate(result)
        assert "total cost (residual + review spend) rises" in str(exc.value)
        assert "significantly worse on captured_loss, n_review, review_spend" in str(exc.value)

    def test_fails_on_costlier_trade_off(self) -> None:
        """Fewer reviews but much less loss caught: a trade-off by dominance,
        but total cost is significantly higher, so the gate fails."""
        result = run(wider(), cur(), copies=50, n_bootstrap=200)
        assert result.verdict["ci"]["label"] == "trade_off"
        with pytest.raises(CandidateRegression, match="total cost"):
            compare.check_regression_gate(result)

    def test_noise_does_not_fail_the_gate(self) -> None:
        single = policy_dict(CAL, {"A": 0.6, "B": None})  # drops a3 (no error)
        compare.check_regression_gate(run(cur(), single, n_bootstrap=300))

    def test_needs_the_bootstrap(self) -> None:
        with pytest.raises(ContractViolation, match="needs the bootstrap CI"):
            compare.check_regression_gate(run(cur(), cur(), n_bootstrap=0))


def test_comparison_hash_is_content_hash() -> None:
    result = run(cur(), wider(), n_bootstrap=20)
    as_dict = result.to_dict()
    assert as_dict["comparison_hash"] == content_hash(as_dict)
    as_dict["created_at"] = "2000-01-01T00:00:00+00:00"
    assert content_hash(as_dict) == result.comparison_hash

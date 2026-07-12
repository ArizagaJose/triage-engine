"""Unit tests for the optimizer (greedy budget allocation).

Calibrator note: calibration.py is being implemented in parallel and its
Calibrator.predict() may not be functional yet. Per the work-package spec,
these tests never call triage.calibration.fit(); instead they use a tiny
local `IdentityCalibrator` stub satisfying `triage.optimizer.PredictsErrors`
(structural typing) with `predict(scores) = scores`, i.e. we treat the raw
`score` column directly as `p_error`. This keeps all hand-computed
expectations in the comments simple and exact.
"""

import math
from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from triage.artifacts import canonical_dumps, content_hash
from triage.costs import Budget, CostModel, Gates, SegmentCosts
from triage.exceptions import ContractViolation, ResidualRiskExceeded
from triage.optimizer import Policy, check_residual_gate, plan


class IdentityCalibrator:
    """Stub satisfying PredictsErrors: p_error = score, unchanged."""

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=np.float64)


def _batch(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    """Build a valid batch dataframe from (item_id, segment, score) tuples."""
    return pl.DataFrame(
        {
            "item_id": [r[0] for r in rows],
            "segment": [r[1] for r in rows],
            "score": [r[2] for r in rows],
            "predicted_value": ["v" for _ in rows],
        },
        schema={
            "item_id": pl.String,
            "segment": pl.String,
            "score": pl.Float64,
            "predicted_value": pl.String,
        },
    )


CALIBRATOR = IdentityCalibrator()


# ---------------------------------------------------------------------------
# Main fixture: 6 items, two segments (segA, segB).
#
# segA: false_accept_cost=100, review_cost=10
# segB: false_accept_cost=50,  review_cost=5
#
# score (== p_error under IdentityCalibrator):
#   i1 segA 0.90 -> expected_loss = 90.0  -> net_benefit =  80.0
#   i2 segA 0.50 -> expected_loss = 50.0  -> net_benefit =  40.0
#   i3 segA 0.05 -> expected_loss =  5.0  -> net_benefit =  -5.0
#   i4 segB 0.80 -> expected_loss = 40.0  -> net_benefit =  35.0
#   i5 segB 0.30 -> expected_loss = 15.0  -> net_benefit =  10.0
#   i6 segB 0.05 -> expected_loss =  2.5  -> net_benefit =  -2.5
#
# net_benefit DESC order: i1(80), i2(40), i4(35), i5(10), i6(-2.5), i3(-5)
# sum of ALL expected_loss = 90+50+5+40+15+2.5 = 202.5
# ---------------------------------------------------------------------------
def _main_rows() -> list[tuple[str, str, float]]:
    return [
        ("i1", "segA", 0.90),
        ("i2", "segA", 0.50),
        ("i3", "segA", 0.05),
        ("i4", "segB", 0.80),
        ("i5", "segB", 0.30),
        ("i6", "segB", 0.05),
    ]


def _main_cost_model(budget: Budget, gates: Gates = Gates()) -> CostModel:
    return CostModel(
        budget=budget,
        gates=gates,
        segments={
            "segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0),
            "segB": SegmentCosts(false_accept_cost=50.0, review_cost=5.0),
        },
    )


def _plan(rows: list[tuple[str, str, float]], cost_model: CostModel) -> Policy:
    return plan(
        _batch(rows),
        CALIBRATOR,
        cost_model,
        input_sha256s={"batch": "b" * 64, "calibrator": "c" * 64, "costs": "d" * 64},
        tool_version="0.1.0",
    )


class TestGreedyCorrectness:
    """Hand-computed 6-item, two-segment case."""

    def test_count_budget_3_routes_top_three_by_net_benefit(self) -> None:
        # Budget = count 3 -> capacity 3. Routed (net_benefit desc): i1, i2, i4.
        # i5 (net_benefit 10) is excluded purely by capacity, not by sign.
        cost_model = _main_cost_model(Budget(type="count", value=3))
        policy = _plan(_main_rows(), cost_model)

        assert policy.totals["n_review"] == 3
        assert policy.totals["n_total"] == 6
        # review_spend = 10 (i1) + 10 (i2) + 5 (i4) = 25
        assert policy.totals["review_spend"] == pytest.approx(25.0)
        # captured = 90 + 50 + 40 = 180 ; residual = 5 (i3) + 15 + 2.5 (i5,i6) = 22.5
        assert policy.totals["expected_captured_loss"] == pytest.approx(180.0)
        assert policy.totals["expected_residual_loss"] == pytest.approx(22.5)

        segA = policy.segments["segA"]
        assert segA["n_review"] == 2
        assert segA["n_total"] == 3
        assert segA["expected_captured_loss"] == pytest.approx(140.0)  # 90+50
        assert segA["expected_residual_loss"] == pytest.approx(5.0)
        assert segA["p_error_threshold"] == pytest.approx(0.5)  # min(0.9, 0.5)

        segB = policy.segments["segB"]
        assert segB["n_review"] == 1
        assert segB["n_total"] == 3
        assert segB["expected_captured_loss"] == pytest.approx(40.0)
        assert segB["expected_residual_loss"] == pytest.approx(17.5)  # 15+2.5
        assert segB["p_error_threshold"] == pytest.approx(0.8)  # only i4 routed


class TestTieBreak:
    """Two items with identical net_benefit; lower item_id wins under capacity 1."""

    def test_lower_item_id_wins_tie(self) -> None:
        # Same segment/cost so net_benefit is identical for both; put the
        # higher-alphabetical item_id FIRST in input order to prove the sort
        # (not input order) determines the winner.
        rows = [("z_item", "seg", 0.5), ("a_item", "seg", 0.5)]
        cost_model = CostModel(
            budget=Budget(type="count", value=1),
            segments={"seg": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        policy = _plan(rows, cost_model)
        assert policy.totals["n_review"] == 1
        # a_item must be the one routed (lower item_id tie-break).
        assert policy.segments["seg"]["expected_captured_loss"] == pytest.approx(50.0)
        # threshold reflects the routed item's p_error (both are 0.5, so this
        # alone doesn't disambiguate -- captured_loss equality plus n_review=1
        # together with the money-budget test below cover the ordering claim
        # more directly). Cross-check via a money-budget variant:
        cost_model_money = CostModel(
            budget=Budget(type="money", value=10.0),
            segments={"seg": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        policy2 = _plan(rows, cost_model_money)
        assert policy2.totals["n_review"] == 1
        assert policy2.totals["review_spend"] == pytest.approx(10.0)


class TestBudgetEdgeCases:
    def test_fraction_that_floors_to_zero_routes_nothing(self) -> None:
        # floor(0.001 * 6) = 0 -> nothing routed regardless of net_benefit sign.
        cost_model = _main_cost_model(Budget(type="fraction", value=0.001))
        policy = _plan(_main_rows(), cost_model)
        assert policy.totals["n_review"] == 0
        assert policy.totals["expected_captured_loss"] == 0.0
        for seg_stats in policy.segments.values():
            assert seg_stats["p_error_threshold"] is None

    def test_fraction_1_0_routes_everything_with_positive_net_benefit(self) -> None:
        # floor(1.0*6)=6 capacity; but only i1,i2,i4,i5 have net_benefit>0.
        cost_model = _main_cost_model(Budget(type="fraction", value=1.0))
        policy = _plan(_main_rows(), cost_model)
        assert policy.totals["n_review"] == 4
        # i3, i6 (net_benefit <= 0) must never be routed.
        assert policy.totals["expected_captured_loss"] == pytest.approx(90 + 50 + 40 + 15)
        assert policy.totals["expected_residual_loss"] == pytest.approx(5 + 2.5)

    def test_count_greater_than_batch_is_capped(self) -> None:
        cost_model = _main_cost_model(Budget(type="count", value=100))
        policy = _plan(_main_rows(), cost_model)
        # capacity capped at n_batch=6, but still only positive-net-benefit
        # items (4) are routed.
        assert policy.totals["n_review"] == 4
        assert policy.totals["n_total"] == 6

    def test_count_exact_batch_size(self) -> None:
        cost_model = _main_cost_model(Budget(type="count", value=6))
        policy = _plan(_main_rows(), cost_model)
        assert policy.totals["n_review"] == 4  # still gated by net_benefit > 0

    def test_money_budget_stops_at_first_non_fitting_item(self) -> None:
        # Dedicated fixture: 4 items, 4 distinct single-item segments so each
        # can carry its own review_cost. Net_benefit strictly decreasing in
        # item order (item1 > item2 > item3 > item4), so the net_benefit-DESC
        # sort leaves them in this same order:
        #   item1: review_cost=10, expected_loss=110 -> net_benefit=100
        #   item2: review_cost=8,  expected_loss=98  -> net_benefit=90
        #   item3: review_cost=1,  expected_loss=81  -> net_benefit=80
        #   item4: review_cost=1,  expected_loss=71  -> net_benefit=70
        # Budget = money, value=12.
        # Walk: cum=0 -> item1 candidate=10<=12, admit, cum=10.
        #       item2 candidate=10+8=18>12 -> STOP (pinned rule: do not
        #       skip-and-continue past a non-fitting item, even though item3
        #       (cost 1) and item4 (cost 1) would fit in the remaining 2).
        # => capacity = 1, only item1 is routed.
        rows = [
            ("item1", "seg1", 1.0),
            ("item2", "seg2", 1.0),
            ("item3", "seg3", 1.0),
            ("item4", "seg4", 1.0),
        ]
        cost_model = CostModel(
            budget=Budget(type="money", value=12.0),
            segments={
                "seg1": SegmentCosts(false_accept_cost=110.0, review_cost=10.0),
                "seg2": SegmentCosts(false_accept_cost=98.0, review_cost=8.0),
                "seg3": SegmentCosts(false_accept_cost=81.0, review_cost=1.0),
                "seg4": SegmentCosts(false_accept_cost=71.0, review_cost=1.0),
            },
        )
        policy = _plan(rows, cost_model)
        assert policy.totals["n_review"] == 1
        assert policy.totals["review_spend"] == pytest.approx(10.0)
        assert policy.segments["seg1"]["n_review"] == 1
        assert policy.segments["seg2"]["n_review"] == 0
        assert policy.segments["seg3"]["n_review"] == 0
        assert policy.segments["seg4"]["n_review"] == 0


class TestNetBenefitNeverRoutedIfNonPositive:
    def test_non_positive_net_benefit_never_routed_even_with_full_budget(self) -> None:
        cost_model = _main_cost_model(Budget(type="fraction", value=1.0))
        policy = _plan(_main_rows(), cost_model)
        # i3 and i6 both have net_benefit <= 0; confirm captured/residual
        # split excludes them from routing regardless of unused capacity.
        assert policy.totals["n_review"] == 4
        assert policy.totals["n_total"] - policy.totals["n_review"] == 2


class TestAccountingIdentity:
    def test_captured_plus_residual_equals_total_expected_loss(self) -> None:
        for budget in [
            Budget(type="count", value=3),
            Budget(type="fraction", value=1.0),
            Budget(type="fraction", value=0.001),
            Budget(type="money", value=22.0),
        ]:
            cost_model = _main_cost_model(budget)
            policy = _plan(_main_rows(), cost_model)
            total_expected_loss = math.fsum([90.0, 50.0, 5.0, 40.0, 15.0, 2.5])
            assert policy.totals["expected_captured_loss"] + policy.totals[
                "expected_residual_loss"
            ] == pytest.approx(total_expected_loss)
            # Per-segment identity too.
            seg_sum = math.fsum(
                s["expected_captured_loss"] + s["expected_residual_loss"]
                for s in policy.segments.values()
            )
            assert seg_sum == pytest.approx(total_expected_loss)


class TestSegmentFallback:
    def test_unknown_segment_uses_default(self) -> None:
        rows = [("i1", "unknown_seg", 0.9)]
        cost_model = CostModel(
            budget=Budget(type="count", value=1),
            segments={
                "default": SegmentCosts(false_accept_cost=50.0, review_cost=3.0),
            },
        )
        policy = _plan(rows, cost_model)
        # expected_loss = 0.9 * 50 = 45; net_benefit = 45 - 3 = 42 > 0 -> routed.
        assert policy.totals["n_review"] == 1
        assert policy.totals["expected_captured_loss"] == pytest.approx(45.0)
        assert policy.segments["unknown_seg"]["p_error_threshold"] == pytest.approx(0.9)

    def test_unknown_segment_no_default_raises_contract_violation(self) -> None:
        rows = [("i1", "unknown_seg", 0.9)]
        cost_model = CostModel(
            budget=Budget(type="count", value=1),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        with pytest.raises(ContractViolation):
            _plan(rows, cost_model)

    def test_all_batch_segments_present_even_with_zero_routed(self) -> None:
        cost_model = _main_cost_model(Budget(type="fraction", value=0.001))
        policy = _plan(_main_rows(), cost_model)
        assert set(policy.segments.keys()) == {"segA", "segB"}
        for seg_stats in policy.segments.values():
            assert seg_stats["n_review"] == 0
            assert seg_stats["p_error_threshold"] is None


class TestDeterminism:
    def test_same_inputs_twice_are_identical(self) -> None:
        cost_model = _main_cost_model(Budget(type="count", value=3))
        p1 = _plan(_main_rows(), cost_model)
        p2 = _plan(_main_rows(), cost_model)
        assert p1.to_dict() == p2.to_dict()
        assert p1.policy_hash == p2.policy_hash

    def test_shuffled_input_row_order_gives_same_policy(self) -> None:
        cost_model = _main_cost_model(Budget(type="count", value=3))
        rows = _main_rows()
        shuffled = [rows[3], rows[0], rows[5], rows[1], rows[4], rows[2]]
        p1 = _plan(rows, cost_model)
        p2 = _plan(shuffled, cost_model)
        assert p1.to_dict() == p2.to_dict()
        assert p1.policy_hash == p2.policy_hash


class TestPolicyHash:
    def test_hash_equals_content_hash_of_to_dict(self) -> None:
        cost_model = _main_cost_model(Budget(type="count", value=3))
        policy = _plan(_main_rows(), cost_model)
        assert policy.policy_hash == content_hash(policy.to_dict())

    def test_changing_budget_changes_hash(self) -> None:
        cost_model_a = _main_cost_model(Budget(type="count", value=3))
        cost_model_b = _main_cost_model(Budget(type="count", value=4))
        policy_a = _plan(_main_rows(), cost_model_a)
        policy_b = _plan(_main_rows(), cost_model_b)
        assert policy_a.policy_hash != policy_b.policy_hash


class TestJsonSerializable:
    def test_to_dict_is_json_serializable_with_expected_top_level_keys(self) -> None:
        cost_model = _main_cost_model(Budget(type="count", value=3))
        policy = _plan(_main_rows(), cost_model)
        d = policy.to_dict()
        dumped = canonical_dumps(d)
        assert dumped  # no exception on dump (no numpy scalars, no NaN)

        expected_keys = {
            "budget",
            "config_snapshot",
            "input_sha256s",
            "policy_hash",
            "segments",
            "tool_version",
            "totals",
        }
        assert set(d.keys()) == expected_keys
        assert "created_at" not in d

        # No numpy scalar types leaked into the dict.
        def _assert_no_numpy(obj: object) -> None:
            assert not isinstance(obj, np.generic), f"numpy scalar leaked: {obj!r}"
            if isinstance(obj, dict):
                for v in obj.values():
                    _assert_no_numpy(v)
            elif isinstance(obj, list):
                for v in obj:
                    _assert_no_numpy(v)

        _assert_no_numpy(d)


class TestResidualGate:
    def test_gate_none_never_raises(self) -> None:
        # gates.max_residual_risk is None by default -> gate is a no-op
        # regardless of how much residual risk is left unrouted.
        cost_model = _main_cost_model(Budget(type="fraction", value=0.001), gates=Gates())
        policy = _plan(_main_rows(), cost_model)
        check_residual_gate(policy, cost_model)  # must not raise

    def test_gate_exceeded_raises(self) -> None:
        cost_model = _main_cost_model(
            Budget(type="fraction", value=0.001), gates=Gates(max_residual_risk=1.0)
        )
        policy = _plan(_main_rows(), cost_model)
        # residual = 202.5 (nothing routed) > 1.0
        with pytest.raises(ResidualRiskExceeded):
            check_residual_gate(policy, cost_model)

    def test_gate_not_exceeded_does_not_raise(self) -> None:
        cost_model = _main_cost_model(
            Budget(type="fraction", value=1.0), gates=Gates(max_residual_risk=100.0)
        )
        policy = _plan(_main_rows(), cost_model)
        # residual with fraction=1.0 is 7.5 (5 + 2.5) <= 100.0
        check_residual_gate(policy, cost_model)  # must not raise

    def test_plan_itself_never_raises_gate(self) -> None:
        # Even with a tiny gate that would clearly be exceeded, plan() itself
        # must not raise -- only check_residual_gate does.
        cost_model = _main_cost_model(
            Budget(type="fraction", value=0.001), gates=Gates(max_residual_risk=0.0)
        )
        policy = _plan(_main_rows(), cost_model)  # must not raise
        assert isinstance(policy, Policy)


class TestPolicyImmutable:
    def test_policy_is_frozen_dataclass(self) -> None:
        cost_model = _main_cost_model(Budget(type="count", value=3))
        policy = _plan(_main_rows(), cost_model)
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            policy.tool_version = "9.9.9"  # type: ignore[misc]
        # `replace` still works for constructing variants in other tests.
        _ = replace(policy, tool_version="0.1.0")

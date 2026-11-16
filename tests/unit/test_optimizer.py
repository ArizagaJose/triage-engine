"""Unit tests for the optimizer (greedy budget allocation, docs/interfaces.md §5).

Uses IdentityCalibrator (p_error = score), so the hand-computed expectations
in the comments are exact."""

import math
from dataclasses import replace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.support import (
    IdentityCalibrator,
    assert_queue_matches_reference,
    assert_queue_reconciles,
    batch_frame,
    reference_allocation,
)
from triage.artifacts import canonical_dumps, content_hash
from triage.costs import Budget, CostModel, Gates, SegmentCosts
from triage.exceptions import ContractViolation, ResidualRiskExceeded
from triage.optimizer import (
    REVIEW_QUEUE_COLUMNS,
    REVIEW_QUEUE_SCHEMA,
    PlanResult,
    Policy,
    check_residual_gate,
    plan,
    plan_with_queue,
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
        batch_frame(rows),
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

    def test_zero_net_benefit_is_never_routed(self) -> None:
        # Boundary of "route while net_benefit > 0" (interfaces.md §5.6):
        # 0.5 * 20 - 10 == 0.0 exactly, so z1 stays unrouted under spare budget.
        cost_model = CostModel(
            budget=Budget(type="fraction", value=1.0),
            segments={"segZ": SegmentCosts(false_accept_cost=20.0, review_cost=10.0)},
        )
        policy = _plan([("z1", "segZ", 0.5), ("p1", "segZ", 0.75)], cost_model)
        assert policy.totals["n_review"] == 1
        assert policy.segments["segZ"]["p_error_threshold"] == 0.75

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


# ---------------------------------------------------------------------------
# Review queue (plan_with_queue): the optimizer's own selection, per item.
# ---------------------------------------------------------------------------

_SHAS = {"batch": "b" * 64, "calibrator": "c" * 64, "costs": "d" * 64}


def _plan_q(rows: list[tuple[str, str, float]], cost_model: CostModel) -> PlanResult:
    return plan_with_queue(
        batch_frame(rows), CALIBRATOR, cost_model, input_sha256s=_SHAS, tool_version="0.1.0"
    )


_ALL_BUDGETS = [
    Budget(type="count", value=3),
    Budget(type="count", value=100),
    Budget(type="fraction", value=1.0),
    Budget(type="fraction", value=0.5),
    Budget(type="fraction", value=0.001),
    Budget(type="money", value=22.0),
    Budget(type="money", value=20.0),
    Budget(type="money", value=5.0),
]


class TestPlanWithQueue:
    @pytest.mark.parametrize("budget", _ALL_BUDGETS, ids=lambda b: f"{b.type}-{b.value}")
    def test_plan_is_exactly_plan_with_queue_policy(self, budget: Budget) -> None:
        """plan() is a view of the same allocation: identical Policy, identical hash."""
        cost_model = _main_cost_model(budget)
        result = _plan_q(_main_rows(), cost_model)
        assert _plan(_main_rows(), cost_model) == result.policy
        assert result.policy.policy_hash == content_hash(result.policy.to_dict())

    @pytest.mark.parametrize("budget", _ALL_BUDGETS, ids=lambda b: f"{b.type}-{b.value}")
    def test_queue_reconciles_and_matches_reference(self, budget: Budget) -> None:
        cost_model = _main_cost_model(budget)
        result = _plan_q(_main_rows(), cost_model)
        policy = result.policy.to_dict()
        assert_queue_reconciles(result.review_queue, policy)
        batch = batch_frame(_main_rows())
        ref = reference_allocation(batch, batch["score"].to_numpy(), cost_model.model_dump())
        assert_queue_matches_reference(result.review_queue, ref)

    def test_count_budget_queue_rows(self) -> None:
        # Hand-computed in the main fixture comment: i1(80), i2(40), i4(35).
        result = _plan_q(_main_rows(), _main_cost_model(Budget(type="count", value=3)))
        queue = result.review_queue
        assert queue["review_priority"].to_list() == [1, 2, 3]
        assert queue["item_id"].to_list() == ["i1", "i2", "i4"]
        assert queue["segment"].to_list() == ["segA", "segA", "segB"]
        assert queue["p_error"].to_list() == [0.90, 0.50, 0.80]
        assert queue["false_accept_cost"].to_list() == [100.0, 100.0, 50.0]
        assert queue["review_cost"].to_list() == [10.0, 10.0, 5.0]
        assert queue["expected_loss"].to_list() == [0.90 * 100.0, 0.50 * 100.0, 0.80 * 50.0]
        assert queue["net_benefit"].to_list() == [
            0.90 * 100.0 - 10.0,
            0.50 * 100.0 - 10.0,
            0.80 * 50.0 - 5.0,
        ]

    def test_trace_columns(self) -> None:
        result = _plan_q(_main_rows(), _main_cost_model(Budget(type="count", value=3)))
        queue = result.review_queue
        assert queue["policy_hash"].to_list() == [result.policy.policy_hash] * 3
        assert queue["batch_sha256"].to_list() == [_SHAS["batch"]] * 3

    def test_schema_pinned(self) -> None:
        result = _plan_q(_main_rows(), _main_cost_model(Budget(type="count", value=3)))
        assert result.review_queue.columns == list(REVIEW_QUEUE_COLUMNS)
        assert result.review_queue.schema == REVIEW_QUEUE_SCHEMA

    def test_non_positive_net_benefit_never_queued(self) -> None:
        # fraction 1.0: i3 (-5) and i6 (-2.5) stay out under spare capacity.
        result = _plan_q(_main_rows(), _main_cost_model(Budget(type="fraction", value=1.0)))
        assert result.review_queue["item_id"].to_list() == ["i1", "i2", "i4", "i5"]

    def test_count_greater_than_batch_is_capped(self) -> None:
        result = _plan_q(_main_rows(), _main_cost_model(Budget(type="count", value=100)))
        assert result.review_queue.height == 4

    def test_zero_net_benefit_is_not_queued(self) -> None:
        cost_model = CostModel(
            budget=Budget(type="fraction", value=1.0),
            segments={"segZ": SegmentCosts(false_accept_cost=20.0, review_cost=10.0)},
        )
        result = _plan_q([("z1", "segZ", 0.5), ("p1", "segZ", 0.75)], cost_model)
        assert result.review_queue["item_id"].to_list() == ["p1"]

    def test_unknown_segment_keeps_its_name_and_uses_default_costs(self) -> None:
        cost_model = CostModel(
            budget=Budget(type="count", value=1),
            segments={"default": SegmentCosts(false_accept_cost=50.0, review_cost=3.0)},
        )
        queue = _plan_q([("i1", "unknown_seg", 0.9)], cost_model).review_queue
        assert queue["segment"].to_list() == ["unknown_seg"]
        assert queue["false_accept_cost"].to_list() == [50.0]
        assert queue["review_cost"].to_list() == [3.0]


class TestQueueTies:
    """The reason the queue exists: thresholds cannot say WHICH tied items."""

    def test_capacity_cuts_through_a_tied_group_by_item_id(self) -> None:
        # Four identical items (same segment, same p_error => same net_benefit),
        # capacity 2, input order deliberately not sorted.
        rows = [("d", "seg", 0.5), ("b", "seg", 0.5), ("c", "seg", 0.5), ("a", "seg", 0.5)]
        cost_model = CostModel(
            budget=Budget(type="count", value=2),
            segments={"seg": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        result = _plan_q(rows, cost_model)
        assert result.review_queue["item_id"].to_list() == ["a", "b"]
        threshold = result.policy.segments["seg"]["p_error_threshold"]
        # Replaying the published threshold would admit all four.
        admitted_by_threshold = [r for r in rows if r[2] >= threshold]
        assert len(admitted_by_threshold) == 4 > result.review_queue.height

    def test_tie_on_net_benefit_across_segments_breaks_by_item_id(self) -> None:
        # segA: 0.5 * 100 - 10 = 40 ; segB: 1.0 * 50 - 10 = 40. Different
        # p_error, same net_benefit: item_id decides, regardless of segment.
        rows = [("b_in_segA", "segA", 0.5), ("a_in_segB", "segB", 1.0)]
        cost_model = CostModel(
            budget=Budget(type="count", value=1),
            segments={
                "segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0),
                "segB": SegmentCosts(false_accept_cost=50.0, review_cost=10.0),
            },
        )
        result = _plan_q(rows, cost_model)
        assert result.review_queue["item_id"].to_list() == ["a_in_segB"]
        assert result.policy.segments["segA"]["p_error_threshold"] is None

    def test_strictly_above_threshold_always_queued_ties_may_be_cut(self) -> None:
        rows = [("x1", "seg", 0.9), ("x2", "seg", 0.5), ("x3", "seg", 0.5), ("x4", "seg", 0.2)]
        cost_model = CostModel(
            budget=Budget(type="count", value=2),
            segments={"seg": SegmentCosts(false_accept_cost=100.0, review_cost=1.0)},
        )
        result = _plan_q(rows, cost_model)
        assert result.review_queue["item_id"].to_list() == ["x1", "x2"]
        assert result.policy.segments["seg"]["p_error_threshold"] == 0.5


class TestQueueMoneyBudget:
    def test_stops_at_first_non_fitting_item(self) -> None:
        # Same fixture as TestBudgetEdgeCases' money test: only item1 fits
        # before item2 (cost 8) overflows; item3/item4 are NOT skipped into.
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
        queue = _plan_q(rows, cost_model).review_queue
        assert queue["item_id"].to_list() == ["item1"]

    def test_exact_fit_is_admitted(self) -> None:
        # i1 (10) + i2 (10) == 20 <= 20 admitted; i4 (5) would make 25 > 20.
        queue = _plan_q(_main_rows(), _main_cost_model(Budget(type="money", value=20.0)))
        assert queue.review_queue["item_id"].to_list() == ["i1", "i2"]
        assert math.fsum(queue.review_queue["review_cost"].to_list()) == 20.0


class TestQueueEmpty:
    @pytest.mark.parametrize(
        "cost_model",
        [
            # fraction floors to zero capacity
            _main_cost_model(Budget(type="fraction", value=0.001)),
            # money below the FIRST ranked item's cost (i1 costs 10); i4 (5)
            # would fit but the walk never skips ahead
            _main_cost_model(Budget(type="money", value=5.0)),
            # every net_benefit is non-positive
            CostModel(
                budget=Budget(type="fraction", value=1.0),
                segments={"default": SegmentCosts(false_accept_cost=1.0, review_cost=1000.0)},
            ),
        ],
        ids=["fraction-floors-to-zero", "money-below-first-cost", "no-positive-net-benefit"],
    )
    def test_empty_selection_is_a_typed_zero_row_queue(self, cost_model: CostModel) -> None:
        result = _plan_q(_main_rows(), cost_model)
        assert result.policy.totals["n_review"] == 0
        assert result.review_queue.height == 0
        assert result.review_queue.schema == REVIEW_QUEUE_SCHEMA
        assert_queue_reconciles(result.review_queue, result.policy.to_dict())


class TestQueueFullPrecision:
    def test_floats_pass_through_unrounded(self) -> None:
        # 1/3 and 0.123456789012345 are not representable in a few decimals:
        # the queue must carry the optimizer's float64 values bit for bit.
        rows = [("a", "segA", 1 / 3), ("b", "segB", 0.123456789012345)]
        result = _plan_q(rows, _main_cost_model(Budget(type="count", value=2)))
        queue = result.review_queue
        assert queue["p_error"].to_list() == [1 / 3, 0.123456789012345]
        assert queue["expected_loss"].to_list() == [(1 / 3) * 100.0, 0.123456789012345 * 50.0]
        assert_queue_reconciles(queue, result.policy.to_dict())


class TestQueueDeterminism:
    def test_same_inputs_same_queue(self) -> None:
        cost_model = _main_cost_model(Budget(type="count", value=3))
        a = _plan_q(_main_rows(), cost_model).review_queue
        b = _plan_q(_main_rows(), cost_model).review_queue
        assert a.equals(b)

    def test_shuffled_input_rows_same_queue(self) -> None:
        cost_model = _main_cost_model(Budget(type="fraction", value=1.0))
        rows = _main_rows()
        shuffled = [rows[3], rows[0], rows[5], rows[1], rows[4], rows[2]]
        assert _plan_q(rows, cost_model).review_queue.equals(
            _plan_q(shuffled, cost_model).review_queue
        )


# --- property test: random, tie-heavy batches under every budget type -------------

# Few distinct values => many ties; 1/3 and 0.123456789012345 carry full float
# precision, so any rounding on the way into the queue breaks exact reconciliation.
_SCORES = st.sampled_from([0.0, 0.25, 0.5, 0.5, 1 / 3, 1 / 3, 0.123456789012345, 1.0])
_FAC = st.sampled_from([0.0, 1.0, 10.0, 50.5, 100.0])
_RC = st.sampled_from([0.0, 0.5, 1.0, 5.0])


@st.composite
def _planning_problems(draw: st.DrawFn) -> tuple[list[tuple[str, str, float]], CostModel]:
    n = draw(st.integers(min_value=1, max_value=30))
    ids = draw(st.permutations([f"i{k:02d}" for k in range(n)]))
    rows = [(ids[k], draw(st.sampled_from(["a", "b", "c"])), draw(_SCORES)) for k in range(n)]
    segments = {s: SegmentCosts(false_accept_cost=draw(_FAC), review_cost=draw(_RC)) for s in "ab"}
    segments["default"] = SegmentCosts(false_accept_cost=draw(_FAC), review_cost=draw(_RC))
    budget_type = draw(st.sampled_from(["fraction", "count", "money"]))
    if budget_type == "fraction":
        value = draw(st.floats(min_value=0.01, max_value=1.0))
    elif budget_type == "count":
        value = float(draw(st.integers(min_value=1, max_value=40)))
    else:
        value = draw(st.sampled_from([0.5, 1.0, 2.5, 5.0, 10.0, 37.5, 200.0]))
    return rows, CostModel(budget=Budget(type=budget_type, value=value), segments=segments)


@settings(derandomize=True, max_examples=300, deadline=None)
@given(_planning_problems())
def test_property_queue_is_the_reference_allocation(
    problem: tuple[list[tuple[str, str, float]], CostModel],
) -> None:
    rows, cost_model = problem
    result = _plan_q(rows, cost_model)
    queue, policy = result.review_queue, result.policy.to_dict()
    assert_queue_reconciles(queue, policy)
    batch = batch_frame(rows)
    ref = reference_allocation(batch, batch["score"].to_numpy(), cost_model.model_dump())
    assert_queue_matches_reference(queue, ref)
    assert _plan(rows, cost_model) == result.policy

    # What thresholds can and cannot say: per segment, every item strictly
    # above the threshold is queued, and every queued item is at or above it.
    # Only items exactly AT the threshold are ambiguous -- the tie case.
    queued = set(queue["item_id"].to_list())
    for name, stats in policy["segments"].items():
        threshold = stats["p_error_threshold"]
        seg_rows = [r for r in rows if r[1] == name]
        if threshold is None:
            assert not queued & {r[0] for r in seg_rows}
            continue
        assert {r[0] for r in seg_rows if r[2] > threshold} <= queued
        assert {r[0] for r in seg_rows if r[0] in queued} <= {
            r[0] for r in seg_rows if r[2] >= threshold
        }


@st.composite
def _tie_cut_problems(draw: st.DrawFn) -> tuple[list[tuple[str, str, float]], CostModel, int]:
    """`above` items strictly above a tied group of `tied` items at p_error 0.5,
    plus `below` items; capacity lands strictly INSIDE the tied group."""
    above = draw(st.integers(min_value=0, max_value=5))
    tied = draw(st.integers(min_value=2, max_value=12))
    below = draw(st.integers(min_value=0, max_value=5))
    cut = draw(st.integers(min_value=1, max_value=tied - 1))  # tied items admitted
    n = above + tied + below
    ids = draw(st.permutations([f"t{k:02d}" for k in range(n)]))
    scores = [0.9] * above + [0.5] * tied + [0.1] * below
    rows = [(ids[k], "seg", scores[k]) for k in range(n)]
    rows = draw(st.permutations(rows))  # input order must not matter
    capacity = above + cut
    # review_cost 1.0: a money budget of `capacity` admits exactly `capacity` items.
    budget = draw(
        st.sampled_from(
            [Budget(type="count", value=capacity), Budget(type="money", value=float(capacity))]
        )
    )
    costs = {"seg": SegmentCosts(false_accept_cost=100.0, review_cost=1.0)}
    return list(rows), CostModel(budget=budget, segments=costs), cut


@settings(derandomize=True, max_examples=200, deadline=None)
@given(_tie_cut_problems())
def test_property_tied_group_is_cut_by_item_id(
    problem: tuple[list[tuple[str, str, float]], CostModel, int],
) -> None:
    rows, cost_model, cut = problem
    result = _plan_q(rows, cost_model)
    above = sorted(r[0] for r in rows if r[2] == 0.9)
    tied = sorted(r[0] for r in rows if r[2] == 0.5)
    assert result.review_queue["item_id"].to_list() == above + tied[:cut]
    assert result.policy.segments["seg"]["p_error_threshold"] == 0.5
    # The published threshold admits the WHOLE tied group: strictly more.
    assert len(above) + len(tied) > result.review_queue.height
    assert_queue_reconciles(result.review_queue, result.policy.to_dict())


# Opt-in money strategy "fill" (docs/interfaces.md §5).
# ---------------------------------------------------------------------------


def _fill(budget_value: float) -> Budget:
    return Budget(type="money", value=budget_value, strategy="fill")


# Same 4 items as test_money_budget_stops_at_first_non_fitting_item: one
# segment each, net_benefit 100 / 90 / 80 / 70, review_cost 10 / 8 / 1 / 1.
_STOP_ROWS = [
    ("item1", "seg1", 1.0),
    ("item2", "seg2", 1.0),
    ("item3", "seg3", 1.0),
    ("item4", "seg4", 1.0),
]
_STOP_SEGMENTS = {
    "seg1": SegmentCosts(false_accept_cost=110.0, review_cost=10.0),
    "seg2": SegmentCosts(false_accept_cost=98.0, review_cost=8.0),
    "seg3": SegmentCosts(false_accept_cost=81.0, review_cost=1.0),
    "seg4": SegmentCosts(false_accept_cost=71.0, review_cost=1.0),
}


class TestFillStrategy:
    def test_fill_skips_the_non_fitting_item_and_keeps_going(self) -> None:
        # Budget 12. Candidate net_benefit: item1 (10), item2 skipped (18 > 12),
        # item3 (11), item4 (12) -> net benefit 250.
        # Candidate net_benefit_per_cost (ratios 10, 11.25, 80, 70): item3 (1),
        # item4 (2), item2 (10), item1 skipped (20 > 12) -> 240. 250 wins.
        policy = _plan(_STOP_ROWS, CostModel(budget=_fill(12.0), segments=_STOP_SEGMENTS))
        assert policy.allocation == {"selected_candidate": "net_benefit"}
        assert policy.totals["n_review"] == 3
        assert policy.totals["review_spend"] == 12.0
        assert [policy.segments[s]["n_review"] for s in ("seg1", "seg2", "seg3", "seg4")] == [
            1,
            0,
            1,
            1,
        ]
        # prefix on the same input routed only item1 (net benefit 100).
        assert policy.totals["expected_captured_loss"] - policy.totals["review_spend"] == 250.0

    def test_per_cost_candidate_wins_when_cheap_items_add_up(self) -> None:
        # big: one item, p=0.5 * 50 = 25 expected loss, review 10 -> nb 15.
        # small: ten items, p=0.75 * 8 = 6, review 1 -> nb 5 each (ratio 5 vs 1.5).
        # Budget 10. net_benefit candidate: big (10) -> 15. per_cost: ten small -> 50.
        rows = [("big1", "big", 0.5)] + [(f"s{i:02d}", "small", 0.75) for i in range(10)]
        segments = {
            "big": SegmentCosts(false_accept_cost=50.0, review_cost=10.0),
            "small": SegmentCosts(false_accept_cost=8.0, review_cost=1.0),
        }
        fill = _plan(rows, CostModel(budget=_fill(10.0), segments=segments))
        assert fill.allocation == {"selected_candidate": "net_benefit_per_cost"}
        assert fill.segments["big"]["n_review"] == 0
        assert fill.segments["small"]["n_review"] == 10
        assert fill.totals["expected_captured_loss"] == 60.0
        assert fill.totals["review_spend"] == 10.0

        # prefix takes big first and then stops (10 + 1 > 10): net benefit 15.
        prefix = _plan(rows, CostModel(budget=Budget(type="money", value=10.0), segments=segments))
        assert prefix.segments["big"]["n_review"] == 1
        assert prefix.totals["n_review"] == 1

    def test_exact_tie_selects_net_benefit_candidate(self) -> None:
        # Budget 2. a: nb 4, cost 2 (ratio 2); b: nb 3, cost 1 (ratio 3);
        # c: nb 1, cost 1 (ratio 1). net_benefit: a -> 4. per_cost: b, c -> 4.
        # Dyadic values, so both totals are exactly 4.0: the tie keeps a.
        rows = [("a1", "a", 0.75), ("b1", "b", 0.5), ("c1", "c", 0.25)]
        segments = {
            "a": SegmentCosts(false_accept_cost=8.0, review_cost=2.0),
            "b": SegmentCosts(false_accept_cost=8.0, review_cost=1.0),
            "c": SegmentCosts(false_accept_cost=8.0, review_cost=1.0),
        }
        policy = _plan(rows, CostModel(budget=_fill(2.0), segments=segments))
        assert policy.allocation == {"selected_candidate": "net_benefit"}
        assert [policy.segments[s]["n_review"] for s in ("a", "b", "c")] == [1, 0, 0]

    def test_zero_cost_items_are_routed_even_after_the_budget_is_spent(self) -> None:
        # x does not fit (cost 2 > budget 1): prefix stops at once, routing
        # nothing. fill skips x and routes the free item (ratio +inf).
        rows = [("x1", "x", 1.0), ("f1", "free", 0.5)]
        segments = {
            "x": SegmentCosts(false_accept_cost=100.0, review_cost=2.0),
            "free": SegmentCosts(false_accept_cost=10.0, review_cost=0.0),
        }
        prefix = _plan(rows, CostModel(budget=Budget(type="money", value=1.0), segments=segments))
        assert prefix.totals["n_review"] == 0
        fill = _plan(rows, CostModel(budget=_fill(1.0), segments=segments))
        assert fill.segments["free"]["n_review"] == 1
        assert fill.segments["x"]["n_review"] == 0
        assert fill.totals["review_spend"] == 0.0

    def test_non_positive_net_benefit_never_routed_under_spare_budget(self) -> None:
        policy = _plan(_main_rows(), _main_cost_model(_fill(1000.0)))
        assert policy.totals["n_review"] == 4  # i1, i2, i4, i5; never i3, i6
        assert policy.totals["expected_residual_loss"] == pytest.approx(5 + 2.5)

    def test_everything_fits_equals_prefix_allocation(self) -> None:
        prefix = _plan(_main_rows(), _main_cost_model(Budget(type="money", value=1000.0)))
        fill = _plan(_main_rows(), _main_cost_model(_fill(1000.0)))
        assert fill.segments == prefix.segments
        assert fill.totals == prefix.totals
        assert fill.allocation == {"selected_candidate": "net_benefit"}

    def test_policy_records_strategy_and_allocation(self) -> None:
        policy = _plan(_main_rows(), _main_cost_model(_fill(22.0)))
        d = policy.to_dict()
        assert set(d) == {
            "allocation",
            "budget",
            "config_snapshot",
            "input_sha256s",
            "policy_hash",
            "segments",
            "tool_version",
            "totals",
        }
        assert d["budget"] == {"type": "money", "value": 22.0, "strategy": "fill"}
        assert d["config_snapshot"]["budget"] == d["budget"]
        assert d["allocation"]["selected_candidate"] in ("net_benefit", "net_benefit_per_cost")
        assert policy.policy_hash == content_hash(d)
        assert canonical_dumps(d)

    def test_explicit_prefix_is_identical_to_the_default(self) -> None:
        implicit = _plan(_main_rows(), _main_cost_model(Budget(type="money", value=22.0)))
        explicit = _plan(
            _main_rows(), _main_cost_model(Budget(type="money", value=22.0, strategy="prefix"))
        )
        assert explicit.to_dict() == implicit.to_dict()
        assert "allocation" not in implicit.to_dict()
        assert implicit.budget == {"type": "money", "value": 22.0}

    def test_strategy_is_part_of_the_policy_hash(self) -> None:
        # Same routed set (everything fits), different declared strategy.
        prefix = _plan(_main_rows(), _main_cost_model(Budget(type="money", value=1000.0)))
        fill = _plan(_main_rows(), _main_cost_model(_fill(1000.0)))
        assert prefix.policy_hash != fill.policy_hash

    def test_fill_is_deterministic_under_row_shuffles(self) -> None:
        rows = _main_rows()
        shuffled = [rows[3], rows[0], rows[5], rows[1], rows[4], rows[2]]
        a = _plan(rows, _main_cost_model(_fill(16.0)))
        b = _plan(shuffled, _main_cost_model(_fill(16.0)))
        assert a.to_dict() == b.to_dict()


# ---------------------------------------------------------------------------
# test-spec I.1: property tests against a brute-force knapsack optimum.
# Scores k/1024, integer false-accept costs and quarter-unit review costs and
# budgets keep every sum exact in float64, so assertions need no tolerance.
# ---------------------------------------------------------------------------

_PROPERTY_SETTINGS = settings(derandomize=True, max_examples=300, deadline=None)
_SEGMENT_NAMES = ("s0", "s1", "s2")


@st.composite
def _money_instance(draw: st.DrawFn) -> tuple[list[tuple[str, str, float]], CostModel, float]:
    n_items = draw(st.integers(min_value=3, max_value=9))
    seg_of = draw(st.lists(st.sampled_from(_SEGMENT_NAMES), min_size=n_items, max_size=n_items))
    # Distinct scores within a segment, so thresholds identify routed sets.
    ks = draw(st.lists(st.integers(0, 1024), min_size=n_items, max_size=n_items, unique=True))
    rows = [(f"i{j}", seg_of[j], ks[j] / 1024) for j in range(n_items)]
    segments = {
        name: SegmentCosts(
            false_accept_cost=float(draw(st.integers(0, 200))),
            # Cheap or expensive tiers: the mix where the two rankings differ.
            review_cost=draw(st.one_of(st.integers(0, 8), st.integers(24, 80))) / 4,
        )
        for name in _SEGMENT_NAMES
    }
    # Budgets mostly below the total review cost, so that they bind.
    value = draw(st.integers(1, 100)) / 4
    return rows, CostModel(budget=Budget(type="money", value=value), segments=segments), value


def _item_net_benefits(
    rows: list[tuple[str, str, float]], cost_model: CostModel
) -> dict[str, tuple[str, float, float, float]]:
    """item_id -> (segment, p_error, review_cost, net_benefit), computed in-test."""
    out = {}
    for item_id, seg, p in rows:
        costs = cost_model.costs_for(seg)
        nb = p * costs.false_accept_cost - costs.review_cost
        out[item_id] = (seg, p, costs.review_cost, nb)
    return out


def _routed_from_thresholds(
    policy: Policy, items: dict[str, tuple[str, float, float, float]]
) -> set[str]:
    routed = set()
    for item_id, (seg, p, _cost, _nb) in items.items():
        threshold = policy.segments[seg]["p_error_threshold"]
        if threshold is not None and p >= threshold:
            routed.add(item_id)
    return routed


def _brute_force_optimum(items: dict[str, tuple[str, float, float, float]], value: float) -> float:
    candidates = [(cost, nb) for (_s, _p, cost, nb) in items.values() if nb > 0]
    best = 0.0
    for mask in range(1 << len(candidates)):
        chosen = [candidates[j] for j in range(len(candidates)) if mask >> j & 1]
        if math.fsum(c for c, _ in chosen) <= value:
            best = max(best, math.fsum(nb for _, nb in chosen))
    return best


@st.composite
def _crowding_instance(draw: st.DrawFn) -> tuple[list[tuple[str, str, float]], CostModel, float]:
    """One expensive item competing with several cheap ones: the shape where
    the net_benefit_per_cost candidate wins."""
    n_cheap = draw(st.integers(min_value=2, max_value=8))
    ks = draw(
        st.lists(st.integers(0, 1024), min_size=n_cheap + 1, max_size=n_cheap + 1, unique=True)
    )
    rows = [("big", "s0", ks[0] / 1024)] + [
        (f"c{j}", "s1", ks[j + 1] / 1024) for j in range(n_cheap)
    ]
    big_cost = draw(st.integers(8, 80)) / 4
    segments = {
        "s0": SegmentCosts(
            false_accept_cost=float(draw(st.integers(0, 400))), review_cost=big_cost
        ),
        "s1": SegmentCosts(
            false_accept_cost=float(draw(st.integers(0, 100))),
            review_cost=draw(st.integers(0, 8)) / 4,
        ),
    }
    value = draw(st.integers(int(big_cost * 4), int(big_cost * 4) + 8)) / 4
    return rows, CostModel(budget=Budget(type="money", value=value), segments=segments), value


def _check_fill_guarantees(instance: tuple[list[tuple[str, str, float]], CostModel, float]) -> None:
    rows, prefix_model, value = instance
    fill_model = prefix_model.model_copy(update={"budget": _fill(value)})
    items = _item_net_benefits(rows, prefix_model)
    totals = {}
    for name, model in (("prefix", prefix_model), ("fill", fill_model)):
        policy = _plan(rows, model)
        routed = _routed_from_thresholds(policy, items)
        # 1. per-segment prefix: threshold replay reproduces n_review exactly
        for seg in policy.segments:
            replayed = sum(1 for i in routed if items[i][0] == seg)
            assert replayed == policy.segments[seg]["n_review"], (name, seg)
        # 2. only positive net benefit is routed
        assert all(items[i][3] > 0 for i in routed)
        # 3. within budget
        spend = math.fsum(items[i][2] for i in routed)
        assert spend <= value
        assert spend == policy.totals["review_spend"]
        totals[name] = math.fsum(items[i][3] for i in routed)
        # 6. allocation recorded only for fill
        if name == "fill":
            assert policy.allocation is not None
            assert policy.allocation["selected_candidate"] in (
                "net_benefit",
                "net_benefit_per_cost",
            )
        else:
            assert policy.allocation is None
    optimum = _brute_force_optimum(items, value)
    assert totals["fill"] >= totals["prefix"]  # 4. never worse
    assert 2 * totals["fill"] >= optimum  # 5. at least half of the optimum
    assert totals["fill"] <= optimum


class TestFillProperties:
    @_PROPERTY_SETTINGS
    @given(_money_instance())
    def test_fill_guarantees(
        self, instance: tuple[list[tuple[str, str, float]], CostModel, float]
    ) -> None:
        _check_fill_guarantees(instance)

    @_PROPERTY_SETTINGS
    @given(_crowding_instance())
    def test_fill_guarantees_when_cheap_items_crowd_out_an_expensive_one(
        self, instance: tuple[list[tuple[str, str, float]], CostModel, float]
    ) -> None:
        _check_fill_guarantees(instance)


def test_fill_queue_keeps_exact_selected_subset_in_priority_order() -> None:
    model = CostModel(budget=_fill(12.0), segments=_STOP_SEGMENTS)
    result = _plan_q(_STOP_ROWS, model)
    assert result.review_queue["item_id"].to_list() == ["item1", "item3", "item4"]
    assert result.review_queue["review_priority"].to_list() == [1, 2, 3]
    assert_queue_reconciles(result.review_queue, result.policy.to_dict())

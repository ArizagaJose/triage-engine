"""Backtester (docs/interfaces.md §9), including the capacity-sweep curves
(§9.11). IdentityCalibrator (p_error = score) and handcrafted frames keep the
expected numbers exact and calibration out of scope."""

import math

import numpy as np
import pytest

from tests.support import IdentityCalibrator, history_frame
from triage.artifacts import content_hash
from triage.backtest import (
    BacktestResult,
    check_budget_overrun_gate,
    check_realized_residual_gate,
    run,
    validate_max_budget_overrun,
    validate_n_random_draws,
)
from triage.costs import Budget, CostModel, Gates, SegmentCosts
from triage.exceptions import BudgetOverrunExceeded, ContractViolation, ResidualRiskExceeded

CALIBRATOR = IdentityCalibrator()


def _cost_model_dict() -> dict:
    return {
        "budget": {"type": "fraction", "value": 0.10},
        "gates": {},
        "segments": {
            "segA": {"false_accept_cost": 100.0, "review_cost": 10.0},
            "segB": {"false_accept_cost": 50.0, "review_cost": 5.0},
            "segC": {"false_accept_cost": 20.0, "review_cost": 2.0},
        },
    }


def _policy_dict(*, segments: dict, config_snapshot: dict | None = None) -> dict:
    """Build a self-consistent policy dict: policy_hash = content_hash(rest)."""
    d = {
        "config_snapshot": config_snapshot if config_snapshot is not None else _cost_model_dict(),
        "segments": segments,
        "totals": {
            "expected_captured_loss": 123.0,
            "expected_residual_loss": 45.0,
        },
    }
    d["policy_hash"] = content_hash(d)
    return d


# ---------------------------------------------------------------------------
# Main fixture: 5 items across 3 segments.
#
# segA: false_accept_cost=100, review_cost=10; threshold=0.5
#   a1 p_error=0.50 (TIE at threshold) error       -> true_loss=100, reviewed
#   a2 p_error=0.49 (below threshold)  no error    -> true_loss=0,   not reviewed
#   a3 p_error=0.51 (above threshold)  error       -> true_loss=100, reviewed
# segB: false_accept_cost=50, review_cost=5; threshold=None (never reviewed)
#   b1 p_error=0.90                    error       -> true_loss=50,  not reviewed
# segC: unseen at plan time (absent from policy.segments)
#   c1 p_error=0.99                    error       -> true_loss=20,  not reviewed
#
# n_review (policy) = 2 (a1, a3); total_true_loss = 100+0+100+50+20 = 270
# ---------------------------------------------------------------------------
def _main_rows() -> list[tuple[str, str, float, str, str]]:
    return [
        ("a1", "segA", 0.50, "x", "v"),
        ("a2", "segA", 0.49, "v", "v"),
        ("a3", "segA", 0.51, "x", "v"),
        ("b1", "segB", 0.90, "x", "v"),
        ("c1", "segC", 0.99, "x", "v"),
    ]


def _main_policy_dict() -> dict:
    return _policy_dict(
        segments={
            "segA": {"p_error_threshold": 0.5},
            "segB": {"p_error_threshold": None},
        }
    )


def _run_main(**kwargs) -> BacktestResult:
    return run(
        _main_policy_dict(),
        CALIBRATOR,
        history_frame(_main_rows()),
        seed=kwargs.pop("seed", 42),
        n_random_draws=kwargs.pop("n_random_draws", 50),
        input_sha256s={"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64},
        tool_version="0.1.0",
        **kwargs,
    )


class TestThresholdReplay:
    def test_tie_at_threshold_is_reviewed(self) -> None:
        result = _run_main()
        policy = result.strategies["policy"]
        # a1 (exactly at threshold) and a3 (above) reviewed; a2 (below) is not.
        assert policy["n_review"] == 2
        assert policy["segments"]["segA"]["n_review"] == 2
        assert policy["segments"]["segA"]["captured_loss"] == pytest.approx(200.0)

    def test_null_threshold_segment_never_reviewed(self) -> None:
        result = _run_main()
        policy = result.strategies["policy"]
        assert policy["segments"]["segB"]["n_review"] == 0
        assert policy["segments"]["segB"]["p_error_threshold"] is None
        assert policy["segments"]["segB"]["residual_loss"] == pytest.approx(50.0)

    def test_unseen_segment_never_reviewed(self) -> None:
        result = _run_main()
        policy = result.strategies["policy"]
        # segC absent from policy.segments -> appears with null threshold, never reviewed.
        assert "segC" in policy["segments"]
        assert policy["segments"]["segC"]["p_error_threshold"] is None
        assert policy["segments"]["segC"]["n_review"] == 0
        assert policy["segments"]["segC"]["residual_loss"] == pytest.approx(20.0)

    def test_total_true_loss(self) -> None:
        result = _run_main()
        assert result.total_true_loss == pytest.approx(270.0)
        assert result.n_errors == 4  # a1, a3, b1, c1 (a2 is not an error)


class TestBaselinesAtEqualCapacity:
    def test_raw_confidence_reviews_same_count_as_policy(self) -> None:
        result = _run_main()
        assert (
            result.strategies["raw_confidence"]["n_review"]
            == result.strategies["policy"]["n_review"]
        )

    def test_random_reviews_same_count_as_policy(self) -> None:
        result = _run_main()
        assert result.strategies["random"]["n_review"] == result.strategies["policy"]["n_review"]

    def test_raw_confidence_ranks_by_score_ascending(self) -> None:
        # n_review=2; ascending scores: a2(0.49), a1(0.50), a3(0.51), b1(0.90), c1(0.99).
        # First two reviewed: a2 (true_loss=0), a1 (true_loss=100) -> captured=100.
        result = _run_main()
        raw = result.strategies["raw_confidence"]
        assert raw["captured_loss"] == pytest.approx(100.0)
        assert raw["residual_loss"] == pytest.approx(270.0 - 100.0)


class TestBudgetEdgeCases:
    def test_n_review_zero_thresholds_all_null(self) -> None:
        policy_dict = _policy_dict(
            segments={
                "segA": {"p_error_threshold": None},
                "segB": {"p_error_threshold": None},
            }
        )
        result = run(
            policy_dict,
            CALIBRATOR,
            history_frame(_main_rows()),
            seed=42,
            n_random_draws=10,
            input_sha256s={"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64},
            tool_version="0.1.0",
        )
        policy = result.strategies["policy"]
        raw = result.strategies["raw_confidence"]
        random = result.strategies["random"]
        assert policy["n_review"] == 0
        assert policy["efficiency"] is None
        assert policy["captured_loss"] == 0.0
        assert raw["n_review"] == 0
        assert raw["efficiency"] is None
        # Random baseline is degenerate (n_review=0 draws) but still defined, not null.
        assert random["n_review"] == 0
        assert random["captured_loss_mean"] == 0.0
        assert random["captured_loss_std"] == 0.0
        assert random["efficiency"] is None
        assert random["capture_rate"] == pytest.approx(0.0)

    def test_review_everything_within_seen_segments(self) -> None:
        # Threshold 0.0 admits every item whose p_error >= 0, i.e. all of segA/segB.
        # segC stays excluded (unseen).
        policy_dict = _policy_dict(
            segments={
                "segA": {"p_error_threshold": 0.0},
                "segB": {"p_error_threshold": 0.0},
            }
        )
        result = run(
            policy_dict,
            CALIBRATOR,
            history_frame(_main_rows()),
            seed=42,
            n_random_draws=10,
            input_sha256s={"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64},
            tool_version="0.1.0",
        )
        policy = result.strategies["policy"]
        assert policy["n_review"] == 4  # a1, a2, a3, b1 (not c1)
        assert policy["captured_loss"] == pytest.approx(100.0 + 0.0 + 100.0 + 50.0)
        assert policy["residual_loss"] == pytest.approx(20.0)  # only c1


class TestPolicySelfHashCheck:
    def test_tampered_policy_hash_raises(self) -> None:
        policy_dict = _main_policy_dict()
        policy_dict["policy_hash"] = "0" * 64  # wrong
        with pytest.raises(ContractViolation, match="self-hash mismatch"):
            run(
                policy_dict,
                CALIBRATOR,
                history_frame(_main_rows()),
                seed=42,
                n_random_draws=10,
                input_sha256s={"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64},
                tool_version="0.1.0",
            )

    def test_missing_policy_hash_key_raises(self) -> None:
        policy_dict = _main_policy_dict()
        del policy_dict["policy_hash"]
        with pytest.raises(ContractViolation, match="policy_hash"):
            run(
                policy_dict,
                CALIBRATOR,
                history_frame(_main_rows()),
                seed=42,
                n_random_draws=10,
                input_sha256s={"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64},
                tool_version="0.1.0",
            )


class TestMalformedConfigSnapshot:
    def test_invalid_config_snapshot_raises_contract_violation(self) -> None:
        bad_config = _cost_model_dict()
        bad_config["budget"]["value"] = -1.0  # violates > 0 validator
        policy_dict = _policy_dict(
            segments={"segA": {"p_error_threshold": 0.5}}, config_snapshot=bad_config
        )
        with pytest.raises(ContractViolation):
            run(
                policy_dict,
                CALIBRATOR,
                history_frame(_main_rows()),
                seed=42,
                n_random_draws=10,
                input_sha256s={"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64},
                tool_version="0.1.0",
            )

    def test_extra_field_in_config_snapshot_raises(self) -> None:
        bad_config = _cost_model_dict()
        bad_config["not_a_real_key"] = 1
        policy_dict = _policy_dict(
            segments={"segA": {"p_error_threshold": 0.5}}, config_snapshot=bad_config
        )
        with pytest.raises(ContractViolation):
            run(
                policy_dict,
                CALIBRATOR,
                history_frame(_main_rows()),
                seed=42,
                n_random_draws=10,
                input_sha256s={"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64},
                tool_version="0.1.0",
            )


class TestAccountingIdentities:
    def test_captured_plus_residual_equals_total_for_every_strategy(self) -> None:
        result = _run_main()
        total = result.total_true_loss
        for name in ("policy", "raw_confidence"):
            s = result.strategies[name]
            assert s["captured_loss"] + s["residual_loss"] == pytest.approx(total)
        random = result.strategies["random"]
        assert random["captured_loss_mean"] + random["residual_loss_mean"] == pytest.approx(total)

    def test_policy_captured_loss_is_sequential_sum_of_segments(self) -> None:
        result = _run_main()
        policy = result.strategies["policy"]
        seg_sum = sum(s["captured_loss"] for s in policy["segments"].values())
        assert policy["captured_loss"] == seg_sum  # exact, not approx: same construction

    def test_per_segment_captured_and_residual_are_fsums(self) -> None:
        # segA: reviewed {a1: 100, a3: 100} -> fsum 200; not-reviewed {a2: 0} -> fsum 0.
        result = _run_main()
        segA = result.strategies["policy"]["segments"]["segA"]
        assert segA["captured_loss"] == math.fsum([100.0, 100.0])
        assert segA["residual_loss"] == math.fsum([0.0])


class TestRandomBaselineDeterminism:
    def test_same_seed_reproduces_exact_stats(self) -> None:
        r1 = _run_main(seed=7, n_random_draws=30)
        r2 = _run_main(seed=7, n_random_draws=30)
        assert r1.strategies["random"] == r2.strategies["random"]

    def test_different_seed_changes_stats(self) -> None:
        r1 = _run_main(seed=1, n_random_draws=30)
        r2 = _run_main(seed=2, n_random_draws=30)
        assert r1.strategies["random"] != r2.strategies["random"]

    def test_std_uses_ddof_0(self) -> None:
        # Cross-check against a manual population-std computation on the same draws.
        policy_dict = _main_policy_dict()
        history = history_frame(_main_rows())
        rng = np.random.default_rng(99)
        n_review = 2
        n_history = 5
        true_loss = {"a1": 100.0, "a2": 0.0, "a3": 100.0, "b1": 50.0, "c1": 20.0}
        ids = ["a1", "a2", "a3", "b1", "c1"]
        draws = []
        for _ in range(20):
            idx = rng.choice(n_history, size=n_review, replace=False)
            draws.append(sum(true_loss[ids[i]] for i in idx))
        expected_std = float(np.std(np.array(draws), ddof=0))

        result = run(
            policy_dict,
            CALIBRATOR,
            history,
            seed=99,
            n_random_draws=20,
            input_sha256s={"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64},
            tool_version="0.1.0",
        )
        assert result.strategies["random"]["captured_loss_std"] == pytest.approx(expected_std)


class TestResidualGate:
    def test_gate_trips_when_realized_residual_exceeds_max(self) -> None:
        result = _run_main()
        residual = result.strategies["policy"]["residual_loss"]
        cost_model = CostModel(
            budget=Budget(type="fraction", value=0.1),
            gates=Gates(max_residual_risk=residual - 1.0),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        with pytest.raises(ResidualRiskExceeded):
            check_realized_residual_gate(result, cost_model)

    def test_no_gate_when_max_residual_risk_is_none(self) -> None:
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="fraction", value=0.1),
            gates=Gates(max_residual_risk=None),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        check_realized_residual_gate(result, cost_model)  # does not raise

    def test_no_gate_when_residual_below_max(self) -> None:
        result = _run_main()
        residual = result.strategies["policy"]["residual_loss"]
        cost_model = CostModel(
            budget=Budget(type="fraction", value=0.1),
            gates=Gates(max_residual_risk=residual + 1.0),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        check_realized_residual_gate(result, cost_model)  # does not raise


class TestBudgetOverrunGate:
    def test_none_is_noop(self) -> None:
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="fraction", value=0.0001),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        check_budget_overrun_gate(result, cost_model, None)  # does not raise

    def test_factor_below_one_raises_contract_violation(self) -> None:
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="fraction", value=0.1),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        with pytest.raises(ContractViolation):
            check_budget_overrun_gate(result, cost_model, 0.5)

    @pytest.mark.parametrize("factor", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_factor_raises_contract_violation(self, factor: float) -> None:
        # nan < 1.0 is False and `actual > nan` / `actual > inf` are False,
        # so without an explicit isfinite check the gate would be silently
        # disabled rather than rejected.
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="fraction", value=0.1),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        with pytest.raises(ContractViolation, match="finite"):
            check_budget_overrun_gate(result, cost_model, factor)

    def test_fraction_budget_raises_just_above_factor(self) -> None:
        # n_history=5, n_review=2; value=0.2 -> allowed=1.0; factor=1.9 -> threshold=1.9 < 2.
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="fraction", value=0.2),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        with pytest.raises(BudgetOverrunExceeded):
            check_budget_overrun_gate(result, cost_model, 1.9)

    def test_fraction_budget_passes_at_exact_factor(self) -> None:
        # value=0.2 -> allowed=1.0; factor=2.0 -> threshold=2.0 == actual (strict >, so passes).
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="fraction", value=0.2),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        check_budget_overrun_gate(result, cost_model, 2.0)  # does not raise

    def test_count_budget_raises_when_over(self) -> None:
        # value=1 -> allowed=1; factor=1.9 -> threshold=1.9 < actual n_review=2.
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="count", value=1),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        with pytest.raises(BudgetOverrunExceeded):
            check_budget_overrun_gate(result, cost_model, 1.9)

    def test_count_budget_passes_within_factor(self) -> None:
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="count", value=1),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        check_budget_overrun_gate(result, cost_model, 2.0)  # does not raise

    def test_money_budget_compares_review_spend(self) -> None:
        # review_spend = 20.0 (a1, a3 at review_cost=10 each); value=10 -> allowed=10;
        # factor=1.9 -> threshold=19 < 20.
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="money", value=10.0),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        with pytest.raises(BudgetOverrunExceeded):
            check_budget_overrun_gate(result, cost_model, 1.9)

    def test_money_budget_passes_at_exact_factor(self) -> None:
        result = _run_main()
        cost_model = CostModel(
            budget=Budget(type="money", value=10.0),
            segments={"segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0)},
        )
        check_budget_overrun_gate(result, cost_model, 2.0)  # does not raise


class TestToDict:
    def test_to_dict_round_trips_into_content_hash(self) -> None:
        result = _run_main()
        d = result.to_dict()
        assert "created_at" not in d
        assert d["backtest_hash"] == result.backtest_hash
        # backtest_hash must equal content_hash of the dict (created_at excluded is a no-op here).
        assert content_hash(d) == result.backtest_hash


# ---------------------------------------------------------------------------
# Option validation (shared with `triage run`, see tests/test_run_parity.py)
# ---------------------------------------------------------------------------


class TestOptionValidation:
    @pytest.mark.parametrize("n", [-1, -200])
    def test_negative_n_random_draws_is_contract_violation(self, n: int) -> None:
        """Used to crash inside numpy (`np.empty(-1)`) -> exit 4."""
        with pytest.raises(ContractViolation, match="n_random_draws must be an integer >= 0"):
            _run_main(n_random_draws=n)

    def test_zero_n_random_draws_is_valid(self) -> None:
        result = _run_main(n_random_draws=0)
        assert result.strategies["random"]["captured_loss_mean"] == 0.0

    @pytest.mark.parametrize("n", [1.5, True, "3"])
    def test_non_integer_n_random_draws_rejected(self, n: object) -> None:
        with pytest.raises(ContractViolation):
            validate_n_random_draws(n)  # type: ignore[arg-type]

    @pytest.mark.parametrize("seed", [-1, True, 1.0])
    def test_invalid_seed_is_contract_violation(self, seed: object) -> None:
        with pytest.raises(ContractViolation, match="seed must be a non-negative integer"):
            _run_main(seed=seed)

    @pytest.mark.parametrize("factor", [0.99, -1.0, math.nan, math.inf, -math.inf])
    def test_validate_max_budget_overrun_rejects(self, factor: float) -> None:
        with pytest.raises(ContractViolation, match="finite float >= 1.0"):
            validate_max_budget_overrun(factor)

    @pytest.mark.parametrize("factor", [1.0, 1.5, 100.0])
    def test_validate_max_budget_overrun_accepts(self, factor: float) -> None:
        validate_max_budget_overrun(factor)


# ---------------------------------------------------------------------------
# Capacity-sweep curves (docs/interfaces.md §9.11). Own fixture data: two
# segments, 6 items, values hand-computed below.
# ---------------------------------------------------------------------------


def _curves_cost_model_dict() -> dict:
    return {
        "budget": {"type": "fraction", "value": 0.10},
        "gates": {},
        "segments": {
            "segA": {"false_accept_cost": 100.0, "review_cost": 10.0},
            "segB": {"false_accept_cost": 50.0, "review_cost": 5.0},
        },
    }


def _curves_policy_dict(*, segments: dict) -> dict:
    d = {
        "config_snapshot": _curves_cost_model_dict(),
        "segments": segments,
        "totals": {"expected_captured_loss": 0.0, "expected_residual_loss": 0.0},
    }
    d["policy_hash"] = content_hash(d)
    return d


_CURVES_INPUT_SHA = {"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64}


# ---------------------------------------------------------------------------
# Main fixture: 6 items across segA (fa=100, review=10) / segB (fa=50, review=5).
#
#   item  seg   score  p_error  predicted/true  true_loss  expected_loss  net_benefit
#   i1    segA  0.90   0.90     x/v (error)      100          90            80
#   i2    segA  0.10   0.10     v/v (no error)      0          10             0
#   i3    segB  0.80   0.80     x/v (error)       50          40            35
#   i4    segB  0.20   0.20     v/v (no error)      0          10             5
#   i5    segA  0.50   0.50     x/v (error)      100          50            40
#   i6    segB  0.60   0.60     v/v (no error)      0          30            25
#
# total_true_loss = 100+0+50+0+100+0 = 250, n_history=6.
#
# Policy ranking (net_benefit DESC, item_id ASC tie-break):
#   i1(80), i5(40), i3(35), i6(25), i4(5), i2(0)
# Raw-confidence ranking (score ASC, item_id ASC tie-break):
#   i2(0.10), i4(0.20), i5(0.50), i6(0.60), i3(0.80), i1(0.90)
# ---------------------------------------------------------------------------
def _curves_rows() -> list[tuple[str, str, float, str, str]]:
    return [
        ("i1", "segA", 0.90, "x", "v"),
        ("i2", "segA", 0.10, "v", "v"),
        ("i3", "segB", 0.80, "x", "v"),
        ("i4", "segB", 0.20, "v", "v"),
        ("i5", "segA", 0.50, "x", "v"),
        ("i6", "segB", 0.60, "v", "v"),
    ]


def _curves_run(**kwargs):
    policy_dict = _curves_policy_dict(
        segments={
            "segA": {"p_error_threshold": 0.5},
            "segB": {"p_error_threshold": 0.6},
        }
    )
    return run(
        policy_dict,
        CALIBRATOR,
        history_frame(_curves_rows()),
        seed=kwargs.pop("seed", 42),
        n_random_draws=kwargs.pop("n_random_draws", 10),
        input_sha256s=_CURVES_INPUT_SHA,
        tool_version="0.1.0",
        **kwargs,
    )


class TestGridPinned:
    def test_grid_is_exactly_pinned(self) -> None:
        result = _curves_run()
        expected_grid = [i / 100 for i in range(51)]
        assert result.curves["grid"] == expected_grid

    def test_grid_length_is_51(self) -> None:
        result = _curves_run()
        assert len(result.curves["grid"]) == 51

    def test_series_lengths_match_grid(self) -> None:
        result = _curves_run()
        curves = result.curves
        assert len(curves["policy"]) == 51
        assert len(curves["raw_confidence"]) == 51
        assert len(curves["random"]) == 51


class TestMonotonicity:
    def test_policy_curve_is_nondecreasing(self) -> None:
        result = _curves_run()
        series = result.curves["policy"]
        assert all(b >= a - 1e-12 for a, b in zip(series, series[1:]))

    def test_raw_confidence_curve_is_nondecreasing(self) -> None:
        result = _curves_run()
        series = result.curves["raw_confidence"]
        assert all(b >= a - 1e-12 for a, b in zip(series, series[1:]))

    def test_random_curve_is_nondecreasing(self) -> None:
        result = _curves_run()
        series = result.curves["random"]
        assert all(b >= a - 1e-12 for a, b in zip(series, series[1:]))


class TestRandomExactness:
    def test_random_curve_equals_k_over_n_exactly(self) -> None:
        result = _curves_run()
        n_history = result.n_history
        grid = result.curves["grid"]
        expected = [math.floor(f * n_history) / n_history for f in grid]
        assert result.curves["random"] == expected


class TestHandcraftedValues:
    """k values computed by hand for n_history=6 (see fixture docstring above)."""

    def test_k_zero_endpoint_is_zero(self) -> None:
        # grid[0] = 0.00 -> k = floor(0 * 6) = 0 for all three series.
        result = _curves_run()
        assert result.curves["policy"][0] == 0.0
        assert result.curves["raw_confidence"][0] == 0.0
        assert result.curves["random"][0] == 0.0

    def test_k_one_at_f_020(self) -> None:
        # f=0.20 (grid index 20): k = floor(0.20*6) = 1.
        result = _curves_run()
        idx = 20
        assert math.floor(result.curves["grid"][idx] * 6) == 1
        # Policy top-1 = i1 (true_loss=100) -> share 100/250 = 0.4.
        assert result.curves["policy"][idx] == pytest.approx(100.0 / 250.0)
        # Raw top-1 = i2 (true_loss=0) -> share 0.
        assert result.curves["raw_confidence"][idx] == pytest.approx(0.0)
        # Random: k/n = 1/6.
        assert result.curves["random"][idx] == pytest.approx(1.0 / 6.0)

    def test_k_three_at_f_050(self) -> None:
        # f=0.50 (grid index 50, the last point): k = floor(0.50*6) = 3.
        result = _curves_run()
        idx = 50
        assert math.floor(result.curves["grid"][idx] * 6) == 3
        # Policy top-3 = i1, i5, i3 -> true_loss 100+100+50=250 -> share 1.0.
        assert result.curves["policy"][idx] == pytest.approx(1.0)
        # Raw top-3 = i2, i4, i5 -> true_loss 0+0+100=100 -> share 100/250=0.4.
        assert result.curves["raw_confidence"][idx] == pytest.approx(100.0 / 250.0)
        # Random: k/n = 3/6 = 0.5.
        assert result.curves["random"][idx] == pytest.approx(0.5)


class TestZeroCapacityAndTotalLoss:
    def test_total_true_loss_zero_gives_all_zero_policy_and_raw_curves(self) -> None:
        # No prediction errors anywhere -> total_true_loss == 0.
        rows = [
            ("z1", "segA", 0.90, "v", "v"),
            ("z2", "segA", 0.10, "v", "v"),
            ("z3", "segB", 0.80, "v", "v"),
            ("z4", "segB", 0.20, "v", "v"),
        ]
        policy_dict = _curves_policy_dict(
            segments={
                "segA": {"p_error_threshold": 0.5},
                "segB": {"p_error_threshold": 0.5},
            }
        )
        result = run(
            policy_dict,
            CALIBRATOR,
            history_frame(rows),
            seed=42,
            n_random_draws=10,
            input_sha256s=_CURVES_INPUT_SHA,
            tool_version="0.1.0",
        )
        assert result.total_true_loss == 0.0
        assert result.curves["policy"] == [0.0] * 51
        assert result.curves["raw_confidence"] == [0.0] * 51
        # Random is unaffected by total_true_loss: still the analytic k/n_history.
        n_history = result.n_history
        expected_random = [math.floor(f * n_history) / n_history for f in result.curves["grid"]]
        assert result.curves["random"] == expected_random


class TestCurvesInHashAndDict:
    def test_curves_present_in_to_dict_and_covered_by_hash(self) -> None:
        result = _curves_run()
        d = result.to_dict()
        assert "curves" in d
        assert d["curves"] == result.curves
        assert content_hash(d) == result.backtest_hash

    def test_backtest_hash_changes_if_curves_would_differ(self) -> None:
        # Sanity: curves participate in the hashed content (content_hash does
        # not special-case "curves" the way it does created_at/policy_hash).
        result = _curves_run()
        d = result.to_dict()
        tampered = dict(d)
        tampered["curves"] = dict(d["curves"])
        tampered["curves"]["policy"] = [0.0] * 51
        assert content_hash(tampered) != result.backtest_hash

"""Unit tests for capacity-sweep curves in backtest.json (docs/interfaces.md §9.11).

Same conventions as tests/test_backtest.py: a lightweight PredictsErrors stub
(IdentityCalibrator, p_error = score) and small handcrafted polars frames, so
calibrator internals stay out of scope here.
"""

import math

import numpy as np
import polars as pl
import pytest

from triage.artifacts import content_hash
from triage.backtest import run


class IdentityCalibrator:
    """Stub satisfying PredictsErrors: p_error = score, unchanged."""

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=np.float64)


CALIBRATOR = IdentityCalibrator()


def _history(rows: list[tuple[str, str, float, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "item_id": [r[0] for r in rows],
            "segment": [r[1] for r in rows],
            "score": [r[2] for r in rows],
            "predicted_value": [r[3] for r in rows],
            "true_value": [r[4] for r in rows],
        },
        schema={
            "item_id": pl.String,
            "segment": pl.String,
            "score": pl.Float64,
            "predicted_value": pl.String,
            "true_value": pl.String,
        },
    )


def _cost_model_dict() -> dict:
    return {
        "budget": {"type": "fraction", "value": 0.10},
        "gates": {},
        "segments": {
            "segA": {"false_accept_cost": 100.0, "review_cost": 10.0},
            "segB": {"false_accept_cost": 50.0, "review_cost": 5.0},
        },
    }


def _policy_dict(*, segments: dict) -> dict:
    d = {
        "config_snapshot": _cost_model_dict(),
        "segments": segments,
        "totals": {"expected_captured_loss": 0.0, "expected_residual_loss": 0.0},
    }
    d["policy_hash"] = content_hash(d)
    return d


_SHARED_INPUT_SHA = {"calibrator": "c" * 64, "history": "h" * 64, "policy": "p" * 64}


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
def _main_rows() -> list[tuple[str, str, float, str, str]]:
    return [
        ("i1", "segA", 0.90, "x", "v"),
        ("i2", "segA", 0.10, "v", "v"),
        ("i3", "segB", 0.80, "x", "v"),
        ("i4", "segB", 0.20, "v", "v"),
        ("i5", "segA", 0.50, "x", "v"),
        ("i6", "segB", 0.60, "v", "v"),
    ]


def _run_main(**kwargs):
    policy_dict = _policy_dict(
        segments={
            "segA": {"p_error_threshold": 0.5},
            "segB": {"p_error_threshold": 0.6},
        }
    )
    return run(
        policy_dict,
        CALIBRATOR,
        _history(_main_rows()),
        seed=kwargs.pop("seed", 42),
        n_random_draws=kwargs.pop("n_random_draws", 10),
        input_sha256s=_SHARED_INPUT_SHA,
        tool_version="0.1.0",
        **kwargs,
    )


class TestGridPinned:
    def test_grid_is_exactly_pinned(self) -> None:
        result = _run_main()
        expected_grid = [i / 100 for i in range(51)]
        assert result.curves["grid"] == expected_grid

    def test_grid_length_is_51(self) -> None:
        result = _run_main()
        assert len(result.curves["grid"]) == 51

    def test_series_lengths_match_grid(self) -> None:
        result = _run_main()
        curves = result.curves
        assert len(curves["policy"]) == 51
        assert len(curves["raw_confidence"]) == 51
        assert len(curves["random"]) == 51


class TestMonotonicity:
    def test_policy_curve_is_nondecreasing(self) -> None:
        result = _run_main()
        series = result.curves["policy"]
        assert all(b >= a - 1e-12 for a, b in zip(series, series[1:]))

    def test_raw_confidence_curve_is_nondecreasing(self) -> None:
        result = _run_main()
        series = result.curves["raw_confidence"]
        assert all(b >= a - 1e-12 for a, b in zip(series, series[1:]))

    def test_random_curve_is_nondecreasing(self) -> None:
        result = _run_main()
        series = result.curves["random"]
        assert all(b >= a - 1e-12 for a, b in zip(series, series[1:]))


class TestRandomExactness:
    def test_random_curve_equals_k_over_n_exactly(self) -> None:
        result = _run_main()
        n_history = result.n_history
        grid = result.curves["grid"]
        expected = [math.floor(f * n_history) / n_history for f in grid]
        assert result.curves["random"] == expected


class TestHandcraftedValues:
    """k values computed by hand for n_history=6 (see fixture docstring above)."""

    def test_k_zero_endpoint_is_zero(self) -> None:
        # grid[0] = 0.00 -> k = floor(0 * 6) = 0 for all three series.
        result = _run_main()
        assert result.curves["policy"][0] == 0.0
        assert result.curves["raw_confidence"][0] == 0.0
        assert result.curves["random"][0] == 0.0

    def test_k_one_at_f_020(self) -> None:
        # f=0.20 (grid index 20): k = floor(0.20*6) = 1.
        result = _run_main()
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
        result = _run_main()
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
        policy_dict = _policy_dict(
            segments={
                "segA": {"p_error_threshold": 0.5},
                "segB": {"p_error_threshold": 0.5},
            }
        )
        result = run(
            policy_dict,
            CALIBRATOR,
            _history(rows),
            seed=42,
            n_random_draws=10,
            input_sha256s=_SHARED_INPUT_SHA,
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
        result = _run_main()
        d = result.to_dict()
        assert "curves" in d
        assert d["curves"] == result.curves
        assert content_hash(d) == result.backtest_hash

    def test_backtest_hash_changes_if_curves_would_differ(self) -> None:
        # Sanity: curves participate in the hashed content (content_hash does
        # not special-case "curves" the way it does created_at/policy_hash).
        result = _run_main()
        d = result.to_dict()
        tampered = dict(d)
        tampered["curves"] = dict(d["curves"])
        tampered["curves"]["policy"] = [0.0] * 51
        assert content_hash(tampered) != result.backtest_hash

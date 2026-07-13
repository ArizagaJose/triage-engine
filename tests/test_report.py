"""Unit tests for the HTML report renderer (docs/interfaces.md §11).

Inputs are small handcrafted dicts (2 segments, tiny reliability_bins), with
content_hash/policy_hash computed on the fly via triage.artifacts so the
chain checks in report.render() pass by construction. report.py never reads
parquet data, so no stub calibrator/optimizer machinery is needed here.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from triage.artifacts import content_hash
from triage.exceptions import ContractViolation
from triage.report import render


def _calibrator_dict() -> dict[str, Any]:
    return {
        "method": "isotonic",
        "params": {"x_thresholds": [0.0, 1.0], "y_thresholds": [0.0, 1.0]},
        "ece_holdout": 0.05,
        "ece_ci": [0.03, 0.07],
        "ece_raw_holdout": 0.10,
        "n_golden": 100,
        "seed": 42,
        "input_sha256": "a" * 64,
        "tool_version": "0.1.0",
        "config": {
            "holdout_fraction": 0.3,
            "n_bins": 10,
            "n_bootstrap": 1000,
            "ci_level": 0.95,
        },
        "reliability_bins": {
            "before": [
                {"bin_lo": 0.0, "bin_hi": 0.5, "mean_pred": 0.2, "mean_observed": 0.3, "count": 10},
                {"bin_lo": 0.5, "bin_hi": 1.0, "mean_pred": 0.7, "mean_observed": 0.6, "count": 5},
            ],
            "after": [
                {
                    "bin_lo": 0.0,
                    "bin_hi": 0.5,
                    "mean_pred": 0.25,
                    "mean_observed": 0.28,
                    "count": 10,
                },
                {
                    "bin_lo": 0.5,
                    "bin_hi": 1.0,
                    "mean_pred": 0.65,
                    "mean_observed": 0.62,
                    "count": 5,
                },
            ],
        },
    }


def _cost_model_dict() -> dict[str, Any]:
    return {
        "budget": {"type": "fraction", "value": 0.1},
        "gates": {"max_residual_risk": None, "max_ece": 0.05},
        "segments": {
            "segA": {"false_accept_cost": 100.0, "review_cost": 10.0},
            "<script>alert(1)</script>": {"false_accept_cost": 20.0, "review_cost": 2.0},
        },
    }


def _policy_dict(*, calibrator_hash: str) -> dict[str, Any]:
    pre_hash = {
        "budget": {"type": "fraction", "value": 0.1},
        "config_snapshot": _cost_model_dict(),
        "input_sha256s": {
            "batch": "b" * 64,
            "calibrator": calibrator_hash,
            "costs": "c" * 64,
        },
        # One segment name deliberately contains a script tag: this is
        # artifact-derived (attacker-controlled-ish) data and must arrive
        # HTML-escaped in the rendered report, never raw.
        "segments": {
            "segA": {
                "expected_captured_loss": 500.0,
                "expected_residual_loss": 50.0,
                "n_review": 10,
                "n_total": 20,
                "p_error_threshold": 0.4,
            },
            "<script>alert(1)</script>": {
                "expected_captured_loss": 100.0,
                "expected_residual_loss": 10.0,
                "n_review": 2,
                "n_total": 5,
                "p_error_threshold": None,
            },
        },
        "tool_version": "0.1.0",
        "totals": {
            "expected_captured_loss": 600.0,
            "expected_residual_loss": 60.0,
            "n_review": 12,
            "n_total": 25,
            "review_spend": 120.0,
        },
    }
    pre_hash["policy_hash"] = content_hash(pre_hash)
    return pre_hash


def _backtest_dict(*, calibrator_hash: str, policy_hash: str) -> dict[str, Any]:
    grid = [i / 100 for i in range(51)]
    pre_hash: dict[str, Any] = {
        "curves": {
            "grid": grid,
            "policy": [min(1.0, i * 0.025) for i in range(51)],
            "raw_confidence": [min(1.0, i * 0.015) for i in range(51)],
            "random": [i / 100 for i in range(51)],
        },
        "expected": {"expected_captured_loss": 480.0, "expected_residual_loss": 90.0},
        "input_sha256s": {
            "calibrator": calibrator_hash,
            "history": "h" * 64,
            "policy": policy_hash,
        },
        "n_errors": 20,
        "n_history": 25,
        "n_random_draws": 50,
        "seed": 42,
        "strategies": {
            "policy": {
                "captured_loss": 500.0,
                "residual_loss": 100.0,
                "n_review": 12,
                "review_spend": 120.0,
                "efficiency": 41.67,
                "capture_rate": 0.83,
                "segments": {
                    "segA": {
                        "captured_loss": 450.0,
                        "residual_loss": 50.0,
                        "n_review": 10,
                        "n_total": 20,
                        "p_error_threshold": 0.4,
                    },
                    "<script>alert(1)</script>": {
                        "captured_loss": 50.0,
                        "residual_loss": 50.0,
                        "n_review": 2,
                        "n_total": 5,
                        "p_error_threshold": None,
                    },
                },
            },
            "raw_confidence": {
                "captured_loss": 400.0,
                "residual_loss": 200.0,
                "n_review": 12,
                "review_spend": 100.0,
                "efficiency": 33.33,
                "capture_rate": 0.67,
            },
            "random": {
                "captured_loss_mean": 300.0,
                "captured_loss_std": 20.0,
                "residual_loss_mean": 300.0,
                "review_spend_mean": 100.0,
                "n_review": 12,
                "efficiency": 25.0,
                "capture_rate": 0.5,
            },
        },
        "tool_version": "0.1.0",
        "total_true_loss": 600.0,
    }
    pre_hash["backtest_hash"] = content_hash(pre_hash)
    return pre_hash


def _valid_triplet() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    calibrator_dict = _calibrator_dict()
    calibrator_hash = content_hash(calibrator_dict)
    policy_dict = _policy_dict(calibrator_hash=calibrator_hash)
    backtest_dict = _backtest_dict(
        calibrator_hash=calibrator_hash, policy_hash=policy_dict["policy_hash"]
    )
    return calibrator_dict, policy_dict, backtest_dict


CREATED_AT = "2026-07-12T00:00:00+00:00"
TOOL_VERSION = "0.1.0"


class TestPurity:
    def test_render_is_pure(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        html1 = render(
            calibrator_dict,
            policy_dict,
            backtest_dict,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        html2 = render(
            copy.deepcopy(calibrator_dict),
            copy.deepcopy(policy_dict),
            copy.deepcopy(backtest_dict),
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        assert html1 == html2

    def test_render_is_pure_backtest_none(self) -> None:
        calibrator_dict, policy_dict, _ = _valid_triplet()
        html1 = render(
            calibrator_dict, policy_dict, None, created_at=CREATED_AT, tool_version=TOOL_VERSION
        )
        html2 = render(
            copy.deepcopy(calibrator_dict),
            copy.deepcopy(policy_dict),
            None,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        assert html1 == html2


class TestChainVerification:
    def test_wrong_calibrator_hash_raises(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        policy_dict = dict(policy_dict)
        policy_dict["input_sha256s"] = dict(policy_dict["input_sha256s"])
        policy_dict["input_sha256s"]["calibrator"] = "0" * 64
        with pytest.raises(ContractViolation, match="calibrator"):
            render(
                calibrator_dict,
                policy_dict,
                backtest_dict,
                created_at=CREATED_AT,
                tool_version=TOOL_VERSION,
            )

    def test_wrong_backtest_to_policy_link_raises(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        backtest_dict = dict(backtest_dict)
        backtest_dict["input_sha256s"] = dict(backtest_dict["input_sha256s"])
        backtest_dict["input_sha256s"]["policy"] = "0" * 64
        with pytest.raises(ContractViolation, match="policy"):
            render(
                calibrator_dict,
                policy_dict,
                backtest_dict,
                created_at=CREATED_AT,
                tool_version=TOOL_VERSION,
            )

    def test_wrong_backtest_to_calibrator_link_raises(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        backtest_dict = dict(backtest_dict)
        backtest_dict["input_sha256s"] = dict(backtest_dict["input_sha256s"])
        backtest_dict["input_sha256s"]["calibrator"] = "0" * 64
        with pytest.raises(ContractViolation, match="calibrator"):
            render(
                calibrator_dict,
                policy_dict,
                backtest_dict,
                created_at=CREATED_AT,
                tool_version=TOOL_VERSION,
            )

    def test_valid_chain_does_not_raise(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        render(
            calibrator_dict,
            policy_dict,
            backtest_dict,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )


class TestBacktestOptional:
    def test_backtest_none_renders_note(self) -> None:
        calibrator_dict, policy_dict, _ = _valid_triplet()
        html = render(
            calibrator_dict, policy_dict, None, created_at=CREATED_AT, tool_version=TOOL_VERSION
        )
        assert "backtest not run" in html.lower()

    def test_backtest_none_omits_backtest_and_coverage_figures(self) -> None:
        calibrator_dict, policy_dict, _ = _valid_triplet()
        html = render(
            calibrator_dict, policy_dict, None, created_at=CREATED_AT, tool_version=TOOL_VERSION
        )
        assert 'id="fig-backtest"' not in html
        assert 'id="fig-coverage"' not in html
        # The plan-only report still gets the calibration reliability figure.
        assert 'id="fig-reliability"' in html

    def test_backtest_given_includes_all_three_div_ids(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        html = render(
            calibrator_dict,
            policy_dict,
            backtest_dict,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        assert 'id="fig-reliability"' in html
        assert 'id="fig-backtest"' in html
        assert 'id="fig-coverage"' in html


class TestCreatedAt:
    def test_created_at_appears_exactly_once_and_on_one_line(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        html = render(
            calibrator_dict,
            policy_dict,
            backtest_dict,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        assert html.count(CREATED_AT) == 1

        lines = html.split("\n")
        matching_lines = [line for line in lines if 'id="created-at"' in line]
        assert len(matching_lines) == 1
        assert CREATED_AT in matching_lines[0]
        assert matching_lines[0].strip() == f'<p id="created-at" class="meta">{CREATED_AT}</p>'


class TestEscaping:
    def test_segment_name_with_script_tag_is_escaped(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        html = render(
            calibrator_dict,
            policy_dict,
            backtest_dict,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    def test_segment_name_escaped_when_backtest_none(self) -> None:
        calibrator_dict, policy_dict, _ = _valid_triplet()
        html = render(
            calibrator_dict, policy_dict, None, created_at=CREATED_AT, tool_version=TOOL_VERSION
        )
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

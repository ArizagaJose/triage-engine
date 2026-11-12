"""Unit tests for the HTML report renderer (docs/interfaces.md §11, v1.1).

Inputs are small handcrafted dicts (2 segments, tiny reliability_bins), with
content_hash/policy_hash computed on the fly via triage.artifacts so the
chain checks in report.render() pass by construction. report.py never reads
parquet data, so no stub calibrator/optimizer machinery is needed here.

Rendered HTML is read through the stable hooks of §11.11.5 (section ids,
`data-*` attributes) via tests/report_html.py, never through headings or
copy. The at-a-glance model itself (`_build_summary`) is unit-tested in
tests/test_report_summary.py; this file covers how it is RENDERED
(docs/test-spec-invariants.md §H.3).
"""

from __future__ import annotations

import copy
import re
from typing import Any

import pytest

from tests import report_html as rh
from triage.artifacts import content_hash
from triage.exceptions import ContractViolation
from triage.report import (
    ATTENTION_RESIDUAL_RATIO,
    _build_summary,
    fmt_cost,
    fmt_count,
    fmt_ece,
    fmt_lift,
    fmt_pct,
    fmt_ratio,
    render,
)


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
        assert 'id="fig-segments"' not in html
        # The plan-only report still gets the calibration reliability figure.
        assert 'id="fig-reliability"' in html

    def test_backtest_given_includes_all_four_div_ids(self) -> None:
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
        assert 'id="fig-segments"' in html


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


class TestGates:
    """Spec: docs/interfaces.md §11.9 -- informational gates panel. The
    section is located by its pinned id (#gates), not by sibling headings."""

    def test_gates_section_present_with_max_ece_verdict(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        html = render(
            calibrator_dict,
            policy_dict,
            backtest_dict,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        assert "<h2>Gates</h2>" in html

        # Compute the expected verdict from the fixture's own values rather
        # than hardcoding it: fixture costs set max_ece=0.05, calibrator
        # ece_holdout=0.05 -> strict `>` means this is a PASS, not a FAIL.
        max_ece = policy_dict["config_snapshot"]["gates"]["max_ece"]
        ece_holdout = calibrator_dict["ece_holdout"]
        assert max_ece is not None
        expected_verdict = "FAIL" if ece_holdout > max_ece else "PASS"
        assert expected_verdict == "PASS"  # sanity: fixture is 0.05 vs 0.05

        gates_section = rh.section(html, "gates")
        assert "max_ece" in gates_section
        assert f">{expected_verdict}<" in gates_section

    def test_gate_threshold_null_renders_not_set(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        policy_dict = copy.deepcopy(policy_dict)
        backtest_dict = copy.deepcopy(backtest_dict)

        # Both gates are already null/absent in the fixture's max_residual_risk;
        # additionally null out max_ece to exercise the "not set" path for it too.
        policy_dict["config_snapshot"]["gates"]["max_ece"] = None
        policy_dict["policy_hash"] = content_hash(policy_dict)
        backtest_dict["input_sha256s"]["policy"] = policy_dict["policy_hash"]
        backtest_dict["backtest_hash"] = content_hash(backtest_dict)

        html = render(
            calibrator_dict,
            policy_dict,
            backtest_dict,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        gates_section = rh.section(html, "gates")
        assert gates_section.count("not set") >= 2  # max_ece and max_residual_risk (planned)

    def test_failing_gate_renders_fail_and_does_not_raise(self) -> None:
        calibrator_dict, policy_dict, backtest_dict = _valid_triplet()
        policy_dict = copy.deepcopy(policy_dict)
        backtest_dict = copy.deepcopy(backtest_dict)

        # Set max_ece below the calibrator's actual ece_holdout (0.05) so the
        # gate strictly fails.
        policy_dict["config_snapshot"]["gates"]["max_ece"] = 0.01
        policy_dict["policy_hash"] = content_hash(policy_dict)
        backtest_dict["input_sha256s"]["policy"] = policy_dict["policy_hash"]
        backtest_dict["backtest_hash"] = content_hash(backtest_dict)

        assert calibrator_dict["ece_holdout"] > 0.01  # sanity: gate must actually fail

        html = render(
            calibrator_dict,
            policy_dict,
            backtest_dict,
            created_at=CREATED_AT,
            tool_version=TOOL_VERSION,
        )
        gates_section = rh.section(html, "gates")
        assert ">FAIL<" in gates_section

    def test_gates_present_without_backtest_no_realized_or_adherence_rows(self) -> None:
        calibrator_dict, policy_dict, _ = _valid_triplet()
        html = render(
            calibrator_dict, policy_dict, None, created_at=CREATED_AT, tool_version=TOOL_VERSION
        )
        assert "<h2>Gates</h2>" in html
        gates_section = rh.section(html, "gates")
        assert "max_residual_risk (planned)" in gates_section
        assert "max_residual_risk (realized)" not in gates_section
        assert "budget adherence" not in gates_section

    def test_gates_render_is_pure(self) -> None:
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
        gates1 = rh.section(html1, "gates")
        gates2 = rh.section(html2, "gates")
        assert gates1 == gates2


# ===========================================================================
# Report redesign (interfaces.md §11 v1.1; test-spec §H.3)
# ===========================================================================

SCRIPT_SEGMENT = "<script>alert(1)</script>"
ESCAPED_SCRIPT_SEGMENT = "&lt;script&gt;alert(1)&lt;/script&gt;"

Triplet = tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]


def _rehash(
    calibrator_dict: dict[str, Any],
    policy_dict: dict[str, Any],
    backtest_dict: dict[str, Any] | None,
) -> Triplet:
    """Re-link the hash chain after mutating fixture content, so variants
    exercise rendering rather than tripping chain verification."""
    policy_dict["input_sha256s"]["calibrator"] = content_hash(calibrator_dict)
    policy_dict["policy_hash"] = content_hash(policy_dict)
    if backtest_dict is not None:
        backtest_dict["input_sha256s"]["calibrator"] = content_hash(calibrator_dict)
        backtest_dict["input_sha256s"]["policy"] = policy_dict["policy_hash"]
        backtest_dict["backtest_hash"] = content_hash(backtest_dict)
    return calibrator_dict, policy_dict, backtest_dict


def _attention_triplet() -> Triplet:
    """Realized residual 200 vs 90 expected (2.2× > R): plan.vs_actual warns.
    The script-tag segment carries the largest gap, so it is `worst_segment`
    and gets NAMED in the check detail (closes the escaping gap, H.3.7)."""
    calibrator_dict, policy_dict, backtest_dict = copy.deepcopy(_valid_triplet())
    backtest_dict["strategies"]["policy"]["residual_loss"] = 200.0
    backtest_dict["strategies"]["policy"]["segments"][SCRIPT_SEGMENT]["residual_loss"] = 150.0
    return _rehash(calibrator_dict, policy_dict, backtest_dict)


def _failed_triplet() -> Triplet:
    """Attention + a declared residual limit of 150: planned 60 passes,
    realized 200 fails -> the only red path (a failed DECLARED gate)."""
    calibrator_dict, policy_dict, backtest_dict = _attention_triplet()
    policy_dict["config_snapshot"]["gates"]["max_residual_risk"] = 150.0
    return _rehash(calibrator_dict, policy_dict, backtest_dict)


def _plan_only_triplet() -> Triplet:
    calibrator_dict, policy_dict, _ = _valid_triplet()
    return calibrator_dict, policy_dict, None


VARIANTS = {
    "ok_full": _valid_triplet,
    "attention_full": _attention_triplet,
    "failed_full": _failed_triplet,
    "ok_plan_only": _plan_only_triplet,
}
FULL_VARIANTS = ["ok_full", "attention_full", "failed_full"]
EXPECTED_STATE = {
    "ok_full": ("ok", "full"),
    "attention_full": ("attention", "full"),
    "failed_full": ("failed", "full"),
    "ok_plan_only": ("ok", "plan_only"),
}
STATUS_LABEL = {
    "ok_full": "On track",
    "attention_full": "Needs attention",
    "failed_full": "Gate failed",
    "ok_plan_only": "Plan only \u2014 not backtested",
}
BADGE_CLASS = {
    "ok_full": "status-ok",
    "attention_full": "status-attention",
    "failed_full": "status-failed",
    "ok_plan_only": "status-neutral",
}
SEVERITY_ICON = {"fail": "\u2715", "warn": "!", "info": "i", "pass": "\u2713"}

_HTML_CACHE: dict[str, str] = {}


def _render(triplet: Triplet) -> str:
    calibrator_dict, policy_dict, backtest_dict = triplet
    return render(
        calibrator_dict,
        policy_dict,
        backtest_dict,
        created_at=CREATED_AT,
        tool_version=TOOL_VERSION,
    )


def _html(variant: str) -> str:
    """`render` is pure (TestPurity), so each variant is rendered once."""
    if variant not in _HTML_CACHE:
        _HTML_CACHE[variant] = _render(VARIANTS[variant]())
    return _HTML_CACHE[variant]


def _summary(variant: str) -> Any:
    return _build_summary(*VARIANTS[variant]())


def _big(html: str, kpi: str) -> tuple[list[str], str]:
    """(css classes, visible text) of a KPI tile's headline number."""
    match = re.search(r'<div class="(big[^"]*)">(.*?)</div>', rh.kpi_block(html, kpi), re.S)
    assert match, f"no headline number in KPI {kpi!r}"
    return match.group(1).split(), rh.text(match.group(2))


def _segment_rows(html: str) -> list[tuple[str, str]]:
    """(segment name, raw row html) for each data row, in document order,
    excluding the totals row."""
    body = rh.section(html, "segments").split("<tbody>", 1)[1].split("</tbody>", 1)[0]
    rows = re.findall(r"<tr>(.*?)</tr>", body, re.S)
    return [(rh.text(re.search(r"<td>(.*?)</td>", r, re.S).group(1)), r) for r in rows]  # type: ignore[union-attr]


def test_variants_are_what_their_names_say() -> None:
    """Guard for the fixtures themselves: every other test in this block
    trusts these four states."""
    for variant, expected in EXPECTED_STATE.items():
        summary = _summary(variant)
        assert (summary.status, summary.mode) == expected, variant


class TestPageStructure:
    """§11.4: the answer first, the evidence second, the audit trail last."""

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_section_order(self, variant: str) -> None:
        # Plan-only keeps the #results slot: it holds the "backtest not run" note.
        assert rh.section_order(_html(variant)) == list(rh.SECTION_IDS)

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_only_the_header_precedes_at_a_glance(self, variant: str) -> None:
        html = _html(variant)
        before = html.split("<main", 1)[1].split('<section id="at-a-glance">', 1)[0]
        for forbidden in ("<table", "<svg", "plotly-graph-div", "<script", "<h2", "<section"):
            assert forbidden not in before, forbidden
        assert 'id="created-at"' in before
        assert "artifact chain verified" in before

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_at_a_glance_holds_no_figure_and_no_script(self, variant: str) -> None:
        glance = rh.section(_html(variant), "at-a-glance")
        assert "plotly-graph-div" not in glance
        assert "<script" not in glance
        assert rh.PLOTLY_BUNDLE_MARKER not in glance

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_plotly_bundle_once_after_glance_and_before_first_figure(self, variant: str) -> None:
        """H.3.2: exactly one embedded plotly.js (so no figure embeds its own
        copy), placed so the first screen never waits for it."""
        html = _html(variant)
        assert html.count(rh.PLOTLY_BUNDLE_MARKER) == 1
        glance_close = html.index("</section>", html.index('<section id="at-a-glance">'))
        bundle = html.index(rh.PLOTLY_BUNDLE_MARKER)
        assert glance_close < bundle < rh.first_figure_index(html)

    def test_plan_only_results_slot_holds_the_note_and_no_figure(self) -> None:
        results = rh.section(_html("ok_plan_only"), "results")
        assert "backtest not run" in results.lower()
        assert "plotly-graph-div" not in results

    @pytest.mark.parametrize("variant", FULL_VARIANTS)
    def test_backtest_not_run_phrase_absent_everywhere_in_full_mode(self, variant: str) -> None:
        """§11.3: not in the note, not in any check copy."""
        assert "backtest not run" not in _html(variant).lower()

    @pytest.mark.parametrize("variant", ["ok_full", "ok_plan_only"])
    def test_audit_trail_collapsed_but_complete(self, variant: str) -> None:
        calibrator_dict, policy_dict, backtest_dict = VARIANTS[variant]()
        audit = rh.section(_html(variant), "audit")
        details = [attrs for tag, attrs in rh.start_tags(audit) if tag == "details"]
        assert len(details) == 1
        assert "open" not in details[0]

        expected = [content_hash(calibrator_dict), policy_dict["policy_hash"]]
        expected += list(policy_dict["input_sha256s"].values())
        if backtest_dict is not None:
            expected.append(backtest_dict["backtest_hash"])
            expected += list(backtest_dict["input_sha256s"].values())
        for full_hash in expected:
            assert full_hash in audit


class TestStatusAndChecks:
    """§11.5–11.6 as rendered (H.3.3–H.3.5)."""

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_main_carries_status_and_mode(self, variant: str) -> None:
        attrs = rh.main_attrs(_html(variant))
        assert (attrs["data-status"], attrs["data-mode"]) == EXPECTED_STATE[variant]

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_rendered_checks_mirror_the_summary_in_order(self, variant: str) -> None:
        rendered = [(c["id"], c["severity"]) for c in rh.rendered_checks(_html(variant))]
        assert rendered == [(c.id, c.severity) for c in _summary(variant).checks]

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_problems_open_everything_else_folded(self, variant: str) -> None:
        for check in rh.rendered_checks(_html(variant)):
            assert check["open"] == (check["severity"] in ("fail", "warn")), check
            assert check["severity"] in check["class"].split()

    @pytest.mark.parametrize("variant", ["attention_full", "failed_full"])
    def test_non_ok_reports_show_at_least_one_open_check(self, variant: str) -> None:
        assert any(c["open"] for c in rh.rendered_checks(_html(variant)))

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_status_badge_label_is_text_not_just_colour(self, variant: str) -> None:
        glance = rh.section(_html(variant), "at-a-glance")
        badge = glance.split('<span class="status-badge', 1)[1].split("<h1>", 1)[0]
        css, inner = badge.split(">", 1)
        assert BADGE_CLASS[variant] in css.replace('"', " ").split()
        assert rh.text(inner) == STATUS_LABEL[variant]

    def test_plan_only_is_never_green(self) -> None:
        """§11.6: unvalidated is not healthy -- neutral colour, not status-ok."""
        glance = rh.section(_html("ok_plan_only"), "at-a-glance")
        assert "status-ok" not in glance
        assert "status-neutral" in glance

    @pytest.mark.parametrize("variant", ["attention_full", "failed_full"])
    def test_headline_is_the_top_check_title_when_not_ok(self, variant: str) -> None:
        glance = rh.section(_html(variant), "at-a-glance")
        headline = rh.text(glance.split("<h1>", 1)[1].split("</h1>", 1)[0])
        assert headline == _summary(variant).checks[0].title

    @pytest.mark.parametrize("variant", ["ok_full", "ok_plan_only"])
    def test_ok_headline_is_not_a_check_title(self, variant: str) -> None:
        glance = rh.section(_html(variant), "at-a-glance")
        headline = rh.text(glance.split("<h1>", 1)[1].split("</h1>", 1)[0])
        assert headline
        assert headline not in {c.title for c in _summary(variant).checks}

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_each_check_links_to_an_existing_evidence_section(self, variant: str) -> None:
        html = _html(variant)
        for check in _summary(variant).checks:
            assert f'href="#{check.section}"' in rh.check_block(html, check.id)
            assert f'<section id="{check.section}">' in html

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_every_check_has_a_severity_glyph_and_its_title(self, variant: str) -> None:
        html = _html(variant)
        for check in _summary(variant).checks:
            block = rh.check_block(html, check.id)
            icon = block.split('class="sev-icon', 1)[1].split("</span>", 1)[0].split(">", 1)[1]
            assert icon == SEVERITY_ICON[check.severity]
            assert check.title in rh.text(block)


class TestKpis:
    """§11.7 as rendered (H.3.6)."""

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_four_tiles_in_pinned_order(self, variant: str) -> None:
        assert rh.kpi_order(_html(variant)) == list(rh.KPI_NAMES)

    @pytest.mark.parametrize("variant", FULL_VARIANTS)
    def test_full_mode_tile_numbers_match_the_summary(self, variant: str) -> None:
        html, kpis = _html(variant), _summary(variant).kpis
        assert _big(html, "reviewed")[1] == fmt_count(kpis["reviewed"]["value"])
        assert _big(html, "capture")[1] == fmt_pct(kpis["capture"]["value"])
        assert _big(html, "lift")[1] == fmt_lift(kpis["lift"]["random"])
        assert fmt_lift(kpis["lift"]["raw"]) in rh.text(rh.kpi_block(html, "lift"))
        assert _big(html, "residual")[1] == fmt_cost(kpis["residual"]["value"])
        assert fmt_cost(kpis["residual"]["plan_expected"]) in rh.text(
            rh.kpi_block(html, "residual")
        )

    def test_plan_only_lift_is_a_dash_and_says_why(self) -> None:
        html = _html("ok_plan_only")
        assert _big(html, "lift")[1] == "\u2014"
        assert "needs a backtest" in rh.text(rh.kpi_block(html, "lift"))

    def test_plan_only_capture_and_residual_are_labelled_expected(self) -> None:
        html = _html("ok_plan_only")
        assert "(expected)" in rh.text(rh.kpi_block(html, "capture"))
        assert "(expected)" in rh.text(rh.kpi_block(html, "residual"))

    @pytest.mark.parametrize(
        "variant,colour",
        [
            ("ok_full", None),
            ("attention_full", "c-warn"),
            ("failed_full", "c-fail"),
            ("ok_plan_only", None),
        ],
    )
    def test_residual_tile_colour(self, variant: str, colour: str | None) -> None:
        classes = set(_big(_html(variant), "residual")[0])
        if colour is None:
            assert not classes & {"c-warn", "c-fail"}
        else:
            assert colour in classes

    def test_declared_limit_shown_on_the_residual_tile(self) -> None:
        html = _html("failed_full")
        assert fmt_cost(150.0) in rh.text(rh.kpi_block(html, "residual"))


class TestSegmentsTable:
    """§11.8 as rendered."""

    @staticmethod
    def _ratio_boundary_triplet() -> Triplet:
        """segA actual 62.5 vs plan 50 -> EXACTLY R; the script segment 50 vs
        10 -> 5×. Only the strictly-above-R row may be amber."""
        calibrator_dict, policy_dict, backtest_dict = copy.deepcopy(_valid_triplet())
        backtest_dict["strategies"]["policy"]["segments"]["segA"]["residual_loss"] = 62.5
        return _rehash(calibrator_dict, policy_dict, backtest_dict)

    @staticmethod
    def _header_count(html: str) -> int:
        head = rh.section(html, "segments").split("</thead>", 1)[0]
        return len(re.findall(r"<th[\s>]", head))  # <th> cells, not the <thead> tag

    def test_full_mode_columns_and_totals_row(self) -> None:
        html = _html("ok_full")
        assert self._header_count(html) == 7  # + actual, actual ÷ plan
        assert 'class="totals"' in rh.section(html, "segments")
        assert len(_segment_rows(html)) == 2

    def test_plan_only_has_no_actual_columns_and_no_figure(self) -> None:
        html = _html("ok_plan_only")
        assert self._header_count(html) == 5
        assert "plotly-graph-div" not in rh.section(html, "segments")
        assert 'class="totals"' in rh.section(html, "segments")

    def test_rows_sorted_by_actual_desc_then_name(self) -> None:
        # ok_full: both segments realize 50 -> tie -> name ASC ("<" sorts before "s").
        assert [name for name, _ in _segment_rows(_html("ok_full"))] == [SCRIPT_SEGMENT, "segA"]
        # boundary variant: segA realizes 62.5 > 50 -> first.
        html = _render(self._ratio_boundary_triplet())
        assert [name for name, _ in _segment_rows(html)] == ["segA", SCRIPT_SEGMENT]

    def test_plan_only_rows_sorted_by_name(self) -> None:
        assert [name for name, _ in _segment_rows(_html("ok_plan_only"))] == [
            SCRIPT_SEGMENT,
            "segA",
        ]

    def test_ratio_cell_is_amber_only_strictly_above_r(self) -> None:
        assert 62.5 / 50.0 == ATTENTION_RESIDUAL_RATIO  # precondition: exactly R
        rows = dict(_segment_rows(_render(self._ratio_boundary_triplet())))
        assert "warn-text" not in rows["segA"]
        assert "warn-text" in rows[SCRIPT_SEGMENT]

    def test_costs_come_from_the_config_snapshot(self) -> None:
        rows = dict(_segment_rows(_html("ok_full")))
        assert "100 / 10" in rh.text(rows["segA"])  # false_accept_cost / review_cost
        assert "20 / 2" in rh.text(rows[SCRIPT_SEGMENT])


class TestEscapingRedesign:
    """H.3.7: artifact-derived text reaches the page escaped in every new
    place it can appear -- including the check detail that NAMES a segment."""

    def test_worst_segment_name_is_escaped_inside_the_drift_check(self) -> None:
        check = {c.id: c for c in _summary("attention_full").checks}["plan.vs_actual"]
        assert check.severity == "warn"  # precondition: the detail names the segment
        assert check.params["worst_segment"] == SCRIPT_SEGMENT

        html = _html("attention_full")
        assert ESCAPED_SCRIPT_SEGMENT in rh.check_block(html, "plan.vs_actual")
        assert SCRIPT_SEGMENT not in html
        assert "&amp;lt;" not in html  # and not double-escaped

    @pytest.mark.parametrize("variant", ["ok_full", "ok_plan_only"])
    def test_segment_name_is_escaped_in_the_segments_table(self, variant: str) -> None:
        segments = rh.section(_html(variant), "segments")
        assert ESCAPED_SCRIPT_SEGMENT in segments
        assert SCRIPT_SEGMENT not in segments


class TestFormatFilters:
    """§11.11.3 (H.3.9): locale-free, None-safe."""

    @pytest.mark.parametrize("fmt", [fmt_count, fmt_cost, fmt_pct, fmt_ratio, fmt_lift, fmt_ece])
    def test_none_is_an_em_dash_everywhere(self, fmt: Any) -> None:
        assert fmt(None) == "\u2014"

    @pytest.mark.parametrize(
        "value,expected",
        [(0, "0"), (1183, "1,183"), (10000, "10,000"), (-5, "-5"), (1183.6, "1,184")],
    )
    def test_count(self, value: float, expected: str) -> None:
        assert fmt_count(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [(336100.0, "336,100"), (142706.27, "142,706"), (0.0, "0"), (-2374.68, "-2,375")],
    )
    def test_cost(self, value: float, expected: str) -> None:
        assert fmt_cost(value) == expected

    @pytest.mark.parametrize(
        "value,decimals,expected",
        [
            (0.41911, 1, "41.9%"),
            (0.0, 1, "0.0%"),
            (-0.05, 1, "-5.0%"),
            (0.50287, 2, "50.29%"),
            (0.1, 0, "10%"),
        ],
    )
    def test_pct(self, value: float, decimals: int, expected: str) -> None:
        assert fmt_pct(value, decimals) == expected

    @pytest.mark.parametrize("value,expected", [(2.36, "2.36\u00d7"), (0.984, "0.98\u00d7")])
    def test_ratio(self, value: float, expected: str) -> None:
        assert fmt_ratio(value) == expected

    @pytest.mark.parametrize(
        "value,expected",
        [
            (4.17, "4.2\u00d7"),
            (9.94, "9.9\u00d7"),
            (9.99, "10.0\u00d7"),  # below 10 -> 1 decimal, THEN rounded
            (10.0, "10\u00d7"),  # from 10 on -> no decimals
            (20.16, "20\u00d7"),
        ],
    )
    def test_lift_boundary(self, value: float, expected: str) -> None:
        assert fmt_lift(value) == expected

    def test_ece(self) -> None:
        assert fmt_ece(0.04888) == "0.049"
        assert fmt_ece(0.0) == "0.000"


class TestNoExternalResources:
    """H.3.10: nothing fetched at view time -- scanned over the WHOLE file,
    embedded plotly.js included."""

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_no_external_references(self, variant: str) -> None:
        html = _html(variant)
        lowered = html.lower()
        for needle in ("<link", "@import", "url(http"):
            assert needle not in lowered, needle
        assert re.findall(r"<script[^>]*\bsrc\s*=", html, re.IGNORECASE) == []


class TestDegenerateInputs:
    """Every count, loss and rate at zero (capture_rate / efficiency null),
    gates declared at 0: the report still renders, and says "failed"."""

    @staticmethod
    def _zeroed() -> Triplet:
        calibrator_dict, policy_dict, backtest_dict = copy.deepcopy(_valid_triplet())
        for bins in calibrator_dict["reliability_bins"].values():
            for b in bins:
                b["count"] = 0
        policy_dict["config_snapshot"]["gates"] = {"max_ece": 0.0, "max_residual_risk": 0.0}
        for stats in policy_dict["segments"].values():
            stats.update(
                expected_captured_loss=0.0,
                expected_residual_loss=0.0,
                n_review=0,
                n_total=0,
                p_error_threshold=None,
            )
        policy_dict["totals"] = {
            "expected_captured_loss": 0.0,
            "expected_residual_loss": 0.0,
            "n_review": 0,
            "n_total": 0,
            "review_spend": 0.0,
        }
        backtest_dict["total_true_loss"] = 0.0
        backtest_dict["expected"] = {"expected_captured_loss": 0.0, "expected_residual_loss": 0.0}
        backtest_dict["curves"] = {k: [0.0] * 51 for k in ("policy", "raw_confidence", "random")}
        backtest_dict["curves"]["grid"] = [i / 100 for i in range(51)]
        strategies = backtest_dict["strategies"]
        strategies["policy"].update(
            captured_loss=0.0,
            residual_loss=0.0,
            n_review=0,
            review_spend=0.0,
            efficiency=None,
            capture_rate=None,
        )
        for stats in strategies["policy"]["segments"].values():
            stats.update(captured_loss=0.0, residual_loss=0.0, n_review=0, n_total=0)
        strategies["raw_confidence"].update(
            captured_loss=0.0,
            residual_loss=0.0,
            n_review=0,
            review_spend=0.0,
            efficiency=None,
            capture_rate=None,
        )
        strategies["random"].update(
            captured_loss_mean=0.0,
            captured_loss_std=0.0,
            residual_loss_mean=0.0,
            review_spend_mean=0.0,
            n_review=0,
            efficiency=None,
            capture_rate=None,
        )
        return _rehash(calibrator_dict, policy_dict, backtest_dict)

    @pytest.mark.parametrize("with_backtest", [True, False])
    def test_all_zero_inputs_render(self, with_backtest: bool) -> None:
        calibrator_dict, policy_dict, backtest_dict = self._zeroed()
        html = _render((calibrator_dict, policy_dict, backtest_dict if with_backtest else None))
        assert rh.main_attrs(html)["data-status"] == "failed"  # ECE 0.05 > declared 0
        assert rh.section_order(html) == list(rh.SECTION_IDS)
        assert _big(html, "capture")[1] == "\u2014"


def _row_cells(html: str, gate_name: str) -> list[str]:
    """The <td> texts of the gates-table row whose first cell is `gate_name`."""
    section = rh.section(html, "gates")
    row = section.split(f"<td>{gate_name}</td>", 1)[1].split("</tr>", 1)[0]
    cells = [gate_name]
    for chunk in row.split("<td>")[1:]:
        cells.append(re.sub(r"<[^>]+>", "", chunk.split("</td>", 1)[0]).strip())
    return cells


def _triplet_with_gates(
    *, calibrator_dict: dict[str, Any] | None = None, **gates: Any
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """A valid, re-hashed triplet whose policy config_snapshot.gates is
    updated with `gates` (a key set to ... is removed)."""
    calibrator_dict = calibrator_dict if calibrator_dict is not None else _calibrator_dict()
    calibrator_hash = content_hash(calibrator_dict)
    policy_dict = _policy_dict(calibrator_hash=calibrator_hash)
    snapshot_gates = policy_dict["config_snapshot"]["gates"]
    for key, value in gates.items():
        if value is ...:
            snapshot_gates.pop(key, None)
        else:
            snapshot_gates[key] = value
    policy_dict["policy_hash"] = content_hash(policy_dict)
    backtest_dict = _backtest_dict(
        calibrator_hash=calibrator_hash, policy_hash=policy_dict["policy_hash"]
    )
    return calibrator_dict, policy_dict, backtest_dict


def _render_gate(triplet: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> str:
    calibrator_dict, policy_dict, backtest_dict = triplet
    return render(
        calibrator_dict,
        policy_dict,
        backtest_dict,
        created_at=CREATED_AT,
        tool_version=TOOL_VERSION,
    )


class TestEceGateModeRow:
    """The max_ece row names the checked statistic and follows
    config_snapshot.gates.ece_gate_mode (absent => "point").

    Fixture numbers: ece_holdout=0.05, ece_ci=[0.03, 0.07], max_ece=0.05.
    """

    def test_absent_mode_checks_point_estimate(self) -> None:
        """Backward compatibility: policies written before gate modes carry no
        ece_gate_mode key and render exactly as point-mode gates."""
        triplet = _valid_triplet()
        assert "ece_gate_mode" not in triplet[1]["config_snapshot"]["gates"]
        cells = _row_cells(_render_gate(triplet), "max_ece")
        assert cells == [
            "max_ece",
            "calibrator.ece_holdout (point estimate; ece_gate_mode=point)",
            "0.050",
            "0.050",
            "PASS",  # equality at the threshold passes
        ]

    def test_explicit_point_renders_like_absent(self) -> None:
        explicit = _row_cells(_render_gate(_triplet_with_gates(ece_gate_mode="point")), "max_ece")
        absent = _row_cells(_render_gate(_valid_triplet()), "max_ece")
        assert explicit == absent

    def test_ci_upper_checks_upper_bound_and_fails(self) -> None:
        """Point estimate passes (0.05 <= 0.05) but the upper bound fails
        (0.07 > 0.05); the row shows the value that was actually checked."""
        cells = _row_cells(_render_gate(_triplet_with_gates(ece_gate_mode="ci_upper")), "max_ece")
        assert cells == [
            "max_ece",
            "calibrator.ece_ci[1] (upper bound of 95% bootstrap CI; ece_gate_mode=ci_upper)",
            "0.050",
            "0.070",
            "FAIL",
        ]

    def test_ci_upper_equality_at_threshold_passes(self) -> None:
        cells = _row_cells(
            _render_gate(_triplet_with_gates(ece_gate_mode="ci_upper", max_ece=0.07)), "max_ece"
        )
        assert cells[2:] == ["0.070", "0.070", "PASS"]

    def test_ci_upper_without_threshold_is_not_set(self) -> None:
        """Cannot come from `plan` (Gates rejects it), but the informational
        panel must never raise or invent a verdict."""
        cells = _row_cells(
            _render_gate(_triplet_with_gates(ece_gate_mode="ci_upper", max_ece=None)), "max_ece"
        )
        assert cells[2] == "not set"
        assert cells[4] == "not set"

    def test_ci_level_label_follows_calibrator_config(self) -> None:
        calibrator_dict = _calibrator_dict()
        calibrator_dict["config"]["ci_level"] = 0.9
        html = _render_gate(
            _triplet_with_gates(calibrator_dict=calibrator_dict, ece_gate_mode="ci_upper")
        )
        assert "upper bound of 90% bootstrap CI" in _row_cells(html, "max_ece")[1]
        calibration_section = rh.section(html, "calibration")
        assert "90% interval" in calibration_section
        assert "0.030 &ndash; 0.070" in calibration_section
        assert "95% CI" not in calibration_section

    def test_unknown_mode_renders_without_verdict(self) -> None:
        """A hand-edited (and re-hashed) snapshot with an unknown mode is shown
        as such -- no PASS/FAIL is invented and render does not raise."""
        cells = _row_cells(_render_gate(_triplet_with_gates(ece_gate_mode="bogus")), "max_ece")
        assert cells[1] == "unrecognized ece_gate_mode &#39;bogus&#39;"
        assert cells[3] == "&mdash;"
        assert cells[4] == "&mdash;"

    def test_residual_rows_name_checked_fields(self) -> None:
        html = _render_gate(_valid_triplet())
        assert _row_cells(html, "max_residual_risk (planned)")[1] == (
            "policy.totals.expected_residual_loss"
        )
        assert _row_cells(html, "max_residual_risk (realized)")[1] == (
            "backtest.strategies.policy.residual_loss"
        )

    def test_mode_change_is_pure(self) -> None:
        triplet = _triplet_with_gates(ece_gate_mode="ci_upper")
        assert _render_gate(triplet) == _render_gate(copy.deepcopy(triplet))

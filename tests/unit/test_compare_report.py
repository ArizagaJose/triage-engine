"""Unit tests for triage.compare_report (docs/interfaces.md §14.8): the
comparison page, rendered from comparisons built on the hand-checkable
history of tests/support.py. Tests read the page through its stable hooks
(`main[data-status|data-verdict|data-basis]`, section ids, `data-kpi`,
`data-check-id`, `data-segment`, `data-side`, figure div ids), never copy.
"""

from __future__ import annotations

import copy
import functools
import re
from typing import Any

import pytest

from tests.report_html import (
    PLOTLY_BUNDLE_MARKER,
    first_figure_index,
    kpi_block,
    main_attrs,
    rendered_checks,
    section,
    start_tags,
    strip_scripts,
    table_shapes,
    text,
)
from tests.support import calibrator_dict, compare_history, cost_model_dict, policy_dict
from triage import compare, compare_report
from triage.artifacts import content_hash
from triage.costs import CostModel
from triage.exceptions import CandidateRegression, ContractViolation

CAL = calibrator_dict()
INV_CAL = calibrator_dict(inverted=True)
CREATED_AT = "2000-01-01T00:00:00+00:00"
COMPARE_SECTIONS = (
    "at-a-glance",
    "results",
    "segments",
    "workload",
    "uncertainty",
    "basis",
    "audit",
)

POLICIES: dict[str, tuple[dict[str, Any], dict[str, float | None]]] = {
    "cur": (CAL, {"A": 0.45, "B": None}),
    "improved": (CAL, {"A": 0.6, "B": 0.75}),
    "wider": (CAL, {"A": 0.45, "B": 0.35}),
    "only_a": (CAL, {"A": 0.45}),
    "inverted": (INV_CAL, {"A": 0.45, "B": 0.35}),
    "cur_plus_b1": (CAL, {"A": 0.45, "B": 0.75}),
}


@functools.cache
def comparison(
    current: str,
    candidate: str,
    *,
    copies: int = 50,
    n_bootstrap: int = 200,
    candidate_costs: tuple[tuple[str, float, float], ...] | None = None,
    eval_costs: bool = False,
) -> dict[str, Any]:
    """comparison.json dict (cached; always deep-copy before mutating)."""
    cur_cal, cur_thr = POLICIES[current]
    cand_cal, cand_thr = POLICIES[candidate]
    costs = {name: (fa, rc) for name, fa, rc in candidate_costs} if candidate_costs else None
    result = compare.run(
        current_policy=policy_dict(cur_cal, cur_thr),
        current_calibrator=cur_cal,
        candidate_policy=policy_dict(cand_cal, cand_thr, costs=costs),
        candidate_calibrator=cand_cal,
        history=compare_history(copies),
        history_sha256="history-sha",
        eval_costs=CostModel.model_validate(cost_model_dict()) if eval_costs else None,
        eval_costs_sha256="eval-sha" if eval_costs else None,
        n_bootstrap=n_bootstrap,
        tool_version="0.1.0",
    )
    d = result.to_dict()
    d["created_at"] = CREATED_AT
    return d


def render(c: dict[str, Any], created_at: str = CREATED_AT) -> str:
    return compare_report.render(copy.deepcopy(c), created_at=created_at, tool_version="0.1.0")


@functools.cache
def page(current: str, candidate: str, **kw: Any) -> str:
    return render(comparison(current, candidate, **kw))


def rehashed(c: dict[str, Any]) -> dict[str, Any]:
    c["comparison_hash"] = content_hash(c)
    return c


# ---------------------------------------------------------------------------
# Purity and page structure
# ---------------------------------------------------------------------------


class TestPurityAndStructure:
    def test_byte_identical_for_identical_input(self) -> None:
        c = comparison("cur", "wider")
        assert render(c) == render(copy.deepcopy(c))

    def test_created_at_is_the_only_variance(self) -> None:
        c = comparison("cur", "wider")
        a, b = render(c, "A-TIME").splitlines(), render(c, "B-TIME").splitlines()
        diff = [(x, y) for x, y in zip(a, b) if x != y]
        line = '  <p id="created-at" class="meta">{}</p>'
        assert diff == [(line.format("A-TIME"), line.format("B-TIME"))]

    def test_section_order(self) -> None:
        html = page("cur", "wider")
        positions = [html.index(f'<section id="{sid}">') for sid in COMPARE_SECTIONS]
        assert positions == sorted(positions)
        assert html.index("<header") < positions[0]

    def test_shared_shell(self) -> None:
        html = page("cur", "wider")
        header = html.split("<header", 1)[1].split("</header>", 1)[0]
        assert "policy comparison report" in header
        assert "comparison verified" in header
        # Not the single report's claim: only the comparison file is checked here.
        assert "artifact chain verified" not in header
        assert "<title>triage comparison</title>" in html
        assert 'class="page-foot"' in html

    def test_plotly_bundle_once_between_glance_and_first_figure(self) -> None:
        html = page("cur", "wider")
        assert html.count(PLOTLY_BUNDLE_MARKER) == 1
        bundle = html.index(PLOTLY_BUNDLE_MARKER)
        glance_end = html.index("</section>", html.index('<section id="at-a-glance">'))
        assert glance_end < bundle < first_figure_index(html)
        for div_id in ("fig-compare", "fig-compare-segments"):
            assert f'id="{div_id}"' in html

    def test_no_external_resources(self) -> None:
        """Same rule as the single report (H.3.10), scanned over the whole
        file, embedded plotly.js included."""
        html = page("cur", "wider")
        lowered = html.lower()
        for needle in ("<link", "@import", "url(http"):
            assert needle not in lowered, needle
        assert re.findall(r"<script[^>]*\bsrc\s*=", html, re.IGNORECASE) == []

    @pytest.mark.parametrize(
        "pair", [("cur", "cur"), ("cur", "wider"), ("cur", "only_a"), ("wider", "cur")]
    )
    def test_every_table_is_rectangular(self, pair: tuple[str, str]) -> None:
        for widths in table_shapes(page(*pair)):
            assert len(set(widths)) == 1, widths


# ---------------------------------------------------------------------------
# Status, verdict, headline hooks
# ---------------------------------------------------------------------------


class TestStatus:
    @pytest.mark.parametrize(
        "pair,kw,status,verdict",
        [
            (("cur", "cur"), {}, "neutral", "no_change"),
            (("cur", "improved"), {}, "ok", "improvement"),
            (("cur", "inverted"), {}, "failed", "regression"),
            (("cur", "wider"), {}, "attention", "trade_off"),
            # Cheaper overall? No: fewer reviews but far less caught -> the
            # regression gate would fail, so the page says so.
            (("wider", "cur"), {}, "failed", "trade_off"),
            (("cur", "cur_plus_b1"), {"copies": 1, "n_bootstrap": 300}, "neutral", "inconclusive"),
            # No CI: the point verdict is shown; only a point regression is red.
            (("cur", "inverted"), {"n_bootstrap": 0}, "failed", "regression"),
            (("wider", "cur"), {"n_bootstrap": 0}, "attention", "trade_off"),
        ],
    )
    def test_status_and_verdict(
        self, pair: tuple[str, str], kw: dict[str, Any], status: str, verdict: str
    ) -> None:
        attrs = main_attrs(page(*pair, **kw))
        assert attrs["data-status"] == status
        assert attrs["data-verdict"] == verdict
        assert attrs["data-basis"] == "policies"

    def test_failed_status_matches_the_regression_gate(self) -> None:
        """The page's red status is exactly `check_regression_gate`'s condition."""
        for pair in [("cur", "cur"), ("cur", "improved"), ("cur", "wider"), ("wider", "cur")]:
            c = comparison(*pair)
            result = compare.ComparisonResult(**{k: v for k, v in c.items() if k != "created_at"})
            try:
                compare.check_regression_gate(result)
                gate_fails = False
            except CandidateRegression:
                gate_fails = True
            assert (main_attrs(page(*pair))["data-status"] == "failed") == gate_fails

    def test_worse_overall_badge_and_headline(self) -> None:
        glance = text(section(page("wider", "cur"), "at-a-glance"))
        assert "Worse overall" in glance
        assert "costs more overall" in glance


# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------


class TestKpis:
    def test_tiles_in_order(self) -> None:
        html = page("cur", "wider")
        ids = [a["data-kpi"] for t, a in start_tags(html) if "data-kpi" in a]
        assert ids == ["captured", "residual", "reviews", "spend"]

    @pytest.mark.parametrize("pair", [("cur", "wider"), ("wider", "cur"), ("cur", "improved")])
    def test_residual_direction_is_captured_direction(self, pair: tuple[str, str]) -> None:
        """Less risk left unreviewed is better: the residual tile must say
        "better" exactly when the captured tile does (regression test for a
        tile that once read -42,500 as "worse")."""
        tags = {a["data-kpi"]: a for t, a in start_tags(page(*pair)) if "data-kpi" in a}
        assert tags["residual"]["data-direction"] == tags["captured"]["data-direction"]

    def test_tile_values(self) -> None:
        block = text(kpi_block(page("cur", "wider"), "reviews"))
        assert "+150" in block  # 3 more reviews x 50 copies
        assert "150 \u2192 300" in block
        assert "worse" in block
        assert "CI [" in block

    def test_no_ci_line_without_bootstrap(self) -> None:
        assert "CI [" not in text(kpi_block(page("cur", "wider", n_bootstrap=0), "captured"))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_ids(html: str) -> dict[str, str]:
    return {c["id"]: c["severity"] for c in rendered_checks(html)}


class TestChecks:
    def test_shared_basis(self) -> None:
        checks = check_ids(page("cur", "wider"))
        assert checks["basis.shared"] == "pass"
        assert checks["verdict.total_cost"] == "pass"

    def test_eval_costs_and_planned_under_other_costs(self) -> None:
        html = page("cur", "improved", candidate_costs=(("A", 100.0, 2.0), ("B", 40.0, 1.0)),
                    eval_costs=True)  # fmt: skip
        checks = check_ids(html)
        assert checks["basis.eval_costs"] == "info"
        assert checks["basis.planned_under_other_costs.candidate"] == "warn"
        assert "basis.planned_under_other_costs.current" not in checks
        assert main_attrs(html)["data-basis"] == "eval_costs"
        block = text(html.split('data-check-id="basis.planned_under_other_costs.candidate"')[1])
        assert "B: planned with 40 / 1, priced here at 10 / 1" in block

    def test_segment_missing_from_a_policy(self) -> None:
        checks = check_ids(page("wider", "only_a"))
        assert checks["segments.not_in_plan.candidate"] == "warn"

    def test_costlier_total_fails_with_ci_warns_without(self) -> None:
        assert check_ids(page("wider", "cur"))["verdict.total_cost"] == "fail"
        assert check_ids(page("wider", "cur", n_bootstrap=0))["verdict.total_cost"] == "warn"

    def test_uncertainty_checks(self) -> None:
        assert check_ids(page("cur", "wider", n_bootstrap=0))["uncertainty.not_computed"] == "info"
        inconclusive = page("cur", "cur_plus_b1", copies=1, n_bootstrap=300)
        assert check_ids(inconclusive)["uncertainty.inconclusive"] == "info"

    def test_identical_policies_have_no_total_cost_check(self) -> None:
        assert "verdict.total_cost" not in check_ids(page("cur", "cur"))

    def test_checks_sorted_by_severity(self) -> None:
        order = {"fail": 0, "warn": 1, "info": 2, "pass": 3}
        severities = [c["severity"] for c in rendered_checks(page("wider", "only_a"))]
        assert severities == sorted(severities, key=order.__getitem__)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


class TestSections:
    def test_segment_statuses_per_side(self) -> None:
        html = strip_scripts(page("cur", "only_a"))
        seg = section(html, "segments")
        row_b = seg.split('data-segment="B"', 1)[1].split("</tr>", 1)[0]
        assert 'data-side="current" data-status="no_threshold"' in row_b
        assert 'data-side="candidate" data-status="not_in_plan"' in row_b
        assert "not in plan" in text(row_b) and "no threshold" in text(row_b)

    def test_workload_section_identity(self) -> None:
        body = text(section(page("cur", "wider"), "workload"))
        assert "only the candidate reviews (added) 150 1,000 150" in body
        assert "1,000 \u2212 0 = +1,000" in body

    def test_uncertainty_section_without_ci(self) -> None:
        body = text(section(page("cur", "wider", n_bootstrap=0), "uncertainty"))
        assert "--n-bootstrap 0" in body

    def test_audit_lists_every_recorded_hash(self) -> None:
        c = comparison("cur", "wider")
        audit = section(page("cur", "wider"), "audit")
        assert c["comparison_hash"] in audit
        for value in c["input_sha256s"].values():
            if value is not None:
                assert value in audit


class TestEscaping:
    def test_segment_names_are_escaped(self) -> None:
        evil = "<img src=x onerror=alert(1)>"
        history = compare_history().with_columns(compare_history()["segment"].replace({"B": evil}))
        costs = {"A": (100.0, 2.0), evil: (10.0, 1.0)}
        current = policy_dict(CAL, {"A": 0.45, evil: None}, costs=costs)
        candidate = policy_dict(CAL, {"A": 0.45, evil: 0.35}, costs=costs)
        result = compare.run(
            current_policy=current,
            current_calibrator=CAL,
            candidate_policy=candidate,
            candidate_calibrator=CAL,
            history=history,
            history_sha256="h",
            n_bootstrap=0,
            tool_version="0.1.0",
        )
        html = render(result.to_dict())
        assert evil not in strip_scripts(html)
        assert "&lt;img src=x onerror=alert(1)&gt;" in html


# ---------------------------------------------------------------------------
# Verification (§14.8): the comparison file itself
# ---------------------------------------------------------------------------


class TestVerification:
    def test_edited_comparison_is_rejected(self) -> None:
        c = copy.deepcopy(comparison("cur", "wider"))
        c["delta"]["n_review"] = 0
        with pytest.raises(ContractViolation, match="comparison_hash=.* but content_hash"):
            render(c)

    def test_rehashed_inconsistent_delta_is_rejected(self) -> None:
        c = copy.deepcopy(comparison("cur", "wider"))
        c["delta"]["n_review"] = 0
        with pytest.raises(ContractViolation, match=r"delta.n_review=0 but candidate - current"):
            render(rehashed(c))

    def test_rehashed_broken_tie_out_is_rejected(self) -> None:
        c = copy.deepcopy(comparison("cur", "wider"))
        c["policies"]["current"]["outcome"]["residual_loss"] += 1.0
        c["delta"]["residual_loss"] -= 1.0
        with pytest.raises(ContractViolation, match="does not tie out"):
            render(rehashed(c))

    def test_rehashed_overlap_mismatch_is_rejected(self) -> None:
        c = copy.deepcopy(comparison("cur", "wider"))
        c["overlap"]["neither"]["n_items"] += 1
        with pytest.raises(ContractViolation, match="overlap groups cover"):
            render(rehashed(c))

    def test_rehashed_policy_hash_mismatch_is_rejected(self) -> None:
        c = copy.deepcopy(comparison("cur", "wider"))
        c["input_sha256s"]["candidate_policy"] = "0" * 64
        with pytest.raises(ContractViolation, match="candidate_policy differs"):
            render(rehashed(c))

    def test_missing_key(self) -> None:
        c = copy.deepcopy(comparison("cur", "wider"))
        del c["overlap"]
        with pytest.raises(ContractViolation, match=r"missing required key\(s\) \['overlap'\]"):
            render(c)

    def test_malformed_structure_is_a_contract_violation_not_a_crash(self) -> None:
        c = copy.deepcopy(comparison("cur", "wider"))
        del c["policies"]["candidate"]
        with pytest.raises(ContractViolation, match="malformed comparison.json"):
            render(rehashed(c))

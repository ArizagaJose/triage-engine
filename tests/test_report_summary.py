"""Unit + property tests for the report's at-a-glance model (`_build_summary`).

Implements docs/test-spec-invariants.md §H.1 and §H.2 against
docs/interfaces.md §11.5–11.8. Pure tests: no HTML, no CLI, no hashing
(`_build_summary` never verifies the chain, `render` does — see
tests/unit/test_report.py for everything rendered).

Assertions are on `Summary.status` / `.mode` and `Check.id` / `.severity` /
`.params` only, never on `.title` / `.detail` wording (§11.5.2), except where
the spec pins a property of the text itself (H.1.13: text is NOT pre-escaped).

The check catalogue order and severity ranks are written out independently
below rather than imported from triage.report, so these tests pin the SPEC,
not whatever the module currently says.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from triage.report import (
    ATTENTION_RESIDUAL_RATIO,
    Check,
    Summary,
    _build_gates_view,
    _build_segments_view,
    _build_summary,
    _status_from_checks,
)

# interfaces.md §11.5 table, in catalogue order.
CATALOGUE = [
    "gate.max_ece",
    "gate.max_residual_risk.planned",
    "gate.max_residual_risk.realized",
    "gate.max_residual_risk",
    "plan.vs_actual",
    "value.vs_baselines",
    "calibration.vs_raw",
    "calibration.ci_margin",
    "budget.adherence",
    "backtest.not_run",
]
SEVERITY_RANK = {"fail": 0, "warn": 1, "info": 2, "pass": 3}
BACKTEST_ONLY = {
    "gate.max_residual_risk.realized",
    "plan.vs_actual",
    "value.vs_baselines",
    "budget.adherence",
}

PROPERTY_SETTINGS = settings(derandomize=True, max_examples=200, deadline=None)

Triplet = tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------


def seg(
    exp_res: float,
    real_res: float,
    *,
    exp_cap: float = 0.0,
    real_cap: float = 0.0,
    n_plan: int = 0,
    n_replay: int | None = None,
    n_total: int = 50,
    threshold: float | None = None,
) -> dict[str, Any]:
    return {
        "exp_res": exp_res,
        "real_res": real_res,
        "exp_cap": exp_cap,
        "real_cap": real_cap,
        "n_plan": n_plan,
        "n_replay": n_plan if n_replay is None else n_replay,
        "n_total": n_total,
        "threshold": threshold,
    }


def default_segments() -> dict[str, dict[str, Any]]:
    # segA: reviewed, slightly worse than plan. segB: unreviewed, on plan.
    return {
        "segA": seg(100.0, 110.0, exp_cap=400.0, real_cap=420.0, n_plan=10, threshold=0.4),
        "segB": seg(50.0, 50.0),
    }


def build(
    *,
    ece: float = 0.04,
    ece_raw: float = 0.10,
    ci_upper: float = 0.045,
    max_ece: float | None = 0.05,
    max_residual: float | None = None,
    segments: dict[str, dict[str, Any]] | None = None,
    realized_only: dict[str, dict[str, Any]] | None = None,
    random_captured: float = 100.0,
    raw_captured: float = 50.0,
    budget: dict[str, Any] | None = None,
    with_backtest: bool = True,
) -> Triplet:
    """A minimal, internally consistent (calibrator, policy, backtest) triple.

    Defaults evaluate to status "ok": every gate passes or is unset, realized
    residual is 160 vs 150 planned (1.07×), the policy beats both baselines,
    and reviews match the plan. Totals are summed in SORTED segment order
    (the artifacts' tie-out rule, §5.8/§9.8) so dict insertion order can never
    change a total. `realized_only` segments exist only in the backtest
    (unseen at plan time, §9.4).
    """
    segs = segments if segments is not None else default_segments()
    extra = realized_only or {}
    names = sorted(segs)
    realized_names = sorted(set(segs) | set(extra))
    allseg = {**segs, **extra}

    calibrator = {
        "method": "isotonic",
        "ece_holdout": ece,
        "ece_raw_holdout": ece_raw,
        "ece_ci": [min(ece, ci_upper) / 2, ci_upper],
        "n_golden": 500,
        "seed": 42,
        "reliability_bins": {"before": [], "after": []},
    }

    totals = {
        "expected_captured_loss": sum(segs[n]["exp_cap"] for n in names),
        "expected_residual_loss": sum(segs[n]["exp_res"] for n in names),
        "n_review": sum(segs[n]["n_plan"] for n in names),
        "n_total": sum(segs[n]["n_total"] for n in names),
        "review_spend": float(sum(segs[n]["n_plan"] for n in names)),
    }
    budget = budget if budget is not None else {"type": "fraction", "value": 0.1}
    policy = {
        "budget": budget,
        "config_snapshot": {
            "budget": budget,
            "gates": {"max_ece": max_ece, "max_residual_risk": max_residual},
            "segments": {
                **{n: {"false_accept_cost": 100.0, "review_cost": 1.0} for n in names},
                "default": {"false_accept_cost": 10.0, "review_cost": 1.0},
            },
        },
        "segments": {
            n: {
                "expected_captured_loss": segs[n]["exp_cap"],
                "expected_residual_loss": segs[n]["exp_res"],
                "n_review": segs[n]["n_plan"],
                "n_total": segs[n]["n_total"],
                "p_error_threshold": segs[n]["threshold"],
            }
            for n in segs  # insertion order preserved on purpose (H.2.3)
        },
        "totals": totals,
    }
    if not with_backtest:
        return calibrator, policy, None

    captured = sum(allseg[n]["real_cap"] for n in realized_names)
    residual = sum(allseg[n]["real_res"] for n in realized_names)
    total = captured + residual
    n_replay = sum(allseg[n]["n_replay"] for n in realized_names)
    backtest = {
        "expected": {
            "expected_captured_loss": totals["expected_captured_loss"],
            "expected_residual_loss": totals["expected_residual_loss"],
        },
        "total_true_loss": total,
        "n_history": totals["n_total"],
        "n_errors": 10,
        "strategies": {
            "policy": {
                "captured_loss": captured,
                "residual_loss": residual,
                "n_review": n_replay,
                "review_spend": float(n_replay),
                "efficiency": (captured / n_replay) if n_replay else None,
                "capture_rate": (captured / total) if total else None,
                "segments": {
                    n: {
                        "captured_loss": allseg[n]["real_cap"],
                        "residual_loss": allseg[n]["real_res"],
                        "n_review": allseg[n]["n_replay"],
                        "n_total": allseg[n]["n_total"],
                        "p_error_threshold": allseg[n]["threshold"],
                    }
                    for n in allseg
                },
            },
            "raw_confidence": {
                "captured_loss": raw_captured,
                "residual_loss": total - raw_captured,
                "n_review": n_replay,
                "review_spend": float(n_replay),
                "efficiency": (raw_captured / n_replay) if n_replay else None,
                "capture_rate": (raw_captured / total) if total else None,
            },
            "random": {
                "captured_loss_mean": random_captured,
                "captured_loss_std": 1.0,
                "residual_loss_mean": total - random_captured,
                "review_spend_mean": float(n_replay),
                "n_review": n_replay,
                "efficiency": (random_captured / n_replay) if n_replay else None,
                "capture_rate": (random_captured / total) if total else None,
            },
        },
    }
    return calibrator, policy, backtest


def by_id(summary: Summary) -> dict[str, Check]:
    return {c.id: c for c in summary.checks}


def ids(summary: Summary) -> list[str]:
    return [c.id for c in summary.checks]


def make_check(severity: str, check_id: str = "gate.max_ece") -> Check:
    return Check(
        id=check_id,
        severity=severity,
        title="",
        detail="",
        value="",
        section="gates",  # type: ignore[arg-type]
    )


def test_defaults_are_a_clean_ok_baseline() -> None:
    """Sanity for the builder itself: every other test perturbs ONE thing away
    from this baseline, so it must really be ok with no fail/warn anywhere."""
    summary = _build_summary(*build())
    assert summary.status == "ok"
    assert summary.mode == "full"
    assert {c.severity for c in summary.checks} <= {"info", "pass"}


# ---------------------------------------------------------------------------
# H.1.1 / H.1.2 — gate checks
# ---------------------------------------------------------------------------


class TestGateChecks:
    @pytest.mark.parametrize(
        "check_id",
        ["gate.max_ece", "gate.max_residual_risk.planned", "gate.max_residual_risk.realized"],
    )
    @pytest.mark.parametrize("where,expected", [("equal", "pass"), ("one_ulp_below", "fail")])
    def test_gate_comparison_is_strict_and_boundary_exact(
        self, check_id: str, where: str, expected: str
    ) -> None:
        """value == threshold passes; threshold one ULP below value fails
        (strict `>`, same as the CLI gates)."""
        cal, pol, bt = build(max_residual=1e12)
        assert bt is not None
        value = {
            "gate.max_ece": cal["ece_holdout"],
            "gate.max_residual_risk.planned": pol["totals"]["expected_residual_loss"],
            "gate.max_residual_risk.realized": bt["strategies"]["policy"]["residual_loss"],
        }[check_id]
        threshold = value if where == "equal" else math.nextafter(value, -math.inf)
        gate_key = "max_ece" if check_id == "gate.max_ece" else "max_residual_risk"
        pol["config_snapshot"]["gates"][gate_key] = threshold

        check = by_id(_build_summary(cal, pol, bt))[check_id]

        assert check.severity == expected
        assert check.params["threshold"] == threshold
        assert check.params["value"] == value
        assert check.section == "gates"

    def test_undeclared_max_ece_is_info_not_pass(self) -> None:
        check = by_id(_build_summary(*build(max_ece=None)))["gate.max_ece"]
        assert check.severity == "info"
        assert check.params["threshold"] is None

    def test_gates_block_missing_entirely_never_raises(self) -> None:
        cal, pol, bt = build()
        del pol["config_snapshot"]["gates"]
        checks = by_id(_build_summary(cal, pol, bt))
        assert checks["gate.max_ece"].severity == "info"
        assert checks["gate.max_residual_risk"].severity == "info"

    @pytest.mark.parametrize("with_backtest", [True, False])
    def test_undeclared_residual_gate_collapses_to_one_info_check(
        self, with_backtest: bool
    ) -> None:
        """H.1.2: one merged `info` check, and the .planned / .realized ids
        are ABSENT (not merely outnumbered)."""
        summary = _build_summary(*build(max_residual=None, with_backtest=with_backtest))
        present = ids(summary)
        assert by_id(summary)["gate.max_residual_risk"].severity == "info"
        assert "gate.max_residual_risk.planned" not in present
        assert "gate.max_residual_risk.realized" not in present

    def test_declared_residual_gate_emits_planned_and_realized_not_merged(self) -> None:
        present = ids(_build_summary(*build(max_residual=1e12)))
        assert "gate.max_residual_risk.planned" in present
        assert "gate.max_residual_risk.realized" in present
        assert "gate.max_residual_risk" not in present

    def test_declared_residual_gate_plan_only_has_no_realized_check(self) -> None:
        present = ids(_build_summary(*build(max_residual=1e12, with_backtest=False)))
        assert "gate.max_residual_risk.planned" in present
        assert "gate.max_residual_risk.realized" not in present

    def test_zero_thresholds_are_declared_not_unset(self) -> None:
        """0 is a real limit (falsy, but not None): every positive value fails it."""
        checks = by_id(_build_summary(*build(max_ece=0.0, max_residual=0.0)))
        assert checks["gate.max_ece"].severity == "fail"
        assert checks["gate.max_residual_risk.planned"].severity == "fail"
        assert checks["gate.max_residual_risk.realized"].severity == "fail"


# ---------------------------------------------------------------------------
# H.1.3 / H.1.4 — plan vs actual, worst segment
# ---------------------------------------------------------------------------


class TestPlanVsActual:
    @staticmethod
    def _with(expected: float, realized: float) -> Summary:
        cal, pol, bt = build()
        assert bt is not None
        bt["expected"]["expected_residual_loss"] = expected
        bt["strategies"]["policy"]["residual_loss"] = realized
        return _build_summary(cal, pol, bt)

    def test_exactly_at_attention_ratio_is_pass(self) -> None:
        expected = 100.0
        realized = ATTENTION_RESIDUAL_RATIO * expected
        check = by_id(self._with(expected, realized))["plan.vs_actual"]
        assert check.severity == "pass"
        assert check.params["ratio"] == realized / expected

    def test_one_ulp_above_attention_ratio_is_warn(self) -> None:
        expected = 100.0
        realized = math.nextafter(ATTENTION_RESIDUAL_RATIO * expected, math.inf)
        summary = self._with(expected, realized)
        assert by_id(summary)["plan.vs_actual"].severity == "warn"
        assert summary.status == "attention"

    def test_better_than_plan_is_pass(self) -> None:
        assert by_id(self._with(100.0, 10.0))["plan.vs_actual"].severity == "pass"

    def test_zero_expected_zero_realized_is_pass_with_null_ratio(self) -> None:
        check = by_id(self._with(0.0, 0.0))["plan.vs_actual"]
        assert check.severity == "pass"
        assert check.params["ratio"] is None

    def test_zero_expected_positive_realized_is_warn_with_null_ratio(self) -> None:
        check = by_id(self._with(0.0, 5.0))["plan.vs_actual"]
        assert check.severity == "warn"
        assert check.params["ratio"] is None

    def test_params_carry_the_raw_numbers(self) -> None:
        check = by_id(self._with(150.0, 400.0))["plan.vs_actual"]
        assert check.params["expected"] == 150.0
        assert check.params["realized"] == 400.0
        assert check.section == "segments"


class TestWorstSegment:
    def test_largest_gap_wins(self) -> None:
        segments = {
            "a": seg(10.0, 20.0),  # gap 10
            "b": seg(10.0, 40.0),  # gap 30
            "c": seg(10.0, 30.0),  # gap 20
        }
        params = by_id(_build_summary(*build(segments=segments)))["plan.vs_actual"].params
        assert params["worst_segment"] == "b"
        assert params["worst_ratio"] == 40.0 / 10.0

    def test_tie_is_broken_by_name_ascending_regardless_of_insertion_order(self) -> None:
        segments = {"b": seg(10.0, 35.0), "a": seg(20.0, 45.0)}  # both gap 25
        params = by_id(_build_summary(*build(segments=segments)))["plan.vs_actual"].params
        assert params["worst_segment"] == "a"

    def test_unseen_at_plan_time_segment_counts_with_zero_expected(self) -> None:
        """A segment only in backtest segments (§9.4) contributes expected 0.0
        and is eligible to win; its ratio is null (divide by zero avoided)."""
        params = by_id(_build_summary(*build(realized_only={"zz_unseen": seg(0.0, 500.0)})))[
            "plan.vs_actual"
        ].params
        assert params["worst_segment"] == "zz_unseen"
        assert params["worst_ratio"] is None

    def test_worst_segment_named_in_detail_only_when_warn(self) -> None:
        calm = by_id(_build_summary(*build()))["plan.vs_actual"]
        assert calm.severity == "pass"
        assert calm.params["worst_segment"] is not None  # computed regardless...
        assert calm.params["worst_segment"] not in calm.detail  # ...but not alarmed

        segments = {"segA": seg(100.0, 400.0), "segB": seg(50.0, 50.0)}
        loud = by_id(_build_summary(*build(segments=segments)))["plan.vs_actual"]
        assert loud.severity == "warn"
        assert "segA" in loud.detail


# ---------------------------------------------------------------------------
# H.1.5 — value vs baselines
# ---------------------------------------------------------------------------


class TestValueVsBaselines:
    def test_beating_both_baselines_is_pass_with_lifts(self) -> None:
        cal, pol, bt = build(random_captured=100.0, raw_captured=50.0)
        check = by_id(_build_summary(cal, pol, bt))["value.vs_baselines"]
        assert check.severity == "pass"
        assert check.params["lift_random"] == 420.0 / 100.0
        assert check.params["lift_raw"] == 420.0 / 50.0
        assert check.section == "results"

    def test_below_random_is_warn(self) -> None:
        check = by_id(_build_summary(*build(random_captured=1000.0)))["value.vs_baselines"]
        assert check.severity == "warn"

    def test_below_raw_confidence_only_is_warn(self) -> None:
        check = by_id(_build_summary(*build(raw_captured=1000.0)))["value.vs_baselines"]
        assert check.severity == "warn"

    def test_tying_a_baseline_is_not_beating_it(self) -> None:
        check = by_id(_build_summary(*build(random_captured=420.0)))["value.vs_baselines"]
        assert check.severity == "warn"

    def test_zero_baselines_still_count_as_beaten(self) -> None:
        """The rule compares captured loss directly: a baseline that caught
        nothing is beaten by any positive capture, even though the lift ratio
        is undefined (null)."""
        check = by_id(_build_summary(*build(random_captured=0.0, raw_captured=0.0)))[
            "value.vs_baselines"
        ]
        assert check.severity == "pass"
        assert check.params["lift_random"] is None
        assert check.params["lift_raw"] is None

    def test_no_true_loss_is_info_not_pass(self) -> None:
        """Nothing was there to catch, so "winning" is not a claim."""
        cal, pol, bt = build()
        assert bt is not None
        bt["total_true_loss"] = 0.0
        check = by_id(_build_summary(cal, pol, bt))["value.vs_baselines"]
        assert check.severity == "info"


# ---------------------------------------------------------------------------
# H.1.6 / H.1.7 — calibration checks
# ---------------------------------------------------------------------------


class TestCalibrationChecks:
    def test_worse_than_raw_is_info_never_warn(self) -> None:
        """Deliberate downgrade from the mockup (§11.5 rationale): ECE noise
        must not turn the status amber. Guard against regressing to warn."""
        summary = _build_summary(*build(ece=0.045, ece_raw=0.040))
        assert by_id(summary)["calibration.vs_raw"].severity == "info"
        assert summary.status == "ok"

    @pytest.mark.parametrize("ece,ece_raw", [(0.04, 0.04), (0.02, 0.13)])
    def test_equal_or_better_than_raw_is_pass(self, ece: float, ece_raw: float) -> None:
        check = by_id(_build_summary(*build(ece=ece, ece_raw=ece_raw)))["calibration.vs_raw"]
        assert check.severity == "pass"
        assert check.section == "calibration"

    def test_ci_margin_emitted_when_point_passes_but_ci_crosses_gate(self) -> None:
        check = by_id(_build_summary(*build(ece=0.04, ci_upper=0.06, max_ece=0.05)))[
            "calibration.ci_margin"
        ]
        assert check.severity == "info"
        assert check.params == {"max_ece": 0.05, "ece_ci_upper": 0.06}

    @pytest.mark.parametrize(
        "ece,ci_upper,max_ece,why",
        [
            (0.04, 0.06, None, "no gate declared"),
            (0.06, 0.07, 0.05, "point estimate already fails"),
            (0.04, 0.05, 0.05, "CI upper == gate is not above it"),
            (0.04, 0.049, 0.05, "CI entirely under the gate"),
        ],
    )
    def test_ci_margin_absent(
        self, ece: float, ci_upper: float, max_ece: float | None, why: str
    ) -> None:
        present = ids(_build_summary(*build(ece=ece, ci_upper=ci_upper, max_ece=max_ece)))
        assert "calibration.ci_margin" not in present, why


# ---------------------------------------------------------------------------
# H.1.8 — budget adherence
# ---------------------------------------------------------------------------


class TestBudgetAdherence:
    @staticmethod
    def _with(planned: int, replayed: int) -> Check:
        cal, pol, bt = build()
        assert bt is not None
        pol["totals"]["n_review"] = planned
        bt["strategies"]["policy"]["n_review"] = replayed
        return by_id(_build_summary(cal, pol, bt))["budget.adherence"]

    def test_matching_plan_is_pass(self) -> None:
        check = self._with(10, 10)
        assert check.severity == "pass"
        assert check.params == {"planned": 10, "replayed": 10, "delta": 0, "ratio": 1.0}

    def test_tie_inflation_is_info_with_ratio_and_delta(self) -> None:
        check = self._with(1000, 1183)
        assert check.severity == "info"
        assert check.params["delta"] == 183
        assert check.params["ratio"] == 1183 / 1000

    def test_fewer_than_planned_is_info_with_negative_delta(self) -> None:
        check = self._with(10, 8)
        assert check.severity == "info"
        assert check.params["delta"] == -2

    def test_zero_planned_has_null_ratio_not_zero_division(self) -> None:
        check = self._with(0, 3)
        assert check.severity == "info"
        assert check.params["ratio"] is None


# ---------------------------------------------------------------------------
# H.1.9 — plan-only
# ---------------------------------------------------------------------------


class TestPlanOnly:
    def test_plan_only_check_set_without_declared_residual_gate(self) -> None:
        summary = _build_summary(*build(with_backtest=False))
        assert summary.mode == "plan_only"
        assert set(ids(summary)) == {
            "gate.max_ece",
            "gate.max_residual_risk",
            "calibration.vs_raw",
            "backtest.not_run",
        }

    def test_plan_only_check_set_with_everything_optional_present(self) -> None:
        summary = _build_summary(*build(with_backtest=False, max_residual=1e12, ci_upper=0.06))
        assert set(ids(summary)) == {
            "gate.max_ece",
            "gate.max_residual_risk.planned",
            "calibration.vs_raw",
            "calibration.ci_margin",
            "backtest.not_run",
        }
        assert not BACKTEST_ONLY & set(ids(summary))

    def test_backtest_not_run_check_is_info(self) -> None:
        check = by_id(_build_summary(*build(with_backtest=False)))["backtest.not_run"]
        assert check.severity == "info"
        assert check.section == "results"

    def test_full_mode_never_has_backtest_not_run(self) -> None:
        assert "backtest.not_run" not in ids(_build_summary(*build()))

    def test_healthy_plan_only_is_ok(self) -> None:
        assert _build_summary(*build(with_backtest=False)).status == "ok"

    def test_plan_only_can_fail_via_ece_gate_alone(self) -> None:
        summary = _build_summary(*build(with_backtest=False, ece=0.2, max_ece=0.05))
        assert summary.status == "failed"

    def test_plan_only_can_fail_via_planned_residual_gate_alone(self) -> None:
        summary = _build_summary(*build(with_backtest=False, max_residual=1.0))
        assert summary.status == "failed"
        assert by_id(summary)["gate.max_residual_risk.planned"].severity == "fail"


# ---------------------------------------------------------------------------
# H.1.10 / H.1.11 — status derivation and ordering
# ---------------------------------------------------------------------------


class TestStatusAndOrdering:
    @pytest.mark.parametrize(
        "severities,expected",
        [
            (["fail"], "failed"),
            (["fail", "warn"], "failed"),
            (["pass", "info", "fail"], "failed"),
            (["warn", "info"], "attention"),
            (["pass", "warn"], "attention"),
            (["info", "pass"], "ok"),
            (["pass"], "ok"),
            (["info"], "ok"),
            ([], "ok"),
        ],
    )
    def test_status_table(self, severities: list[str], expected: str) -> None:
        checks = tuple(make_check(s) for s in severities)
        assert _status_from_checks(checks) == expected

    def test_full_sequence_for_a_fully_populated_fixture(self) -> None:
        """Every severity group populated at once: pins the complete order, so
        a reordering is caught even when no severity changes."""
        segments = {
            "segA": seg(100.0, 150.0, exp_cap=400.0, real_cap=420.0, n_plan=10, n_replay=12),
            "segB": seg(50.0, 50.0),
        }  # realized 200 vs planned 150: ratio 1.33 > 1.25 -> warn
        summary = _build_summary(
            *build(
                segments=segments,
                max_residual=155.0,  # planned 150 passes, realized 200 fails
                ece=0.045,
                ece_raw=0.040,  # calibration worse than raw -> info
                ci_upper=0.06,  # passes 0.05 on the point, CI crosses -> info
            )
        )
        assert summary.status == "failed"
        assert [(c.id, c.severity) for c in summary.checks] == [
            ("gate.max_residual_risk.realized", "fail"),
            ("plan.vs_actual", "warn"),
            ("calibration.vs_raw", "info"),
            ("calibration.ci_margin", "info"),
            ("budget.adherence", "info"),
            ("gate.max_ece", "pass"),
            ("gate.max_residual_risk.planned", "pass"),
            ("value.vs_baselines", "pass"),
        ]

    def test_every_emitted_id_is_in_the_catalogue(self) -> None:
        for kwargs in ({}, {"with_backtest": False}, {"max_residual": 1.0, "ci_upper": 0.06}):
            for check_id in ids(_build_summary(*build(**kwargs))):
                assert check_id in CATALOGUE

    def test_ids_are_unique(self) -> None:
        summary = _build_summary(*build(max_residual=1.0, ci_upper=0.06))
        assert len(ids(summary)) == len(set(ids(summary)))


# ---------------------------------------------------------------------------
# H.1.12 / H.1.13 — purity and text
# ---------------------------------------------------------------------------


class TestPurityAndText:
    @pytest.mark.parametrize("with_backtest", [True, False])
    def test_inputs_are_not_mutated(self, with_backtest: bool) -> None:
        cal, pol, bt = build(
            with_backtest=with_backtest,
            realized_only={"zz": seg(0.0, 1.0)},
            max_residual=1e12,
        )
        snapshot = copy.deepcopy((cal, pol, bt))
        _build_summary(cal, pol, bt)
        assert (cal, pol, bt) == snapshot

    def test_same_inputs_same_summary(self) -> None:
        assert _build_summary(*build()) == _build_summary(*build())

    def test_check_text_is_plain_not_pre_escaped(self) -> None:
        """Escaping is the template's job. If Python pre-escaped, the
        autoescaping template would double-escape (`&amp;lt;`)."""
        name = "<script>alert(1)</script>"
        segments = {name: seg(10.0, 500.0), "segB": seg(50.0, 50.0)}
        check = by_id(_build_summary(*build(segments=segments)))["plan.vs_actual"]
        assert check.severity == "warn"
        assert check.params["worst_segment"] == name
        assert name in check.detail
        assert "&lt;" not in check.detail and "&amp;" not in check.detail

    def test_all_check_text_fields_are_plain_strings(self) -> None:
        for c in _build_summary(*build(max_residual=1.0, ci_upper=0.06)).checks:
            for field_value in (c.title, c.detail, c.value):
                assert isinstance(field_value, str) and field_value
                assert "<" not in field_value  # no markup from Python


# ---------------------------------------------------------------------------
# KPIs (§11.7) — unit level; rendered tiles are covered in tests/unit/test_report.py
# ---------------------------------------------------------------------------


class TestKpis:
    def test_kpis_reuse_the_check_numbers(self) -> None:
        """A tile can never disagree with the check that explains it."""
        summary = _build_summary(*build())
        checks, kpis = by_id(summary), summary.kpis
        assert kpis["lift"]["random"] == checks["value.vs_baselines"].params["lift_random"]
        assert kpis["lift"]["raw"] == checks["value.vs_baselines"].params["lift_raw"]
        assert kpis["residual"]["ratio"] == checks["plan.vs_actual"].params["ratio"]
        assert kpis["residual"]["plan_expected"] == checks["plan.vs_actual"].params["expected"]
        assert kpis["residual"]["value"] == checks["plan.vs_actual"].params["realized"]

    def test_full_mode_uses_replayed_counts_and_realized_loss(self) -> None:
        segments = {"segA": seg(100.0, 110.0, real_cap=420.0, n_plan=10, n_replay=13)}
        cal, pol, bt = build(segments=segments)
        assert bt is not None
        kpis = _build_summary(cal, pol, bt).kpis
        assert kpis["reviewed"]["value"] == 13
        assert kpis["reviewed"]["over_plan"] == 3
        assert kpis["capture"]["value"] == bt["strategies"]["policy"]["capture_rate"]
        assert kpis["capture"]["is_expected"] is False
        assert kpis["residual"]["is_expected"] is False

    def test_over_plan_counted_even_when_nothing_was_planned(self) -> None:
        cal, pol, bt = build()
        assert bt is not None
        pol["totals"]["n_review"] = 0
        bt["strategies"]["policy"]["n_review"] = 4
        assert _build_summary(cal, pol, bt).kpis["reviewed"]["over_plan"] == 4

    def test_no_over_plan_when_matching(self) -> None:
        assert _build_summary(*build()).kpis["reviewed"]["over_plan"] is None

    def test_plan_only_uses_expected_figures(self) -> None:
        cal, pol, _ = build(with_backtest=False)
        kpis = _build_summary(cal, pol, None).kpis
        t = pol["totals"]
        assert kpis["reviewed"]["value"] == t["n_review"]
        assert kpis["reviewed"]["over_plan"] is None
        assert kpis["capture"]["is_expected"] is True
        assert kpis["capture"]["value"] == t["expected_captured_loss"] / (
            t["expected_captured_loss"] + t["expected_residual_loss"]
        )
        assert kpis["lift"] == {"random": None, "raw": None}
        assert kpis["residual"]["is_expected"] is True
        assert kpis["residual"]["value"] == t["expected_residual_loss"]

    def test_zero_totals_give_null_shares_not_errors(self) -> None:
        segments = {"segA": seg(0.0, 0.0, n_total=0)}
        kpis = _build_summary(*build(segments=segments, with_backtest=False)).kpis
        assert kpis["reviewed"]["share"] is None
        assert kpis["capture"]["value"] is None

    @pytest.mark.parametrize(
        "kwargs,expected_color",
        [
            ({}, "neutral"),
            ({"segments": {"segA": seg(100.0, 400.0)}}, "warn"),
            ({"segments": {"segA": seg(100.0, 400.0)}, "max_residual": 300.0}, "fail"),
            ({"with_backtest": False}, "neutral"),
            ({"with_backtest": False, "max_residual": 1.0}, "fail"),
        ],
    )
    def test_residual_colour(self, kwargs: dict[str, Any], expected_color: str) -> None:
        """Red only via a failed residual gate; amber via plan.vs_actual; never
        green (a number alone is not a verdict, §11.7)."""
        assert _build_summary(*build(**kwargs)).kpis["residual"]["color"] == expected_color

    @pytest.mark.parametrize(
        "budget,label",
        [
            ({"type": "fraction", "value": 0.1}, "budget 10%"),
            ({"type": "count", "value": 1000}, "budget 1,000 items"),
            ({"type": "money", "value": 4000.0}, "budget 4,000 cost units"),
            ({}, "budget not set"),
        ],
    )
    def test_budget_label_per_type(self, budget: dict[str, Any], label: str) -> None:
        kpis = _build_summary(*build(budget=budget)).kpis
        assert kpis["reviewed"]["budget_label"] == label


# ---------------------------------------------------------------------------
# H.2 — property tests
# ---------------------------------------------------------------------------

severities_st = st.sampled_from(["fail", "warn", "info", "pass"])
money = st.floats(min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False)
unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@PROPERTY_SETTINGS
@given(st.lists(severities_st, max_size=12))
def test_property_status_is_the_worst_severity(severities: list[str]) -> None:
    """H.2.1."""
    expected = "failed" if "fail" in severities else "attention" if "warn" in severities else "ok"
    assert _status_from_checks(tuple(make_check(s) for s in severities)) == expected


@st.composite
def gate_scenarios(draw: st.DrawFn) -> Triplet:
    """Random gate inputs, with thresholds sometimes drawn EQUAL to their
    value so the strict-`>` boundary is exercised, not just sampled around."""
    ece = draw(unit)
    exp_res = draw(money)
    real_res = draw(money)
    max_ece = draw(st.one_of(st.none(), unit, st.just(ece)))
    max_residual = draw(st.one_of(st.none(), money, st.just(exp_res), st.just(real_res)))
    with_backtest = draw(st.booleans())
    return build(
        ece=ece,
        ece_raw=draw(unit),
        ci_upper=draw(unit),
        max_ece=max_ece,
        max_residual=max_residual,
        segments={"segA": seg(exp_res, real_res, real_cap=draw(money), n_plan=1)},
        random_captured=draw(money),
        raw_captured=draw(money),
        with_backtest=with_backtest,
    )


@PROPERTY_SETTINGS
@given(gate_scenarios())
def test_property_gate_checks_agree_with_gates_table(triplet: Triplet) -> None:
    """H.2.2: the at-a-glance checks and the Gates table (§11.9) are two views
    of one comparison; they must never disagree."""
    cal, pol, bt = triplet
    checks = by_id(_build_summary(cal, pol, bt))
    rows = {r["name"]: r["verdict"] for r in _build_gates_view(cal, pol, bt)["rows"]}
    as_table = {"pass": "PASS", "fail": "FAIL", "info": "not set"}

    assert rows["max_ece"] == as_table[checks["gate.max_ece"].severity]
    if pol["config_snapshot"]["gates"]["max_residual_risk"] is None:
        assert checks["gate.max_residual_risk"].severity == "info"
        assert rows["max_residual_risk (planned)"] == "not set"
        if bt is not None:
            assert rows["max_residual_risk (realized)"] == "not set"
    else:
        planned = checks["gate.max_residual_risk.planned"].severity
        assert rows["max_residual_risk (planned)"] == as_table[planned]
        if bt is not None:
            realized = checks["gate.max_residual_risk.realized"].severity
            assert rows["max_residual_risk (realized)"] == as_table[realized]


@PROPERTY_SETTINGS
@given(gate_scenarios())
def test_property_checks_are_ordered_by_severity_then_catalogue(triplet: Triplet) -> None:
    keys = [
        (SEVERITY_RANK[c.severity], CATALOGUE.index(c.id)) for c in _build_summary(*triplet).checks
    ]
    assert keys == sorted(keys)


segment_names = st.lists(
    st.sampled_from(["alpha", "beta", "gamma", "delta", "<b>x</b>", "default"]),
    min_size=1,
    max_size=5,
    unique=True,
)


@st.composite
def shuffled_pair(draw: st.DrawFn) -> tuple[Triplet, Triplet]:
    names = draw(segment_names)
    segs = {
        n: seg(draw(money), draw(money), real_cap=draw(money), n_plan=draw(st.integers(0, 5)))
        for n in names
    }
    order = draw(st.permutations(names))
    reordered = {n: segs[n] for n in order}
    kwargs = {"max_residual": draw(st.one_of(st.none(), money))}
    return build(segments=segs, **kwargs), build(segments=reordered, **kwargs)


@PROPERTY_SETTINGS
@given(shuffled_pair())
def test_property_segment_insertion_order_is_irrelevant(pair: tuple[Triplet, Triplet]) -> None:
    """H.2.3: the same segments in a different dict order give an identical
    Summary (worst_segment, checks, kpis) and identical segment rows."""
    (cal1, pol1, bt1), (cal2, pol2, bt2) = pair
    assert _build_summary(cal1, pol1, bt1) == _build_summary(cal2, pol2, bt2)
    assert _build_segments_view(pol1, bt1) == _build_segments_view(pol2, bt2)


@st.composite
def degenerate_triplets(draw: st.DrawFn) -> Triplet:
    """Zero everything the spec lets be zero (H.2.4)."""
    zero_or = lambda s: st.one_of(st.just(0.0), s)  # noqa: E731
    n_total = draw(st.sampled_from([0, 1, 50]))
    n_plan = draw(st.integers(0, max(n_total, 0)))
    cal, pol, bt = build(
        ece=draw(zero_or(unit)),
        ece_raw=draw(zero_or(unit)),
        ci_upper=draw(zero_or(unit)),
        max_ece=draw(st.sampled_from([None, 0.0, 1.0])),
        max_residual=draw(st.sampled_from([None, 0.0])),
        segments={
            "s": seg(
                draw(zero_or(money)),
                draw(zero_or(money)),
                exp_cap=draw(zero_or(money)),
                real_cap=draw(zero_or(money)),
                n_plan=n_plan,
                n_replay=draw(st.integers(0, 3)),
                n_total=n_total,
            )
        },
        random_captured=draw(zero_or(money)),
        raw_captured=draw(zero_or(money)),
        with_backtest=draw(st.booleans()),
    )
    pol["config_snapshot"]["segments"]["s"] = {"false_accept_cost": 0.0, "review_cost": 0.0}
    if bt is not None and draw(st.booleans()):
        bt["total_true_loss"] = 0.0
        bt["strategies"]["policy"]["capture_rate"] = None
    return cal, pol, bt


@PROPERTY_SETTINGS
@given(degenerate_triplets())
def test_property_never_raises_on_degenerate_inputs(triplet: Triplet) -> None:
    """H.2.4: zero n_total / total_true_loss / n_review / costs, gates all
    None or all 0 — the summary, segment rows and gates table still build."""
    cal, pol, bt = triplet
    summary = _build_summary(cal, pol, bt)
    assert summary.status in {"ok", "attention", "failed"}
    assert summary.mode == ("full" if bt is not None else "plan_only")
    _build_segments_view(pol, bt)
    _build_gates_view(cal, pol, bt)


@pytest.mark.parametrize("with_backtest", [True, False])
@pytest.mark.parametrize(
    "mode,threshold,expected",
    [
        ("point", 0.05, "pass"),
        ("ci_upper", 0.05, "fail"),
        ("ci_upper", 0.07, "pass"),
    ],
)
def test_ece_mode_controls_summary_and_table(
    with_backtest: bool, mode: str, threshold: float, expected: str
) -> None:
    triplet = build(ece=0.04, ci_upper=0.07, max_ece=threshold, with_backtest=with_backtest)
    triplet[1]["config_snapshot"]["gates"]["ece_gate_mode"] = mode
    summary = _build_summary(*triplet)
    check = by_id(summary)["gate.max_ece"]
    assert check.severity == expected
    assert check.params["value"] == (0.07 if mode == "ci_upper" else 0.04)
    row = _build_gates_view(*triplet)["rows"][0]
    assert row["verdict"] == expected.upper()
    assert (summary.status == "failed") == (expected == "fail")
    if mode == "ci_upper":
        assert "calibration.ci_margin" not in ids(summary)

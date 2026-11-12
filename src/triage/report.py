"""HTML report: at-a-glance summary, policy/backtest evidence, reliability
diagram, coverage-vs-risk curves, and the artifact hash chain.

See docs/interfaces.md §11 (v1.1) for the pinned contract. `render()` is a
PURE function of its arguments: identical inputs produce byte-identical HTML
(all variance is injected via the `created_at` argument, which the CLI sets
to `artifacts.utc_now_iso()`). Plotly figures use explicit `div_id`s so their
serialized HTML fragments are deterministic (no random client-side ids).

Chain verification happens here, not in the CLI: `render` never renders a
chain it has not itself checked (see `_verify_chain`).

The at-a-glance section (status, headline, KPIs, checks) is built by
`_build_summary`, a pure function of the three artifact dicts that never
raises and never renders HTML itself (§11.5.1): it returns a `Summary` of
plain-text `Check`s that the template turns into markup, so artifact-derived
text (e.g. segment names) is always escaped at the template boundary, never
pre-escaped or pre-styled in Python.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

import plotly.graph_objects as go
import plotly.offline
from jinja2 import Environment, PackageLoader, select_autoescape

from triage.artifacts import content_hash
from triage.exceptions import ContractViolation

__all__ = ["render"]

# Palette (interfaces.md §11.11.1). Status colours (ok/attention/failed) are
# reserved for verdicts, so no data series below uses green, amber or red.
_COLOR_POLICY = "#2a78d6"
_COLOR_RAW = "#7c5cc4"
_COLOR_RANDOM = "#9aa1a9"
_STATUS_COLOR = {"ok": "#15803d", "attention": "#d97706", "failed": "#dc2626"}
_NEUTRAL = "#64748b"
_INK = "#1f2933"
_GRID_COLOR = "#e7eaee"
_BG_COLOR = "#fcfcfb"

_FONT = {
    "family": "-apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif",
    "size": 12,
    "color": _INK,
}

_TEMPLATE_NAME = "report.html.j2"

# Display-only: never affects exit codes (those stay in plan/backtest's
# declared gates). Provisional; changing it is not a spec change. §11.5.
ATTENTION_RESIDUAL_RATIO = 1.25

Severity = Literal["fail", "warn", "info", "pass"]
_SEVERITY_ORDER: dict[Severity, int] = {"fail": 0, "warn": 1, "info": 2, "pass": 3}

# Catalogue order (interfaces.md §11.5 table); used as the tie-break within
# a severity group so check order is fully deterministic.
_CHECK_ORDER = [
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


def _verify_chain(
    calibrator_dict: dict[str, Any],
    policy_dict: dict[str, Any],
    backtest_dict: dict[str, Any] | None,
) -> str:
    """Verify the artifact hash chain; return the calibrator's content_hash.

    Raises ContractViolation (naming both hashes) on any mismatch. See
    docs/interfaces.md §11.2.
    """
    calibrator_hash = content_hash(calibrator_dict)
    declared_calibrator_hash = policy_dict.get("input_sha256s", {}).get("calibrator")
    if declared_calibrator_hash != calibrator_hash:
        raise ContractViolation(
            "Report chain broken: policy.input_sha256s.calibrator="
            f"{declared_calibrator_hash!r} but content_hash(calibrator)={calibrator_hash!r}"
        )

    # Policy self-check: the artifact must hash to its own declared policy_hash.
    declared_policy_hash = policy_dict.get("policy_hash")
    policy_hash = content_hash(policy_dict)
    if policy_hash != declared_policy_hash:
        raise ContractViolation(
            "Report chain broken: policy.policy_hash="
            f"{declared_policy_hash!r} but content_hash(policy)={policy_hash!r}"
        )

    if backtest_dict is not None:
        declared_bt_policy_hash = backtest_dict.get("input_sha256s", {}).get("policy")
        if declared_bt_policy_hash != policy_hash:
            raise ContractViolation(
                "Report chain broken: backtest.input_sha256s.policy="
                f"{declared_bt_policy_hash!r} but policy.policy_hash={policy_hash!r}"
            )
        declared_bt_calibrator_hash = backtest_dict.get("input_sha256s", {}).get("calibrator")
        if declared_bt_calibrator_hash != calibrator_hash:
            raise ContractViolation(
                "Report chain broken: backtest.input_sha256s.calibrator="
                f"{declared_bt_calibrator_hash!r} but content_hash(calibrator)={calibrator_hash!r}"
            )

        # Backtest self-check: the artifact must hash to its own declared backtest_hash.
        declared_backtest_hash = backtest_dict.get("backtest_hash")
        backtest_hash = content_hash(backtest_dict)
        if backtest_hash != declared_backtest_hash:
            raise ContractViolation(
                "Report chain broken: backtest.backtest_hash="
                f"{declared_backtest_hash!r} but content_hash(backtest)={backtest_hash!r}"
            )

    return calibrator_hash


# ---------------------------------------------------------------------------
# Number formatting (§11.11.3) -- locale-free, pure; registered as Jinja
# filters AND used directly when building Check/Summary text below, so a
# number reads the same in a sentence as it does in a table cell.
# ---------------------------------------------------------------------------


def fmt_count(value: float | int | None) -> str:
    return "\u2014" if value is None else f"{int(round(value)):,d}"


def fmt_cost(value: float | None) -> str:
    return "\u2014" if value is None else f"{value:,.0f}"


def fmt_pct(value: float | None, decimals: int = 1) -> str:
    return "\u2014" if value is None else f"{value * 100:.{decimals}f}%"


def fmt_ratio(value: float | None) -> str:
    return "\u2014" if value is None else f"{value:.2f}\u00d7"


def fmt_lift(value: float | None) -> str:
    if value is None:
        return "\u2014"
    return f"{value:.1f}\u00d7" if value < 10 else f"{value:.0f}\u00d7"


def fmt_ece(value: float | None) -> str:
    return "\u2014" if value is None else f"{value:.3f}"


def _base_layout(**overrides: Any) -> dict[str, Any]:
    """No fixed width (§11.11): figures fill their container; only height is
    pinned. `to_html`'s `config={"responsive": True}` (set by callers) makes
    plotly resize the div on the client without changing the serialized
    figure spec, so output stays byte-identical for identical inputs."""
    layout: dict[str, Any] = {
        "autosize": True,
        "plot_bgcolor": _BG_COLOR,
        "paper_bgcolor": _BG_COLOR,
        "font": dict(_FONT),
        "margin": {"l": 60, "r": 110, "t": 40, "b": 50},
        "showlegend": False,
    }
    layout.update(overrides)
    return layout


_PLOTLY_CONFIG = {"responsive": True, "displaylogo": False}


def _to_html(fig: go.Figure, *, div_id: str, include_plotlyjs: bool) -> str:
    html: str = fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        div_id=div_id,
        config=_PLOTLY_CONFIG,
    )
    return html


def _reliability_figure(calibrator_dict: dict[str, Any], *, div_id: str) -> str:
    """Before/after reliability; marker AREA (not radius) proportional to the
    bin's item count, so a 5-item bin does not visually outweigh a 500-item
    one (interfaces.md §11.10)."""
    bins = calibrator_dict["reliability_bins"]
    before = [b for b in bins["before"] if b["count"] > 0]
    after = [b for b in bins["after"] if b["count"] > 0]
    max_count = max([b["count"] for b in before + after], default=1)

    def sizes(bb: list[dict[str, Any]]) -> list[float]:
        return [4.0 + 16.0 * (b["count"] / max_count) ** 0.5 for b in bb]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="perfect calibration",
            line={"color": _NEUTRAL, "width": 1, "dash": "dot"},
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[b["mean_pred"] for b in before],
            y=[b["mean_observed"] for b in before],
            mode="lines+markers",
            name="raw score",
            line={"color": _COLOR_RAW, "width": 1.5},
            marker={"color": _COLOR_RAW, "size": sizes(before)},
            customdata=[b["count"] for b in before],
            hovertemplate="predicted %{x:.2f} / observed %{y:.2f} / n=%{customdata}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[b["mean_pred"] for b in after],
            y=[b["mean_observed"] for b in after],
            mode="lines+markers",
            name="calibrated",
            line={"color": _COLOR_POLICY, "width": 1.5},
            marker={"color": _COLOR_POLICY, "size": sizes(after)},
            customdata=[b["count"] for b in after],
            hovertemplate="predicted %{x:.2f} / observed %{y:.2f} / n=%{customdata}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            xaxis={
                "title": {"text": "mean predicted error rate"},
                "range": [0, 1],
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
            },
            yaxis={
                "title": {"text": "mean observed error rate"},
                "range": [0, 1],
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
            },
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            margin={"l": 50, "r": 20, "t": 30, "b": 45},
            height=300,
        )
    )
    return _to_html(fig, div_id=div_id, include_plotlyjs=False)


def _capture_figure(backtest_dict: dict[str, Any], *, div_id: str) -> str:
    """Horizontal bars: share of loss caught at EQUAL review effort, with the
    policy's lift over each baseline annotated directly (no legend needed;
    replaces v1's absolute-captured-loss bars). interfaces.md §11.10."""
    strategies = backtest_dict["strategies"]
    policy_rate = strategies["policy"]["capture_rate"]
    raw_rate = strategies["raw_confidence"]["capture_rate"]
    random_rate = strategies["random"]["capture_rate"]
    total = backtest_dict["total_true_loss"]
    random_std_rate = (strategies["random"]["captured_loss_std"] / total) if total else 0.0

    rows = [
        ("Random review", random_rate, _COLOR_RANDOM, random_std_rate, policy_rate, raw_rate),
        ("Raw model confidence", raw_rate, _COLOR_RAW, 0.0, policy_rate, raw_rate),
        ("Triage policy", policy_rate, _COLOR_POLICY, 0.0, None, None),
    ]
    labels = [r[0] for r in rows]
    values = [0.0 if r[1] is None else r[1] * 100.0 for r in rows]
    colors = [r[2] for r in rows]
    errors = [r[3] * 100.0 for r in rows]
    text = []
    for _, rate, _c, _e, p_rate, _raw in rows:
        if rate is None or p_rate is None:
            text.append(f"{fmt_pct(rate)}" if rate is not None else "\u2014")
        elif rate > 0 and p_rate is not None:
            lift = p_rate / rate if rate else None
            text.append(f"{fmt_pct(rate)} \u00b7 policy {fmt_lift(lift)} better")
        else:
            text.append(fmt_pct(rate))
    text[-1] = fmt_pct(policy_rate)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            error_x={"type": "data", "array": errors, "visible": True, "color": _INK},
            text=text,
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            xaxis={
                "title": {"text": "share of loss caught"},
                "range": [0, 100],
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
                "ticksuffix": "%",
            },
            yaxis={"automargin": True},
            height=230,
            margin={"l": 140, "r": 60, "t": 10, "b": 40},
        )
    )
    return _to_html(fig, div_id=div_id, include_plotlyjs=False)


def _coverage_figure(backtest_dict: dict[str, Any], *, budget_fraction: float, div_id: str) -> str:
    curves = backtest_dict["curves"]
    grid_pct = [g * 100.0 for g in curves["grid"]]
    series = [
        ("random", "Random", _COLOR_RANDOM, "dot"),
        ("raw_confidence", "Raw confidence", _COLOR_RAW, "dash"),
        ("policy", "Triage policy", _COLOR_POLICY, "solid"),
    ]

    fig = go.Figure()
    for key, name, color, dash in series:
        values = [v * 100.0 for v in curves[key]]
        fig.add_trace(
            go.Scatter(
                x=grid_pct,
                y=values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2.4 if key == "policy" else 1.8, "dash": dash},
                hovertemplate=f"{name}: "
                + "%{x:.0f}% reviewed \u2192 %{y:.1f}% caught<extra></extra>",
            )
        )

    in_range = 0.0 <= budget_fraction <= curves["grid"][-1]
    if in_range:
        idx = min(
            range(len(curves["grid"])), key=lambda i: abs(curves["grid"][i] - budget_fraction)
        )
        fig.add_vline(
            x=budget_fraction * 100.0,
            line_width=1,
            line_color=_INK,
            line_dash="dot",
            annotation_text=f"your budget ({fmt_pct(budget_fraction, 0)})",
            annotation_position="top",
            annotation_font_size=11,
        )
        for key, name, color, _dash in series:
            y = curves[key][idx] * 100.0
            fig.add_annotation(
                x=budget_fraction * 100.0,
                y=y,
                text=f"{y:.0f}%",
                showarrow=False,
                xshift=18,
                font={"color": color, "size": 11},
            )

    fig.update_layout(
        **_base_layout(
            xaxis={
                "title": {"text": "share of items reviewed"},
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
                "ticksuffix": "%",
            },
            yaxis={
                "title": {"text": "share of loss caught"},
                "range": [0, 105],
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
                "ticksuffix": "%",
            },
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            margin={"l": 55, "r": 20, "t": 30, "b": 45},
            height=320,
        )
    )
    return _to_html(fig, div_id=div_id, include_plotlyjs=False)


def _segments_figure(
    policy_dict: dict[str, Any], backtest_dict: dict[str, Any], *, div_id: str
) -> str:
    """Per segment: unreviewed risk, plan (outlined) vs. actual (solid),
    sorted to match the per-segment table (§11.8)."""
    policy_segments = policy_dict.get("segments", {}) or {}
    realized_segments = backtest_dict["strategies"]["policy"].get("segments", {}) or {}
    names = sorted(
        set(policy_segments) | set(realized_segments),
        key=lambda n: (-float(realized_segments.get(n, {}).get("residual_loss") or 0.0), n),
    )
    expected = [
        float(policy_segments.get(n, {}).get("expected_residual_loss") or 0.0) for n in names
    ]
    realized = [float(realized_segments.get(n, {}).get("residual_loss") or 0.0) for n in names]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=expected,
            y=names,
            orientation="h",
            name="plan expected",
            marker={"color": "rgba(0,0,0,0)", "line": {"color": _INK, "width": 1.5}},
            hovertemplate="%{y} (plan): %{x:,.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=realized,
            y=names,
            orientation="h",
            name="actual",
            marker={"color": _COLOR_POLICY},
            hovertemplate="%{y} (actual): %{x:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            barmode="group",
            xaxis={
                "title": {"text": "unreviewed risk (cost units)"},
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
            },
            yaxis={"automargin": True, "autorange": "reversed"},
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            margin={"l": 90, "r": 20, "t": 30, "b": 45},
            height=80 + 46 * max(len(names), 1),
        )
    )
    return _to_html(fig, div_id=div_id, include_plotlyjs=False)


def _gate_verdict(threshold: float | None, value: float | None) -> Severity:
    """The ONE comparison both the Gates table (§11.9) and the at-a-glance
    checks (§11.5) use, so they can never disagree: "info" (not set) when
    the threshold is null/absent, else strict `>` (matching
    calibration.check_ece_gate): "fail" if value > threshold, else "pass"."""
    if threshold is None or value is None:
        return "info"
    return "fail" if value > threshold else "pass"


def _gate_row(
    name: str,
    threshold: float | None,
    value: float | None,
    *,
    kind: str,
    checked: str,
    verdict_override: str | None = None,
) -> dict[str, Any]:
    """Build one verdict-bearing gates row: FAIL iff value > threshold (strict,
    matching calibration.check_ece_gate), "not set" if threshold is null/absent.
    `kind` ("ece" or "loss") selects the template's number format; `checked`
    names the exact artifact field whose value is compared to the threshold."""
    verdict = (
        verdict_override
        or {"fail": "FAIL", "pass": "PASS", "info": "not set"}[_gate_verdict(threshold, value)]
    )
    return {
        "name": name,
        "threshold": threshold,
        "value": value,
        "verdict": verdict,
        "kind": kind,
        "checked": checked,
    }


def _ci_level_label(calibrator_dict: dict[str, Any]) -> str:
    """The calibrator's bootstrap CI level as a percentage label, e.g. "95%"."""
    ci_level = calibrator_dict.get("config", {}).get("ci_level")
    if not isinstance(ci_level, (int, float)):
        return "bootstrap"
    return f"{ci_level * 100:g}%"


def _ece_gate_row(calibrator_dict: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """The max_ece row. The statistic follows `gates.ece_gate_mode` from the
    policy's config_snapshot; an absent key means "point" (the default is
    omitted from snapshots, and pre-existing policies never carry it)."""
    max_ece = gates.get("max_ece")
    mode = gates.get("ece_gate_mode", "point")
    if mode == "point":
        return _gate_row(
            "max_ece",
            max_ece,
            calibrator_dict["ece_holdout"],
            kind="ece",
            checked="calibrator.ece_holdout (point estimate; ece_gate_mode=point)",
        )
    if mode == "ci_upper":
        return _gate_row(
            "max_ece",
            max_ece,
            calibrator_dict["ece_ci"][1],
            kind="ece",
            checked=(
                f"calibrator.ece_ci[1] (upper bound of {_ci_level_label(calibrator_dict)} "
                "bootstrap CI; ece_gate_mode=ci_upper)"
            ),
        )
    # Unreachable for policies written by `plan` (costs.Gates validates the
    # mode); still never raise here and never invent a verdict.
    return _gate_row(
        "max_ece",
        max_ece,
        None,
        kind="ece",
        checked=f"unrecognized ece_gate_mode {mode!r}",
        verdict_override="unknown",
    )


def _build_gates_view(
    calibrator_dict: dict[str, Any],
    policy_dict: dict[str, Any],
    backtest_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    """Purely informational gates table per docs/interfaces.md §11.9. Never
    raises: absent/None thresholds render as "not set", not an error."""
    gates = policy_dict.get("config_snapshot", {}).get("gates", {}) or {}
    max_residual_risk = gates.get("max_residual_risk")

    rows: list[dict[str, Any]] = [
        _ece_gate_row(calibrator_dict, gates),
        _gate_row(
            "max_residual_risk (planned)",
            max_residual_risk,
            policy_dict.get("totals", {}).get("expected_residual_loss"),
            kind="loss",
            checked="policy.totals.expected_residual_loss",
        ),
    ]

    adherence: dict[str, Any] | None = None
    if backtest_dict is not None:
        policy_strategy = backtest_dict["strategies"]["policy"]
        rows.append(
            _gate_row(
                "max_residual_risk (realized)",
                max_residual_risk,
                policy_strategy["residual_loss"],
                kind="loss",
                checked="backtest.strategies.policy.residual_loss",
            )
        )
        planned_n_review = int(policy_dict.get("totals", {}).get("n_review", 0) or 0)
        replayed_n_review = int(policy_strategy["n_review"])
        ratio = (replayed_n_review / planned_n_review) if planned_n_review else None
        adherence = {
            "planned_n_review": planned_n_review,
            "replayed_n_review": replayed_n_review,
            "ratio": ratio,
        }

    return {"rows": rows, "adherence": adherence}


# ---------------------------------------------------------------------------
# At-a-glance: Check / Summary (§11.5) -- pure, never raises, plain text only.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    id: str
    severity: Severity
    title: str
    detail: str
    value: str
    section: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Summary:
    status: Literal["ok", "attention", "failed"]
    mode: Literal["full", "plan_only"]
    checks: tuple[Check, ...]
    kpis: dict[str, Any]


def _sort_checks(checks: list[Check]) -> tuple[Check, ...]:
    order = {cid: i for i, cid in enumerate(_CHECK_ORDER)}
    return tuple(sorted(checks, key=lambda c: (_SEVERITY_ORDER[c.severity], order[c.id])))


def _status_from_checks(checks: tuple[Check, ...]) -> Literal["ok", "attention", "failed"]:
    severities = {c.severity for c in checks}
    if "fail" in severities:
        return "failed"
    if "warn" in severities:
        return "attention"
    return "ok"


def _worst_segment(
    policy_segments: dict[str, Any], realized_segments: dict[str, Any]
) -> tuple[str, float, float] | None:
    """(name, expected_residual, realized_residual) with the largest realized
    - expected gap; ties broken by name ASC (§11.5). A segment missing from
    either side contributes 0.0 on that side (e.g. unseen-at-plan-time
    segments, §9.4). None if there are no segments at all."""
    names = sorted(set(policy_segments) | set(realized_segments))
    if not names:
        return None

    def expected(name: str) -> float:
        return float(policy_segments.get(name, {}).get("expected_residual_loss") or 0.0)

    def realized(name: str) -> float:
        return float(realized_segments.get(name, {}).get("residual_loss") or 0.0)

    # sorted(names) is ascending, and max() returns the FIRST maximal item
    # it sees -> smallest name among ties, matching the name-ASC tie-break.
    best = max(names, key=lambda n: realized(n) - expected(n))
    return best, expected(best), realized(best)


def _budget_label(policy_dict: dict[str, Any]) -> str:
    budget = policy_dict.get("budget", {}) or {}
    kind, value = budget.get("type"), budget.get("value")
    if kind == "fraction" and value is not None:
        return f"budget {fmt_pct(value, 0)}"
    if kind == "count" and value is not None:
        return f"budget {fmt_count(value)} items"
    if kind == "money" and value is not None:
        return f"budget {fmt_cost(value)} cost units"
    return "budget not set"


def _gate_checks(
    calibrator_dict: dict[str, Any],
    policy_dict: dict[str, Any],
    backtest_dict: dict[str, Any] | None,
) -> list[Check]:
    gates = policy_dict.get("config_snapshot", {}).get("gates") or {}
    max_ece = gates.get("max_ece")
    max_residual = gates.get("max_residual_risk")
    totals = policy_dict.get("totals", {}) or {}
    checks: list[Check] = []

    ece_row = _ece_gate_row(calibrator_dict, gates)
    ece = ece_row["value"]
    ece_sev: Severity = (
        "fail"
        if ece_row["verdict"] == "FAIL"
        else "pass"
        if ece_row["verdict"] == "PASS"
        else "info"
    )
    checked = ece_row["checked"]
    checks.append(
        Check(
            id="gate.max_ece",
            severity=ece_sev,
            title={
                "fail": "Calibration error is above the declared limit",
                "pass": "Calibration error is within the declared limit",
                "info": (
                    "Unknown calibration gate mode"
                    if ece_row["verdict"] == "unknown"
                    else "No limit declared for calibration quality"
                ),
            }[ece_sev],
            detail=(
                f"{checked}; gates.max_ece is not set or the mode is unrecognized. "
                f"No calibration verdict is available (value {fmt_ece(ece)})."
                if ece_sev == "info"
                else f"{checked}: {fmt_ece(ece)} vs. gates.max_ece {fmt_ece(max_ece)}."
            ),
            value=fmt_ece(ece) if ece_sev == "info" else f"{fmt_ece(ece)} / {fmt_ece(max_ece)}",
            section="gates",
            params={"threshold": max_ece, "value": ece},
        )
    )

    expected_residual_total = totals.get("expected_residual_loss")
    if max_residual is None:
        checks.append(
            Check(
                id="gate.max_residual_risk",
                severity="info",
                title="No limit declared for unreviewed risk",
                detail=(
                    "gates.max_residual_risk is not set, so CI cannot fail on unreviewed "
                    "risk. Consider declaring one in costs.yaml."
                ),
                value="not set",
                section="gates",
                params={"threshold": None},
            )
        )
    else:
        planned_sev = _gate_verdict(max_residual, expected_residual_total)
        checks.append(
            Check(
                id="gate.max_residual_risk.planned",
                severity=planned_sev,
                title={
                    "fail": "Planned unreviewed risk is above the declared limit",
                    "pass": "Planned unreviewed risk is within the declared limit",
                }[planned_sev],
                detail=(
                    f"Planned residual loss {fmt_cost(expected_residual_total)} vs. the "
                    f"gates.max_residual_risk limit of {fmt_cost(max_residual)}."
                ),
                value=f"{fmt_cost(expected_residual_total)} / {fmt_cost(max_residual)}",
                section="gates",
                params={"threshold": max_residual, "value": expected_residual_total},
            )
        )
        if backtest_dict is not None:
            realized = backtest_dict["strategies"]["policy"]["residual_loss"]
            realized_sev = _gate_verdict(max_residual, realized)
            checks.append(
                Check(
                    id="gate.max_residual_risk.realized",
                    severity=realized_sev,
                    title={
                        "fail": "Unreviewed risk is above the declared limit",
                        "pass": "Unreviewed risk is within the declared limit",
                    }[realized_sev],
                    detail=(
                        f"Realized residual loss {fmt_cost(realized)} vs. the "
                        f"gates.max_residual_risk limit of {fmt_cost(max_residual)}. "
                        f"triage backtest exits with code 2 on this."
                    ),
                    value=f"{fmt_cost(realized)} / {fmt_cost(max_residual)}",
                    section="gates",
                    params={"threshold": max_residual, "value": realized},
                )
            )
    return checks


def _backtest_checks(policy_dict: dict[str, Any], backtest_dict: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    strategies = backtest_dict["strategies"]
    policy_strategy = strategies["policy"]

    expected = backtest_dict["expected"]["expected_residual_loss"]
    realized = policy_strategy["residual_loss"]
    ratio = (realized / expected) if expected else None
    is_warn = (realized > 0) if not expected else (realized > ATTENTION_RESIDUAL_RATIO * expected)
    worst = _worst_segment(
        policy_dict.get("segments", {}) or {}, policy_strategy.get("segments", {}) or {}
    )
    worst_name, worst_expected, worst_realized = worst if worst else (None, 0.0, 0.0)
    worst_ratio = (worst_realized / worst_expected) if worst_expected else None
    checks.append(
        Check(
            id="plan.vs_actual",
            severity="warn" if is_warn else "pass",
            title=(
                f"Actual risk is {fmt_ratio(ratio)} higher than planned"
                if is_warn
                else "Actual risk matches the plan"
            ),
            detail=(
                f"Unreviewed errors cost {fmt_cost(realized)} cost units; the plan expected "
                f"{fmt_cost(expected)}."
                + (
                    f" Largest gap: {worst_name} ({fmt_cost(worst_realized)} vs. "
                    f"{fmt_cost(worst_expected)}, {fmt_ratio(worst_ratio)})."
                    if is_warn and worst_name is not None
                    else ""
                )
            ),
            value=fmt_ratio(ratio),
            section="segments",
            params={
                "expected": expected,
                "realized": realized,
                "ratio": ratio,
                "worst_segment": worst_name,
                "worst_ratio": worst_ratio,
            },
        )
    )

    total_true_loss = backtest_dict["total_true_loss"]
    random_mean = strategies["random"]["captured_loss_mean"]
    raw_captured = strategies["raw_confidence"]["captured_loss"]
    policy_captured = policy_strategy["captured_loss"]
    lift_random = (policy_captured / random_mean) if random_mean else None
    lift_raw = (policy_captured / raw_captured) if raw_captured else None
    capture_rate = policy_strategy.get("capture_rate")
    if total_true_loss == 0:
        value_sev: Severity = "info"
        title = "No realized loss in this history to catch"
        detail = "total_true_loss is 0 on this history; the comparison is not meaningful."
    elif policy_captured > random_mean and policy_captured > raw_captured:
        # Spec rule (§11.5) compares captured loss directly, NOT lift ratios:
        # a zero baseline (lift undefined) must still count as "beaten".
        value_sev = "pass"
        title = (
            f"Policy catches {fmt_lift(lift_random)} more loss than random review"
            if lift_random is not None
            else "Policy catches loss that random review misses"
        )
        detail = (
            f"{fmt_pct(capture_rate)} of loss caught vs. random and raw-confidence routing "
            f"at the same review effort."
        )
    else:
        value_sev = "warn"
        title = "Policy does not clearly beat the baselines"
        detail = "Captured loss is not clearly above random / raw-confidence routing."
    checks.append(
        Check(
            id="value.vs_baselines",
            severity=value_sev,
            title=title,
            detail=detail,
            value=fmt_pct(capture_rate),
            section="results",
            params={"capture_rate": capture_rate, "lift_random": lift_random, "lift_raw": lift_raw},
        )
    )

    planned_n = int(policy_dict.get("totals", {}).get("n_review") or 0)
    replayed_n = int(policy_strategy["n_review"])
    adherence_ratio = (replayed_n / planned_n) if planned_n else None
    adherence_pass = replayed_n == planned_n
    checks.append(
        Check(
            id="budget.adherence",
            severity="pass" if adherence_pass else "info",
            title=(
                "Reviews matched the plan"
                if adherence_pass
                else f"{fmt_count(abs(replayed_n - planned_n))} more reviews than planned"
                if replayed_n > planned_n
                else f"{fmt_count(abs(replayed_n - planned_n))} fewer reviews than planned"
            ),
            detail=(
                f"{fmt_count(replayed_n)} items reviewed vs. {fmt_count(planned_n)} planned "
                f"(items tying exactly at the threshold can inflate the replayed count, §9.4)."
            ),
            value="\u2014" if adherence_ratio is None else f"{adherence_ratio:.2f}\u00d7",
            section="results",
            params={
                "planned": planned_n,
                "replayed": replayed_n,
                "delta": replayed_n - planned_n,
                "ratio": adherence_ratio,
            },
        )
    )
    return checks


def _calibration_checks(
    calibrator_dict: dict[str, Any], policy_dict: dict[str, Any]
) -> list[Check]:
    checks: list[Check] = []
    ece, ece_raw = calibrator_dict["ece_holdout"], calibrator_dict["ece_raw_holdout"]
    improved = ece <= ece_raw
    checks.append(
        Check(
            id="calibration.vs_raw",
            severity="pass" if improved else "info",
            title=(
                "Calibration corrected the model's overconfidence"
                if improved
                else "Calibration did not improve on raw scores"
            ),
            detail=f"ECE {fmt_ece(ece_raw)} (raw) \u2192 {fmt_ece(ece)} (calibrated).",
            value=f"{fmt_ece(ece_raw)} \u2192 {fmt_ece(ece)}",
            section="calibration",
            params={"ece_raw_holdout": ece_raw, "ece_holdout": ece},
        )
    )

    gates = policy_dict.get("config_snapshot", {}).get("gates") or {}
    max_ece = gates.get("max_ece")
    ci_upper = calibrator_dict["ece_ci"][1]
    if gates.get("ece_gate_mode", "point") == "point" and (
        max_ece is not None and ece <= max_ece < ci_upper
    ):
        checks.append(
            Check(
                id="calibration.ci_margin",
                severity="info",
                title="Calibration passes its gate, but with a thin margin",
                detail=(
                    f"ECE {fmt_ece(ece)} \u2264 {fmt_ece(max_ece)}, yet the "
                    f"{_ci_level_label(calibrator_dict)} interval reaches "
                    f"{fmt_ece(ci_upper)}. A larger golden set would make this verdict more "
                    f"reliable (risk-register R3)."
                ),
                value=f"CI \u2264 {fmt_ece(ci_upper)}",
                section="calibration",
                params={"max_ece": max_ece, "ece_ci_upper": ci_upper},
            )
        )
    return checks


def _build_summary(
    calibrator_dict: dict[str, Any],
    policy_dict: dict[str, Any],
    backtest_dict: dict[str, Any] | None,
) -> Summary:
    """Pure, never raises. §11.5."""
    mode: Literal["full", "plan_only"] = "full" if backtest_dict is not None else "plan_only"
    checks: list[Check] = _gate_checks(calibrator_dict, policy_dict, backtest_dict)
    checks += _calibration_checks(calibrator_dict, policy_dict)
    if backtest_dict is not None:
        checks += _backtest_checks(policy_dict, backtest_dict)
    else:
        checks.append(
            Check(
                id="backtest.not_run",
                severity="info",
                title="Backtest not run",
                detail=(
                    "This report reflects the plan-only pipeline (calibrator + policy). Run "
                    "triage backtest and re-run triage report to validate it against real "
                    "outcomes."
                ),
                value="plan only",
                section="results",
            )
        )
    ordered = _sort_checks(checks)
    status = _status_from_checks(ordered)
    kpis = _build_kpis(policy_dict, backtest_dict, {c.id: c for c in ordered})
    return Summary(status=status, mode=mode, checks=ordered, kpis=kpis)


def _build_kpis(
    policy_dict: dict[str, Any],
    backtest_dict: dict[str, Any] | None,
    checks_by_id: dict[str, Check],
) -> dict[str, Any]:
    """§11.7. Reuses numbers already computed for the checks (`checks_by_id`)
    so a KPI tile can never disagree with the check that explains it."""
    totals = policy_dict.get("totals", {}) or {}
    planned_n, n_total = totals.get("n_review"), totals.get("n_total")
    budget_label = _budget_label(policy_dict)

    if backtest_dict is None:
        expected_captured = totals.get("expected_captured_loss")
        expected_residual = totals.get("expected_residual_loss")
        denom = (expected_captured or 0.0) + (expected_residual or 0.0)
        return {
            "reviewed": {
                "value": planned_n,
                "total": n_total,
                "share": (planned_n / n_total) if n_total else None,
                "budget_label": budget_label,
                "over_plan": None,
            },
            "capture": {
                "value": ((expected_captured or 0.0) / denom) if denom else None,
                "captured": expected_captured,
                "denom": denom,
                "is_expected": True,
            },
            "lift": {"random": None, "raw": None},
            "residual": {
                "value": expected_residual,
                "plan_expected": None,
                "ratio": None,
                "limit": (policy_dict.get("config_snapshot", {}).get("gates") or {}).get(
                    "max_residual_risk"
                ),
                "is_expected": True,
                "color": "fail"
                if checks_by_id.get("gate.max_residual_risk.planned")
                and checks_by_id["gate.max_residual_risk.planned"].severity == "fail"
                else "neutral",
            },
        }

    policy_strategy = backtest_dict["strategies"]["policy"]
    replayed_n = policy_strategy["n_review"]
    baselines = checks_by_id["value.vs_baselines"].params
    plan_vs_actual = checks_by_id["plan.vs_actual"].params
    residual_color = "neutral"
    if any(
        checks_by_id.get(cid) and checks_by_id[cid].severity == "fail"
        for cid in ("gate.max_residual_risk.planned", "gate.max_residual_risk.realized")
    ):
        residual_color = "fail"
    elif checks_by_id["plan.vs_actual"].severity == "warn":
        residual_color = "warn"

    return {
        "reviewed": {
            "value": replayed_n,
            "total": n_total,
            "share": (replayed_n / n_total) if n_total else None,
            "budget_label": budget_label,
            "over_plan": (
                (replayed_n - (planned_n or 0)) if replayed_n > (planned_n or 0) else None
            ),
        },
        "capture": {
            "value": policy_strategy.get("capture_rate"),
            "captured": policy_strategy["captured_loss"],
            "denom": backtest_dict["total_true_loss"],
            "is_expected": False,
        },
        "lift": {"random": baselines["lift_random"], "raw": baselines["lift_raw"]},
        "residual": {
            "value": policy_strategy["residual_loss"],
            "plan_expected": plan_vs_actual["expected"],
            "ratio": plan_vs_actual["ratio"],
            "limit": (policy_dict.get("config_snapshot", {}).get("gates") or {}).get(
                "max_residual_risk"
            ),
            "is_expected": False,
            "color": residual_color,
        },
    }


def _segment_cost(config_snapshot: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Mirrors `CostModel.costs_for` (§3) for display: exact key, else
    'default', else None (rendered as "—", never a crash -- if an artifact
    reached `render` its costs already resolved successfully at plan/backtest
    time; this is a display-only best-effort lookup)."""
    segments = config_snapshot.get("segments", {}) or {}
    return segments.get(name) or segments.get("default")


def _build_segments_view(
    policy_dict: dict[str, Any], backtest_dict: dict[str, Any] | None
) -> dict[str, Any]:
    """§11.8: one row per segment plus a totals row."""
    config_snapshot = policy_dict.get("config_snapshot", {}) or {}
    policy_segments = policy_dict.get("segments", {}) or {}
    realized_segments = (
        backtest_dict["strategies"]["policy"].get("segments", {}) or {} if backtest_dict else {}
    )
    names = sorted(set(policy_segments) | set(realized_segments))
    if backtest_dict is not None:
        names.sort(key=lambda n: -float(realized_segments.get(n, {}).get("residual_loss") or 0.0))

    rows = []
    for name in names:
        ps = policy_segments.get(name, {})
        rs = realized_segments.get(name, {})
        cost = _segment_cost(config_snapshot, name)
        expected_residual = ps.get("expected_residual_loss")
        realized_residual = rs.get("residual_loss") if backtest_dict is not None else None
        ratio = (
            (realized_residual / expected_residual)
            if (backtest_dict is not None and expected_residual)
            else None
        )
        rows.append(
            {
                "name": name,
                "false_accept_cost": cost.get("false_accept_cost") if cost else None,
                "review_cost": cost.get("review_cost") if cost else None,
                "threshold": ps.get("p_error_threshold"),
                "n_review_planned": ps.get("n_review", 0),
                "n_review_replayed": rs.get("n_review"),
                "n_total": ps.get("n_total", rs.get("n_total", 0)),
                "expected_residual": expected_residual if expected_residual is not None else 0.0,
                "realized_residual": realized_residual,
                "ratio": ratio,
            }
        )

    totals = policy_dict.get("totals", {}) or {}
    total_realized = (
        backtest_dict["strategies"]["policy"]["residual_loss"] if backtest_dict else None
    )
    total_expected = totals.get("expected_residual_loss")
    total_ratio = (total_realized / total_expected) if (backtest_dict and total_expected) else None
    return {
        "rows": rows,
        "totals": {
            "n_review_planned": totals.get("n_review"),
            "n_review_replayed": (
                backtest_dict["strategies"]["policy"]["n_review"] if backtest_dict else None
            ),
            "n_total": totals.get("n_total"),
            "expected_residual": total_expected,
            "realized_residual": total_realized,
            "ratio": total_ratio,
        },
    }


def render(
    calibrator_dict: dict[str, Any],
    policy_dict: dict[str, Any],
    backtest_dict: dict[str, Any] | None,
    *,
    created_at: str,
    tool_version: str,
) -> str:
    """Render the self-contained HTML report. See docs/interfaces.md §11."""
    calibrator_hash = _verify_chain(calibrator_dict, policy_dict, backtest_dict)
    policy_hash = str(policy_dict.get("policy_hash", ""))
    backtest_hash = str(backtest_dict.get("backtest_hash", "")) if backtest_dict else None
    policy_input_shas = policy_dict.get("input_sha256s", {})

    hash_rows: list[tuple[str, str]] = [
        ("calibrator content_hash", calibrator_hash),
        ("policy policy_hash", policy_hash),
        ("policy.input_sha256s.batch", str(policy_input_shas.get("batch", ""))),
        ("policy.input_sha256s.calibrator", str(policy_input_shas.get("calibrator", ""))),
        ("policy.input_sha256s.costs", str(policy_input_shas.get("costs", ""))),
    ]
    if backtest_dict is not None:
        backtest_input_shas = backtest_dict.get("input_sha256s", {})
        hash_rows.append(("backtest backtest_hash", str(backtest_hash)))
        hash_rows.append(
            ("backtest.input_sha256s.calibrator", str(backtest_input_shas.get("calibrator", "")))
        )
        hash_rows.append(
            ("backtest.input_sha256s.history", str(backtest_input_shas.get("history", "")))
        )
        hash_rows.append(
            ("backtest.input_sha256s.policy", str(backtest_input_shas.get("policy", "")))
        )
    short_chain = [("Calibrator", calibrator_hash), ("Policy", policy_hash)]
    if backtest_hash is not None:
        short_chain.append(("Backtest", backtest_hash))

    summary = _build_summary(calibrator_dict, policy_dict, backtest_dict)

    totals = policy_dict.get("totals", {}) or {}
    n_total = int(totals.get("n_total", 0) or 0)
    n_review_planned = int(totals.get("n_review", 0) or 0)
    budget_fraction = (n_review_planned / n_total) if n_total else 0.0

    # Figure order here is cosmetic (a dict); DOCUMENT order -- and therefore
    # where the plotly.js bundle sits relative to each figure -- is decided
    # by the template, which places `plotly_bundle` right after
    # #at-a-glance and before the first `{{ figs[...] }}` (§11.4).
    figs: dict[str, str] = {
        "fig-reliability": _reliability_figure(calibrator_dict, div_id="fig-reliability")
    }

    results_view: dict[str, Any] | None = None
    if backtest_dict is not None:
        figs["fig-backtest"] = _capture_figure(backtest_dict, div_id="fig-backtest")
        figs["fig-coverage"] = _coverage_figure(
            backtest_dict, budget_fraction=budget_fraction, div_id="fig-coverage"
        )
        figs["fig-segments"] = _segments_figure(policy_dict, backtest_dict, div_id="fig-segments")
        policy_strategy = backtest_dict["strategies"]["policy"]
        results_view = {
            "strategies": backtest_dict["strategies"],
            "n_review": policy_strategy["n_review"],
            "total_true_loss": backtest_dict["total_true_loss"],
            "n_errors": backtest_dict["n_errors"],
            "n_history": backtest_dict["n_history"],
            "tie_inflated": policy_strategy["n_review"] > n_review_planned,
        }

    calibration_view = {
        "method": calibrator_dict["method"],
        "ece_holdout": calibrator_dict["ece_holdout"],
        "ece_ci": calibrator_dict["ece_ci"],
        "ci_label": _ci_level_label(calibrator_dict),
        "ece_raw_holdout": calibrator_dict["ece_raw_holdout"],
        "n_golden": calibrator_dict["n_golden"],
        "seed": calibrator_dict["seed"],
    }

    segments_view = _build_segments_view(policy_dict, backtest_dict)
    gates_view = _build_gates_view(calibrator_dict, policy_dict, backtest_dict)

    env = Environment(
        loader=PackageLoader("triage"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        keep_trailing_newline=True,
    )
    for name, fn in (
        ("fmt_count", fmt_count),
        ("fmt_cost", fmt_cost),
        ("fmt_pct", fmt_pct),
        ("fmt_ratio", fmt_ratio),
        ("fmt_lift", fmt_lift),
        ("fmt_ece", fmt_ece),
    ):
        env.filters[name] = fn
    template = env.get_template(_TEMPLATE_NAME)
    rendered: str = template.render(
        tool_version=tool_version,
        created_at=created_at,
        hash_rows=hash_rows,
        short_chain=short_chain,
        summary=summary,
        calibration=calibration_view,
        segments=segments_view,
        gates=gates_view,
        results=results_view,
        figs=figs,
        plotly_bundle=plotly.offline.get_plotlyjs(),
        budget_fraction=budget_fraction,
        attention_ratio=ATTENTION_RESIDUAL_RATIO,
        severity_icon={"fail": "\u2715", "warn": "!", "info": "i", "pass": "\u2713"},
        status_label={
            ("ok", "full"): "On track",
            ("ok", "plan_only"): "Plan only \u2014 not backtested",
            ("attention", "full"): "Needs attention",
            ("attention", "plan_only"): "Needs attention",
            ("failed", "full"): "Gate failed",
            ("failed", "plan_only"): "Gate failed",
        }[(summary.status, summary.mode)],
        # §11.6: an "ok" plan-only report is unvalidated, not "healthy" --
        # it gets the neutral colour, never green.
        status_color=(
            "neutral" if (summary.status, summary.mode) == ("ok", "plan_only") else summary.status
        ),
    )
    return rendered

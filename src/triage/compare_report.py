"""HTML comparison page: "should we switch from the current policy to the
candidate?" See docs/interfaces.md §14.8.

Same visual system and the same rule as the single-policy report (§11): the
answer first, the evidence second, the audit trail last. Shared pieces
(palette, number filters, figure layout, Jinja environment, base template,
macros) come from `render_utils` and `templates/_base.html.j2` /
`_macros.html.j2`, so the two pages cannot drift apart.

`render()` is a PURE function of its arguments (byte-identical HTML for
identical inputs; `created_at` is the only injected variance, on exactly one
line, as in §11.1).

Verification boundary (§14.8): this page receives comparison.json ONLY. It
recomputes the comparison's own `comparison_hash` and checks the artifact's
internal accounting (deltas, tie-outs, overlap counts, recorded hashes). The
policy/calibrator relationships were verified by `triage compare`, which
refuses to write a comparison for a broken chain; the page shows the hashes
it recorded and says "comparison verified", not "artifact chain verified".
"""

from dataclasses import dataclass
from typing import Any, Literal

import plotly.graph_objects as go
import plotly.offline

from triage.artifacts import content_hash
from triage.exceptions import ContractViolation
from triage.render_utils import (
    COLOR_CANDIDATE,
    COLOR_CURRENT,
    GRID_COLOR,
    base_layout,
    budget_label,
    environment,
    fmt_cost,
    fmt_count,
    fmt_pct,
    fmt_signed_cost,
    to_html,
)

__all__ = ["render"]

_TEMPLATE_NAME = "compare.html.j2"
_SIDES = ("current", "candidate")
_DELTA_KEYS = ("captured_loss", "residual_loss", "n_review", "review_spend", "total_cost")
_OUTCOME_KEYS = (*_DELTA_KEYS, "capture_rate", "efficiency", "segments")
_UNREVIEWED_COLOR = "#e3e6ea"

Severity = Literal["fail", "warn", "info", "pass"]
_SEVERITY_ORDER: dict[Severity, int] = {"fail": 0, "warn": 1, "info": 2, "pass": 3}
_SEVERITY_ICON = {"fail": "\u2715", "warn": "!", "info": "i", "pass": "\u2713"}

# Catalogue order: tie-break within a severity group (deterministic order).
_CHECK_ORDER = [
    "verdict.total_cost",
    "basis.planned_under_other_costs.current",
    "basis.planned_under_other_costs.candidate",
    "segments.not_in_plan.current",
    "segments.not_in_plan.candidate",
    "uncertainty.inconclusive",
    "uncertainty.not_computed",
    "basis.eval_costs",
    "basis.shared",
    "segments.not_in_history.current",
    "segments.not_in_history.candidate",
]

_STATUS_LABEL = {
    "improvement": "Improvement",
    "regression": "Regression",
    "trade_off": "Trade-off",
    "no_change": "No change",
    "inconclusive": "Inconclusive",
}

_DIM_NAMES = {"captured_loss": "loss caught", "n_review": "reviews", "review_spend": "spend"}

_PHRASES = {
    ("captured_loss", "better"): "catches more loss",
    ("captured_loss", "worse"): "catches less loss",
    ("n_review", "better"): "needs fewer reviews",
    ("n_review", "worse"): "needs more reviews",
    ("review_spend", "better"): "spends less on review",
    ("review_spend", "worse"): "spends more on review",
}


# ---------------------------------------------------------------------------
# Verification (§14.8)
# ---------------------------------------------------------------------------


def fmt_signed_pts(value: float | None) -> str:
    """A change in a share, in percentage points: "+14.8 pts", "\u22123.0 pts",
    "0 pts" (same sign conventions as render_utils.fmt_signed_cost)."""
    if value is None:
        return "\u2014"
    text = f"{abs(value) * 100:.1f}"
    if text == "0.0":
        return "0 pts"
    return ("+" if value > 0 else "\u2212") + text + " pts"


def _fail(message: str) -> ContractViolation:
    return ContractViolation(f"Comparison report: {message}")


def _verify(comparison: dict[str, Any]) -> None:
    """Self-hash + internal accounting. Raises ContractViolation; never
    KeyError (a malformed artifact is a contract violation, exit 1)."""
    required = (
        "basis",
        "comparison_hash",
        "delta",
        "history",
        "input_sha256s",
        "overlap",
        "policies",
        "tool_version",
        "uncertainty",
        "verdict",
    )
    missing = [key for key in required if key not in comparison]
    if missing:
        raise _fail(f"comparison.json is missing required key(s) {missing}")

    declared = comparison["comparison_hash"]
    computed = content_hash(comparison)
    if computed != declared:
        raise _fail(
            f"comparison.comparison_hash={declared!r} but content_hash(comparison)={computed!r} "
            "(the comparison was edited after `triage compare` wrote it)"
        )

    try:
        policies = comparison["policies"]
        outcomes = {side: policies[side]["outcome"] for side in _SIDES}
        for side in _SIDES:
            absent = [key for key in _OUTCOME_KEYS if key not in outcomes[side]]
            if absent:
                raise _fail(f"policies.{side}.outcome is missing {absent}")
        total = comparison["history"]["total_true_loss"]
        n_history = comparison["history"]["n_history"]
        delta = comparison["delta"]
        overlap = comparison["overlap"]
        shas = comparison["input_sha256s"]

        for key in _DELTA_KEYS:
            expected = outcomes["candidate"][key] - outcomes["current"][key]
            if delta[key] != expected:
                raise _fail(f"delta.{key}={delta[key]!r} but candidate - current = {expected!r}")
        for side in _SIDES:
            out = outcomes[side]
            if out["residual_loss"] != total - out["captured_loss"]:
                raise _fail(
                    f"policies.{side}: residual_loss does not tie out with "
                    "history.total_true_loss - captured_loss"
                )
            if shas[f"{side}_policy"] != policies[side]["policy_hash"]:
                raise _fail(f"input_sha256s.{side}_policy differs from policies.{side}.policy_hash")
            if shas[f"{side}_calibrator"] != policies[side]["calibrator_hash"]:
                raise _fail(
                    f"input_sha256s.{side}_calibrator differs from policies.{side}.calibrator_hash"
                )
        n_both = overlap["both"]["n_review"]
        if n_both + overlap["current_only"]["n_review"] != outcomes["current"]["n_review"] or (
            n_both + overlap["candidate_only"]["n_review"] != outcomes["candidate"]["n_review"]
        ):
            raise _fail("overlap counts do not add up to each policy's n_review")
        n_groups = (
            n_both
            + overlap["current_only"]["n_review"]
            + overlap["candidate_only"]["n_review"]
            + overlap["neither"]["n_items"]
        )
        if n_groups != n_history:
            raise _fail(f"overlap groups cover {n_groups} items, history has {n_history}")
    except (KeyError, TypeError) as e:
        raise _fail(f"malformed comparison.json ({type(e).__name__}: {e})") from e


# ---------------------------------------------------------------------------
# View model -- pure, plain text only (escaped by the template)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    id: str
    severity: Severity
    title: str
    detail: str
    value: str
    section: str


def _cost_pair(costs: dict[str, Any] | None) -> str:
    if costs is None:
        return "unpriced"
    return f"{fmt_cost(costs['false_accept_cost'])} / {fmt_cost(costs['review_cost'])}"


def _seg_list(names: list[str]) -> str:
    """Plain-English list: "a", "a and b", "a, b and c"."""
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _displayed_verdict(comparison: dict[str, Any]) -> tuple[str, str, dict[str, str]]:
    """(label, total_cost_direction, directions): the CI verdict when the
    bootstrap ran, else the point verdict."""
    verdict = comparison["verdict"]
    source = verdict["ci"] if verdict.get("ci") is not None else verdict
    return source["label"], source["total_cost"], dict(source["directions"])


def _identical_review_sets(comparison: dict[str, Any]) -> bool:
    overlap = comparison["overlap"]
    return bool(
        overlap["current_only"]["n_review"] == 0 and overlap["candidate_only"]["n_review"] == 0
    )


def _status(comparison: dict[str, Any]) -> tuple[str, str]:
    """(data-status, badge text). With a CI, `failed` is exactly the
    regression gate's condition (`compare.check_regression_gate`): a CI
    regression, or total cost significantly worse. Without a CI, only a
    point regression is `failed`."""
    label, total_dir, _ = _displayed_verdict(comparison)
    has_ci = comparison["verdict"].get("ci") is not None
    if label == "regression":
        return "failed", _STATUS_LABEL[label]
    if has_ci and total_dir == "worse":
        return "failed", "Worse overall"
    if label == "trade_off":
        return "attention", _STATUS_LABEL[label]
    if label == "improvement":
        return "ok", _STATUS_LABEL[label]
    return "neutral", _STATUS_LABEL[label]


def _headline(comparison: dict[str, Any], status: str) -> str:
    label, _, directions = _displayed_verdict(comparison)
    if label == "no_change":
        if _identical_review_sets(comparison):
            return "The candidate reviews exactly the same items as the current policy"
        return "The candidate performs the same as the current policy"
    if label == "inconclusive":
        return "No difference this history can confirm"
    if label == "regression":
        return "The candidate is worse than the current policy"
    if status == "failed":
        return "The candidate costs more overall than the current policy"
    if label == "improvement":
        return "The candidate improves on the current policy"
    good = [_PHRASES[(k, v)] for k, v in directions.items() if v == "better"]
    bad = [_PHRASES[(k, v)] for k, v in directions.items() if v == "worse"]
    return f"The candidate {' and '.join(good)}, but {' and '.join(bad)}"


def _more_less(value: float, more: str, less: str) -> str:
    return more if value > 0 else less


def _lede(comparison: dict[str, Any]) -> str:
    cur = comparison["policies"]["current"]["outcome"]
    cand = comparison["policies"]["candidate"]["outcome"]
    d = comparison["delta"]
    n_history = comparison["history"]["n_history"]
    parts = [f"On {fmt_count(n_history)} labeled items, switching to the candidate would"]
    clauses = []
    if d["captured_loss"]:
        clauses.append(
            f"catch {fmt_cost(abs(d['captured_loss']))} "
            f"{_more_less(d['captured_loss'], 'more', 'less')} cost units of loss "
            f"({fmt_cost(cur['captured_loss'])} \u2192 {fmt_cost(cand['captured_loss'])})"
        )
    else:
        clauses.append(f"catch the same loss ({fmt_cost(cur['captured_loss'])} cost units)")
    if d["n_review"]:
        share = d["n_review"] / cur["n_review"] if cur["n_review"] else None
        share_text = (
            f", {'+' if d['n_review'] > 0 else chr(0x2212)}{fmt_pct(abs(share))}"
            if (share is not None)
            else ""
        )
        clauses.append(
            f"review {fmt_count(abs(d['n_review']))} "
            f"{_more_less(d['n_review'], 'more', 'fewer')} items "
            f"({fmt_count(cur['n_review'])} \u2192 {fmt_count(cand['n_review'])}{share_text})"
        )
    else:
        clauses.append(f"review the same number of items ({fmt_count(cur['n_review'])})")
    if d["review_spend"]:
        clauses.append(
            f"spend {fmt_cost(abs(d['review_spend']))} "
            f"{_more_less(d['review_spend'], 'more', 'less')} cost units on review"
        )
    else:
        clauses.append("spend the same on review")
    sentence = " ".join(parts) + " " + ", ".join(clauses[:-1]) + " and " + clauses[-1] + "."
    if d["total_cost"]:
        sentence += (
            f" Total cost (unreviewed loss + review spend) "
            f"{_more_less(d['total_cost'], 'rises', 'falls')} by "
            f"{fmt_cost(abs(d['total_cost']))} cost units."
        )
    else:
        sentence += " Total cost (unreviewed loss + review spend) is unchanged."
    return sentence


def _ci_sentence(comparison: dict[str, Any]) -> str | None:
    unc = comparison["uncertainty"]
    ci = comparison["verdict"].get("ci")
    if unc is None or ci is None:
        return None
    significant = [_DIM_NAMES[k] for k, v in ci["directions"].items() if v in ("better", "worse")]
    inconclusive = [_DIM_NAMES[k] for k, v in ci["directions"].items() if v == "inconclusive"]
    level = fmt_pct(unc["ci_level"], 0)
    if not significant and not inconclusive:
        return None
    text = f"At {level} confidence (paired bootstrap, {fmt_count(unc['n_bootstrap'])} resamples)"
    if significant and inconclusive:
        return (
            f"{text}, the change in {_seg_list(significant)} is real; the change in "
            f"{_seg_list(inconclusive)} could be noise."
        )
    if significant:
        return f"{text}, every change is larger than the noise."
    return f"{text}, none of these changes is distinguishable from noise."


def _checks(comparison: dict[str, Any]) -> tuple[Check, ...]:
    checks: list[Check] = []
    basis = comparison["basis"]
    policies = comparison["policies"]
    d = comparison["delta"]
    _, total_dir, _ = _displayed_verdict(comparison)
    has_ci = comparison["verdict"].get("ci") is not None

    if d["total_cost"] or total_dir != "same":
        worse = d["total_cost"] > 0
        severity: Severity
        if total_dir == "better":
            severity = "pass"
        elif total_dir == "worse":
            severity = "fail" if has_ci else "warn"
        else:
            severity = "info"
        checks.append(
            Check(
                id="verdict.total_cost",
                severity=severity,
                title=(
                    f"Total cost {'rises' if worse else 'falls'} by "
                    f"{fmt_cost(abs(d['total_cost']))} cost units"
                    + (" (within noise)" if total_dir == "inconclusive" else "")
                ),
                detail=(
                    "Total cost = loss left unreviewed + review spend, both priced by the "
                    "evaluation basis. It is the quantity the optimizer minimizes, so it says "
                    "whether the candidate's trade-off pays off under these costs."
                ),
                value=fmt_signed_cost(d["total_cost"]),
                section="results",
            )
        )

    if basis["source"] == "eval_costs":
        checks.append(
            Check(
                id="basis.eval_costs",
                severity="info",
                title="Both policies are priced by an explicit evaluation cost model",
                detail=(
                    "Losses and review spend come from the cost model supplied with "
                    "--eval-costs (see the audit trail for its sha256). Its budget and gates "
                    "are ignored: budgets belong to the policies being compared."
                ),
                value="--eval-costs",
                section="basis",
            )
        )
    else:
        checks.append(
            Check(
                id="basis.shared",
                severity="pass",
                title="Both policies price this history identically",
                detail=(
                    "Every history segment resolves to the same miss and review costs under both "
                    "policies' own cost models, so they are compared in the same units."
                ),
                value="shared costs",
                section="basis",
            )
        )

    for side in _SIDES:
        diffs = basis["differences"][side]
        if diffs:
            names = sorted(diffs)
            details = "; ".join(
                f"{name}: planned with {_cost_pair(diffs[name])}, priced here at "
                f"{_cost_pair(basis['segments'][name])}"
                for name in names
            )
            checks.append(
                Check(
                    id=f"basis.planned_under_other_costs.{side}",
                    severity="warn",
                    title=f"The {side} policy was planned under different costs",
                    detail=(
                        f"{details} (miss / review, cost units). Its thresholds were optimized "
                        "for other costs than the ones it is judged by here."
                    ),
                    value=f"{len(names)} segment{'s' if len(names) != 1 else ''}",
                    section="basis",
                )
            )

        seg_out = policies[side]["outcome"]["segments"]
        not_in_plan = sorted(n for n, s in seg_out.items() if s.get("status") == "not_in_plan")
        if not_in_plan:
            n_items = sum(seg_out[n]["n_total"] for n in not_in_plan)
            checks.append(
                Check(
                    id=f"segments.not_in_plan.{side}",
                    severity="warn",
                    title=f"The {side} policy has no plan for {_seg_list(not_in_plan)}",
                    detail=(
                        f"{_seg_list(not_in_plan)} ({fmt_count(n_items)} history items) did not "
                        f"appear in the batch the {side} policy was planned on, so it reviews "
                        "nothing there."
                    ),
                    value=f"{fmt_count(n_items)} items",
                    section="segments",
                )
            )
        absent = policies[side]["planned_segments_not_in_history"]
        if absent:
            checks.append(
                Check(
                    id=f"segments.not_in_history.{side}",
                    severity="info",
                    title=f"Part of the {side} policy is not exercised by this history",
                    detail=(
                        f"The {side} policy has thresholds for {_seg_list(absent)}, but the "
                        "history has no items there, so this comparison says nothing about them."
                    ),
                    value=f"{len(absent)} segment{'s' if len(absent) != 1 else ''}",
                    section="segments",
                )
            )

    unc = comparison["uncertainty"]
    if unc is None:
        checks.append(
            Check(
                id="uncertainty.not_computed",
                severity="info",
                title="No confidence interval was computed",
                detail=(
                    "The comparison ran with --n-bootstrap 0, so the verdict is a point estimate "
                    "on one history and small differences may be noise."
                ),
                value="point only",
                section="uncertainty",
            )
        )
    elif has_ci:
        inconclusive = [
            _DIM_NAMES[k]
            for k, v in comparison["verdict"]["ci"]["directions"].items()
            if v == "inconclusive"
        ]
        if inconclusive:
            checks.append(
                Check(
                    id="uncertainty.inconclusive",
                    severity="info",
                    title=f"Change in {_seg_list(inconclusive)} could be noise",
                    detail=(
                        f"The {fmt_pct(unc['ci_level'], 0)} confidence interval of these deltas "
                        "includes zero, so this history cannot tell them apart from chance."
                    ),
                    value="inconclusive",
                    section="uncertainty",
                )
            )

    order = {cid: i for i, cid in enumerate(_CHECK_ORDER)}
    return tuple(sorted(checks, key=lambda c: (_SEVERITY_ORDER[c.severity], order[c.id])))


def _direction_label(direction: str | None) -> str:
    return {
        "better": "better",
        "worse": "worse",
        "same": "same",
        "inconclusive": "inconclusive",
    }.get(direction or "", "\u2014")


def _kpis(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    cur = comparison["policies"]["current"]["outcome"]
    cand = comparison["policies"]["candidate"]["outcome"]
    d = comparison["delta"]
    unc = comparison["uncertainty"]
    _, _, dirs = _displayed_verdict(comparison)
    # residual_loss = total - captured_loss and LOWER is better, so whenever
    # more loss is caught, less is left: its direction IS captured_loss's.
    residual_dir = dirs["captured_loss"]
    rows = [
        ("captured", "Loss caught", "captured_loss", "cost", dirs["captured_loss"]),
        ("residual", "Risk left unreviewed", "residual_loss", "cost", residual_dir),
        ("reviews", "Items reviewed", "n_review", "count", dirs["n_review"]),
        ("spend", "Review spend", "review_spend", "cost", dirs["review_spend"]),
    ]
    tiles = []
    for kpi_id, label, key, kind, direction in rows:
        ci = unc["delta_ci"][key] if unc is not None else None
        tiles.append(
            {
                "id": kpi_id,
                "label": label,
                "kind": kind,
                "delta": d[key],
                "current": cur[key],
                "candidate": cand[key],
                "ci": ci,
                "direction": direction,
            }
        )
    return tiles


def _results_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    cur = comparison["policies"]["current"]["outcome"]
    cand = comparison["policies"]["candidate"]["outcome"]
    d = comparison["delta"]
    unc = comparison["uncertainty"]

    def ci(key: str) -> list[float] | None:
        return unc["delta_ci"][key] if unc is not None else None

    def rate_delta(key: str) -> float | None:
        if cur[key] is None or cand[key] is None:
            return None
        value: float = cand[key] - cur[key]
        return value

    return [
        {"label": "loss caught", "kind": "cost", "key": "captured_loss",
         "current": cur["captured_loss"], "candidate": cand["captured_loss"],
         "delta": d["captured_loss"], "ci": ci("captured_loss")},
        {"label": "loss left unreviewed", "kind": "cost", "key": "residual_loss",
         "current": cur["residual_loss"], "candidate": cand["residual_loss"],
         "delta": d["residual_loss"], "ci": ci("residual_loss")},
        {"label": "share of loss caught", "kind": "pct", "key": "capture_rate",
         "current": cur["capture_rate"], "candidate": cand["capture_rate"],
         "delta": rate_delta("capture_rate"), "ci": None},
        {"label": "items reviewed", "kind": "count", "key": "n_review",
         "current": cur["n_review"], "candidate": cand["n_review"],
         "delta": d["n_review"], "ci": ci("n_review")},
        {"label": "review spend", "kind": "cost", "key": "review_spend",
         "current": cur["review_spend"], "candidate": cand["review_spend"],
         "delta": d["review_spend"], "ci": ci("review_spend")},
        {"label": "loss caught per review", "kind": "cost", "key": "efficiency",
         "current": cur["efficiency"], "candidate": cand["efficiency"],
         "delta": rate_delta("efficiency"), "ci": None},
        {"label": "total cost (unreviewed + spend)", "kind": "cost", "key": "total_cost",
         "current": cur["total_cost"], "candidate": cand["total_cost"],
         "delta": d["total_cost"], "ci": ci("total_cost"), "emphasis": True},
    ]  # fmt: skip


def _segment_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per history segment, largest |delta captured| first, then name."""
    basis_segments = comparison["basis"]["segments"]
    cur = comparison["policies"]["current"]["outcome"]["segments"]
    cand = comparison["policies"]["candidate"]["outcome"]["segments"]
    d = comparison["delta"]["segments"]
    names = sorted(d, key=lambda n: (-abs(float(d[n]["captured_loss"])), n))
    rows = []
    for name in names:
        rows.append(
            {
                "name": name,
                "costs": basis_segments.get(name),
                "n_total": cur[name]["n_total"],
                "at_stake": cur[name]["captured_loss"] + cur[name]["residual_loss"],
                "sides": {
                    side: {
                        "status": seg[name].get("status"),
                        "threshold": seg[name].get("p_error_threshold"),
                        "n_review": seg[name]["n_review"],
                        "captured_loss": seg[name]["captured_loss"],
                    }
                    for side, seg in (("current", cur), ("candidate", cand))
                },
                "delta": d[name],
            }
        )
    return rows


def _basis_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    basis = comparison["basis"]
    return [
        {
            "name": name,
            "basis": basis["segments"][name],
            "current": basis["differences"]["current"].get(name, "same"),
            "candidate": basis["differences"]["candidate"].get(name, "same"),
        }
        for name in sorted(basis["segments"])
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _totals_figure(comparison: dict[str, Any], *, div_id: str) -> str:
    """Per policy, one bar split into loss caught and loss left unreviewed
    (together: the history's total loss under the basis)."""
    total = comparison["history"]["total_true_loss"]
    outcomes = [comparison["policies"][side]["outcome"] for side in _SIDES]
    labels = ["Current policy", "Candidate policy"]
    colors = [COLOR_CURRENT, COLOR_CANDIDATE]
    caught = [float(o["captured_loss"]) for o in outcomes]
    left = [float(o["residual_loss"]) for o in outcomes]
    texts = [
        f"{fmt_pct(o['capture_rate'])} caught \u00b7 {fmt_count(o['n_review'])} reviews"
        for o in outcomes
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=caught,
            y=labels,
            orientation="h",
            name="loss caught",
            marker={"color": colors},
            text=texts,
            textposition="inside",
            insidetextanchor="start",
            textfont={"color": "#ffffff"},
            hovertemplate="%{y}: %{x:,.0f} caught<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=left,
            y=labels,
            orientation="h",
            name="left unreviewed",
            marker={"color": _UNREVIEWED_COLOR},
            hovertemplate="%{y}: %{x:,.0f} left unreviewed<extra></extra>",
        )
    )
    fig.update_layout(
        **base_layout(
            barmode="stack",
            # Keep the in-bar label readable on a short bar instead of shrinking it.
            uniformtext={"minsize": 11, "mode": "show"},
            xaxis={
                "title": {"text": f"cost units (total loss on this history: {fmt_cost(total)})"},
                "showgrid": True,
                "gridcolor": GRID_COLOR,
            },
            yaxis={"automargin": True, "autorange": "reversed"},
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            margin={"l": 120, "r": 20, "t": 30, "b": 45},
            height=200,
        )
    )
    return to_html(fig, div_id=div_id)


def _segments_figure(rows: list[dict[str, Any]], *, div_id: str) -> str:
    """Per segment: loss caught by each policy, same order as the table."""
    names = [r["name"] for r in rows]
    fig = go.Figure()
    for side, label, color in (
        ("current", "current", COLOR_CURRENT),
        ("candidate", "candidate", COLOR_CANDIDATE),
    ):
        fig.add_trace(
            go.Bar(
                x=[float(r["sides"][side]["captured_loss"]) for r in rows],
                y=names,
                orientation="h",
                name=label,
                marker={"color": color},
                hovertemplate=f"%{{y}} ({label}): %{{x:,.0f}} caught<extra></extra>",
            )
        )
    fig.update_layout(
        **base_layout(
            barmode="group",
            xaxis={
                "title": {"text": "loss caught (cost units)"},
                "showgrid": True,
                "gridcolor": GRID_COLOR,
            },
            yaxis={"automargin": True, "autorange": "reversed"},
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
            margin={"l": 90, "r": 20, "t": 30, "b": 45},
            height=80 + 56 * max(len(names), 1),
        )
    )
    return to_html(fig, div_id=div_id)


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(comparison_dict: dict[str, Any], *, created_at: str, tool_version: str) -> str:
    """Render the self-contained comparison page. See docs/interfaces.md §14.8."""
    _verify(comparison_dict)
    c = comparison_dict
    status, status_label = _status(c)
    label, _, _ = _displayed_verdict(c)
    segment_rows = _segment_rows(c)
    shas = c["input_sha256s"]

    side_style = {
        "current": ("Current policy", COLOR_CURRENT),
        "candidate": ("Candidate policy", COLOR_CANDIDATE),
    }
    sides_view = {
        side: {
            "label": side_style[side][0],
            "color": side_style[side][1],
            "n_review": c["policies"][side]["outcome"]["n_review"],
            "budget_label": budget_label({"budget": c["policies"][side]["budget"]}),
            "planned": c["policies"][side]["planned"],
            "calibrator_method": c["policies"][side]["calibrator_method"],
            "not_in_history": c["policies"][side]["planned_segments_not_in_history"],
        }
        for side in _SIDES
    }
    hash_rows: list[tuple[str, str]] = [("comparison comparison_hash", str(c["comparison_hash"]))]
    hash_rows += [(f"comparison.input_sha256s.{k}", str(shas[k])) for k in sorted(shas)]
    for side in _SIDES:
        recorded = c["policies"][side].get("policy_input_sha256s") or {}
        hash_rows += [
            (f"{side} policy.input_sha256s.{k}", str(recorded[k])) for k in sorted(recorded)
        ]
    short_chain = [
        ("Current policy", str(shas["current_policy"])),
        ("Candidate policy", str(shas["candidate_policy"])),
        ("Comparison", str(c["comparison_hash"])),
    ]

    figs = {
        "fig-compare": _totals_figure(c, div_id="fig-compare"),
        "fig-compare-segments": _segments_figure(segment_rows, div_id="fig-compare-segments"),
    }

    env = environment({"fmt_signed_pts": fmt_signed_pts})
    template = env.get_template(_TEMPLATE_NAME)
    rendered: str = template.render(
        tool_version=tool_version,
        created_at=created_at,
        status=status,
        status_label=status_label,
        verdict_label=label,
        headline=_headline(c, status),
        lede=_lede(c),
        ci_sentence=_ci_sentence(c),
        checks=_checks(c),
        severity_icon=_SEVERITY_ICON,
        kpis=_kpis(c),
        results=_results_rows(c),
        sides=sides_view,
        segments=segment_rows,
        overlap=c["overlap"],
        delta=c["delta"],
        history=c["history"],
        verdict=c["verdict"],
        uncertainty=c["uncertainty"],
        basis=c["basis"],
        basis_rows=_basis_rows(c),
        figs=figs,
        plotly_bundle=plotly.offline.get_plotlyjs(),
        short_chain=short_chain,
        hash_rows=hash_rows,
    )
    return rendered

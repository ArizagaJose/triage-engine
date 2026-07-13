"""HTML report: reliability diagram, policy summary, backtest comparison,
coverage-vs-risk curves, and the artifact hash chain.

See docs/interfaces.md §11 for the pinned contract. `render()` is a PURE
function of its arguments: identical inputs produce byte-identical HTML
(all variance is injected via the `created_at` argument, which the CLI sets
to `artifacts.utc_now_iso()`). Plotly figures use explicit `div_id`s so their
serialized HTML fragments are deterministic (no random client-side ids).

Chain verification happens here, not in the CLI: `render` never renders a
chain it has not itself checked (see `_verify_chain`).
"""

from typing import Any

import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape

from triage.artifacts import content_hash
from triage.exceptions import ContractViolation

__all__ = ["render"]

# Palette pinned to scripts/headline_chart.py's conventions.
_COLOR_POLICY = "#2a78d6"
_COLOR_RAW = "#1baf7a"
_COLOR_RANDOM = "#eda100"
_LABEL_INK = "#333333"
_GRID_COLOR = "#e6e6e3"
_BG_COLOR = "#fcfcfb"

_FONT = {"family": "Arial, Helvetica, sans-serif", "size": 12, "color": _LABEL_INK}

_TEMPLATE_NAME = "report.html.j2"


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

    if backtest_dict is not None:
        policy_hash = policy_dict.get("policy_hash")
        declared_policy_hash = backtest_dict.get("input_sha256s", {}).get("policy")
        if declared_policy_hash != policy_hash:
            raise ContractViolation(
                "Report chain broken: backtest.input_sha256s.policy="
                f"{declared_policy_hash!r} but policy.policy_hash={policy_hash!r}"
            )
        declared_bt_calibrator_hash = backtest_dict.get("input_sha256s", {}).get("calibrator")
        if declared_bt_calibrator_hash != calibrator_hash:
            raise ContractViolation(
                "Report chain broken: backtest.input_sha256s.calibrator="
                f"{declared_bt_calibrator_hash!r} but content_hash(calibrator)={calibrator_hash!r}"
            )

    return calibrator_hash


def _base_layout(**overrides: Any) -> dict[str, Any]:
    layout: dict[str, Any] = {
        "plot_bgcolor": _BG_COLOR,
        "paper_bgcolor": _BG_COLOR,
        "font": dict(_FONT),
        "margin": {"l": 60, "r": 40, "t": 40, "b": 50},
        "legend": {"orientation": "h", "yanchor": "bottom", "y": -0.3, "xanchor": "left", "x": 0.0},
    }
    layout.update(overrides)
    return layout


def _reliability_figure(
    calibrator_dict: dict[str, Any], *, div_id: str, include_plotlyjs: bool
) -> str:
    bins = calibrator_dict["reliability_bins"]
    before = [b for b in bins["before"] if b["count"] > 0]
    after = [b for b in bins["after"] if b["count"] > 0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="perfect calibration",
            line={"color": _LABEL_INK, "width": 1, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[b["mean_pred"] for b in before],
            y=[b["mean_observed"] for b in before],
            mode="lines+markers",
            name="before (raw score)",
            line={"color": _COLOR_RAW, "width": 2},
            marker={"color": _COLOR_RAW, "size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[b["mean_pred"] for b in after],
            y=[b["mean_observed"] for b in after],
            mode="lines+markers",
            name="after (calibrated)",
            line={"color": _COLOR_POLICY, "width": 2},
            marker={"color": _COLOR_POLICY, "size": 8},
        )
    )
    fig.update_layout(
        **_base_layout(
            title={"text": "Reliability diagram", "font": dict(_FONT)},
            xaxis={
                "title": {"text": "mean predicted p_error"},
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
            width=640,
            height=440,
        )
    )
    html: str = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id=div_id)
    return html


def _backtest_figure(backtest_dict: dict[str, Any], *, div_id: str, include_plotlyjs: bool) -> str:
    strategies = backtest_dict["strategies"]
    policy_captured = strategies["policy"]["captured_loss"]
    raw_captured = strategies["raw_confidence"]["captured_loss"]
    random_captured = strategies["random"]["captured_loss_mean"]
    random_std = strategies["random"]["captured_loss_std"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=["policy", "raw confidence", "random"],
            y=[policy_captured, raw_captured, random_captured],
            marker_color=[_COLOR_POLICY, _COLOR_RAW, _COLOR_RANDOM],
            error_y={"type": "data", "array": [0.0, 0.0, random_std], "visible": True},
            name="captured loss",
        )
    )
    fig.update_layout(
        **_base_layout(
            title={"text": "Captured loss by strategy (equal capacity)", "font": dict(_FONT)},
            xaxis={"title": {"text": "strategy"}},
            yaxis={
                "title": {"text": "captured loss"},
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
            },
            width=640,
            height=440,
            showlegend=False,
        )
    )
    html: str = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id=div_id)
    return html


def _coverage_figure(
    backtest_dict: dict[str, Any],
    *,
    budget_fraction: float,
    div_id: str,
    include_plotlyjs: bool,
) -> str:
    curves = backtest_dict["curves"]
    grid_pct = [g * 100.0 for g in curves["grid"]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=grid_pct,
            y=[v * 100.0 for v in curves["random"]],
            mode="lines",
            name="random",
            line={"color": _COLOR_RANDOM, "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid_pct,
            y=[v * 100.0 for v in curves["raw_confidence"]],
            mode="lines",
            name="raw confidence",
            line={"color": _COLOR_RAW, "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=grid_pct,
            y=[v * 100.0 for v in curves["policy"]],
            mode="lines",
            name="policy",
            line={"color": _COLOR_POLICY, "width": 2},
        )
    )
    fig.add_vline(x=budget_fraction * 100.0, line_width=1, line_color="#bbbbbb", line_dash="dot")
    fig.update_layout(
        **_base_layout(
            title={"text": "Coverage vs. risk captured", "font": dict(_FONT)},
            xaxis={
                "title": {"text": "review capacity (% of history)"},
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
            },
            yaxis={
                "title": {"text": "% of true loss captured"},
                "range": [0, 105],
                "showgrid": True,
                "gridcolor": _GRID_COLOR,
            },
            width=640,
            height=440,
        )
    )
    html: str = fig.to_html(full_html=False, include_plotlyjs=include_plotlyjs, div_id=div_id)
    return html


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
        hash_rows.append(("backtest backtest_hash", str(backtest_dict.get("backtest_hash", ""))))
        hash_rows.append(
            ("backtest.input_sha256s.calibrator", str(backtest_input_shas.get("calibrator", "")))
        )
        hash_rows.append(
            ("backtest.input_sha256s.history", str(backtest_input_shas.get("history", "")))
        )
        hash_rows.append(
            ("backtest.input_sha256s.policy", str(backtest_input_shas.get("policy", "")))
        )

    figs: dict[str, str] = {
        "fig-reliability": _reliability_figure(
            calibrator_dict, div_id="fig-reliability", include_plotlyjs=True
        )
    }

    totals = policy_dict.get("totals", {})
    n_total = int(totals.get("n_total", 0) or 0)
    n_review = int(totals.get("n_review", 0) or 0)
    budget_fraction = (n_review / n_total) if n_total else 0.0

    backtest_view: dict[str, Any] | None = None
    if backtest_dict is not None:
        figs["fig-backtest"] = _backtest_figure(
            backtest_dict, div_id="fig-backtest", include_plotlyjs=False
        )
        figs["fig-coverage"] = _coverage_figure(
            backtest_dict,
            budget_fraction=budget_fraction,
            div_id="fig-coverage",
            include_plotlyjs=False,
        )
        policy_strategy = backtest_dict["strategies"]["policy"]
        realized_residual = policy_strategy["residual_loss"]
        expected_residual = backtest_dict["expected"]["expected_residual_loss"]
        backtest_view = {
            "expected": backtest_dict["expected"],
            "realized_residual_loss": realized_residual,
            "residual_gap": realized_residual - expected_residual,
            "strategies": backtest_dict["strategies"],
            "n_review": policy_strategy["n_review"],
            "total_true_loss": backtest_dict["total_true_loss"],
            "n_errors": backtest_dict["n_errors"],
            "n_history": backtest_dict["n_history"],
        }

    calibration_view = {
        "method": calibrator_dict["method"],
        "ece_holdout": calibrator_dict["ece_holdout"],
        "ece_ci": calibrator_dict["ece_ci"],
        "ece_raw_holdout": calibrator_dict["ece_raw_holdout"],
        "n_golden": calibrator_dict["n_golden"],
        "seed": calibrator_dict["seed"],
    }

    policy_view = {
        "budget": policy_dict.get("budget", {}),
        "totals": totals,
        "segments": sorted(policy_dict.get("segments", {}).items(), key=lambda kv: kv[0]),
    }

    env = Environment(
        loader=PackageLoader("triage"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        keep_trailing_newline=True,
    )
    template = env.get_template(_TEMPLATE_NAME)
    rendered: str = template.render(
        tool_version=tool_version,
        created_at=created_at,
        hash_rows=hash_rows,
        calibration=calibration_view,
        policy=policy_view,
        backtest=backtest_view,
        figs=figs,
        budget_fraction=budget_fraction,
    )
    return rendered

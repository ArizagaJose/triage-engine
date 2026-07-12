#!/usr/bin/env python3
"""Generate the README headline chart: policy vs raw-confidence vs random baselines.

Reproducible from a single seed. Uses the triage Python API directly
(triage.synth / triage.calibration / triage.optimizer / triage.costs) — no CLI
dependency, since cli.py is owned by a parallel work package.

Usage:
    uv run python scripts/headline_chart.py --seed 42 -o docs/headline.png

Prints the 10%-budget numbers as a single machine-readable JSON line to
stdout, so the README's claims can be re-checked in CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

from triage import __version__
from triage.calibration import CalibrationConfig, Calibrator, fit
from triage.costs import Budget, CostModel, Gates, SegmentCosts
from triage.optimizer import plan

# Segment costs pinned by synth.py's overconfident scenario (see
# docs/interfaces.md §6): seg_high 500/4, seg_mid 50/3, seg_low 10/2, default
# mirrors seg_mid.
_SEGMENT_COSTS: dict[str, SegmentCosts] = {
    "seg_high": SegmentCosts(false_accept_cost=500.0, review_cost=4.0),
    "seg_mid": SegmentCosts(false_accept_cost=50.0, review_cost=3.0),
    "seg_low": SegmentCosts(false_accept_cost=10.0, review_cost=2.0),
    "default": SegmentCosts(false_accept_cost=50.0, review_cost=3.0),
}

_BUDGETS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)

_COLOR_POLICY = "#2a78d6"
_COLOR_RAW = "#1baf7a"
_COLOR_RANDOM = "#eda100"
_LABEL_INK = "#333333"
_GRID_COLOR = "#e6e6e3"
_BG_COLOR = "#fcfcfb"

_N_RANDOM_DRAWS = 200


def _cost_model(budget_value: float) -> CostModel:
    """Build the CostModel in code, matching synth's pinned segment costs."""
    return CostModel(
        budget=Budget(type="fraction", value=budget_value),
        gates=Gates(max_ece=0.05),
        segments=dict(_SEGMENT_COSTS),
    )


def _rank_by_net_benefit(
    batch: pl.DataFrame, calibrator: Calibrator, cost_model: CostModel
) -> dict[str, Any]:
    """Replicate the optimizer's ranking and compute per-item TRUE loss.

    "True loss" here is the false_accept_cost incurred if an item is an
    actual error (predicted_value != true_value) and is NOT reviewed. This is
    the ground-truth quantity the optimizer's calibrated `expected_loss` is
    trying to approximate; computing it directly from `true_value` is what
    lets the headline chart claim a verifiable number rather than a modeled
    one.

    Returns the item order under the policy's ranking (net_benefit desc,
    item_id asc tiebreak), the order under raw-confidence ranking (score
    asc, item_id asc tiebreak), the per-item true loss array, and its total.
    """
    item_ids = batch["item_id"].to_list()
    segments = batch["segment"].to_list()
    scores = batch["score"].to_numpy()
    predicted = batch["predicted_value"].to_list()
    true = batch["true_value"].to_list()
    n = len(item_ids)

    p_error = np.asarray(calibrator.predict(scores), dtype=np.float64)

    fa_cost = np.empty(n, dtype=np.float64)
    review_cost = np.empty(n, dtype=np.float64)
    for i, seg in enumerate(segments):
        c = cost_model.costs_for(seg)
        fa_cost[i] = c.false_accept_cost
        review_cost[i] = c.review_cost

    expected_loss = p_error * fa_cost
    net_benefit = expected_loss - review_cost

    is_error = np.array([predicted[i] != true[i] for i in range(n)], dtype=bool)
    true_loss = is_error.astype(np.float64) * fa_cost

    # Policy ranking: net_benefit desc, item_id asc tiebreak (two stable sorts).
    policy_order = sorted(range(n), key=lambda i: item_ids[i])
    policy_order.sort(key=lambda i: -net_benefit[i])

    # Raw-confidence ranking: least confident first (score asc), item_id asc tiebreak.
    raw_order = sorted(range(n), key=lambda i: item_ids[i])
    raw_order.sort(key=lambda i: scores[i])

    return {
        "policy_order": policy_order,
        "raw_order": raw_order,
        "true_loss": true_loss,
        "total_true_loss": float(true_loss.sum()),
    }


def _captured_pct(true_loss: np.ndarray, indices: list[int], total: float) -> float:
    captured = float(true_loss[indices].sum())
    return 100.0 * captured / total if total > 0 else 0.0


def compute_curve(seed: int) -> dict[str, Any]:
    """Run the full pipeline once and compute policy/raw/random curves across budgets.

    Deterministic in `seed`: no unseeded randomness anywhere.
    """
    from triage.synth import generate

    data = generate("overconfident", seed=seed)
    golden = data["golden"]
    batch = data["batch"]
    assert isinstance(golden, pl.DataFrame)
    assert isinstance(batch, pl.DataFrame)

    calibrator = fit(golden, seed=seed, config=CalibrationConfig())

    results: list[dict[str, Any]] = []
    for budget_index, b in enumerate(_BUDGETS):
        cost_model = _cost_model(b)

        policy = plan(
            batch,
            calibrator,
            cost_model,
            input_sha256s={"batch": "", "calibrator": "", "costs": ""},
            tool_version=__version__,
        )
        n_review = int(policy.totals["n_review"])

        ranking = _rank_by_net_benefit(batch, calibrator, cost_model)
        true_loss = ranking["true_loss"]
        total_true_loss = ranking["total_true_loss"]

        policy_pct = _captured_pct(true_loss, ranking["policy_order"][:n_review], total_true_loss)
        raw_pct = _captured_pct(true_loss, ranking["raw_order"][:n_review], total_true_loss)

        n_batch = len(true_loss)
        rng = np.random.default_rng(seed + 1000 + budget_index)
        random_draws = []
        for _ in range(_N_RANDOM_DRAWS):
            idx = rng.choice(n_batch, size=n_review, replace=False)
            random_draws.append(_captured_pct(true_loss, idx.tolist(), total_true_loss))
        random_pct = float(np.mean(random_draws)) if random_draws else 0.0

        results.append(
            {
                "budget": b,
                "n_review": n_review,
                "policy_pct": policy_pct,
                "raw_confidence_pct": raw_pct,
                "random_pct": random_pct,
            }
        )

    return {"seed": seed, "budgets": results}


def _find_at_budget(results: list[dict[str, Any]], budget: float) -> dict[str, Any]:
    for r in results:
        if abs(r["budget"] - budget) < 1e-9:
            return r
    raise ValueError(f"No result computed at budget {budget}")


def render_chart(curve: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """Render the 3-series line chart and write it to `out_path` as PNG."""
    results = curve["budgets"]
    x = [r["budget"] * 100.0 for r in results]
    y_policy = [r["policy_pct"] for r in results]
    y_raw = [r["raw_confidence_pct"] for r in results]
    y_random = [r["random_pct"] for r in results]

    headline = _find_at_budget(results, 0.10)
    title_text = (
        f"At a 10% review budget: policy captures {headline['policy_pct']:.0f}% "
        f"vs {headline['raw_confidence_pct']:.0f}% (raw confidence) and "
        f"{headline['random_pct']:.0f}% (random)"
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_random,
            mode="lines+markers",
            name="random",
            line={"color": _COLOR_RANDOM, "width": 2},
            marker={"size": 8, "color": _COLOR_RANDOM},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_raw,
            mode="lines+markers",
            name="raw confidence",
            line={"color": _COLOR_RAW, "width": 2},
            marker={"size": 8, "color": _COLOR_RAW},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_policy,
            mode="lines+markers",
            name="policy",
            line={"color": _COLOR_POLICY, "width": 2},
            marker={"size": 8, "color": _COLOR_POLICY},
        )
    )

    # Reference line at x = 10% budget.
    fig.add_vline(x=10, line_width=1, line_color="#bbbbbb", line_dash="dot")

    # Direct end-of-line labels: dark ink text + a short colored dash token,
    # never colored text (the series color lives on the token, not the label).
    label_specs = [
        ("policy", y_policy[-1], _COLOR_POLICY),
        ("raw confidence", y_raw[-1], _COLOR_RAW),
        ("random", y_random[-1], _COLOR_RANDOM),
    ]
    # Nudge overlapping labels apart vertically if endpoints are close.
    label_specs.sort(key=lambda t: t[1])
    min_gap = 4.0
    for i in range(1, len(label_specs)):
        name, y, color = label_specs[i]
        prev_name, prev_y, _ = label_specs[i - 1]
        if y - prev_y < min_gap:
            label_specs[i] = (name, prev_y + min_gap, color)

    for name, y, color in label_specs:
        fig.add_annotation(
            x=max(x),
            y=y,
            xref="x",
            yref="y",
            text=f"▬ {name}",
            showarrow=False,
            xanchor="left",
            xshift=10,
            font={"size": 12, "color": _LABEL_INK, "family": "Arial, Helvetica, sans-serif"},
        )
        # Color the token character itself via a separate small annotation
        # so the text stays dark ink while the marker/dash carries the color.
        fig.add_annotation(
            x=max(x),
            y=y,
            xref="x",
            yref="y",
            text="▬",
            showarrow=False,
            xanchor="left",
            xshift=10,
            font={"size": 12, "color": color, "family": "Arial, Helvetica, sans-serif"},
        )

    fig.update_layout(
        title={
            "text": title_text,
            "font": {"size": 16, "family": "Arial, Helvetica, sans-serif", "color": _LABEL_INK},
            "x": 0.02,
            "xanchor": "left",
        },
        plot_bgcolor=_BG_COLOR,
        paper_bgcolor=_BG_COLOR,
        font={"family": "Arial, Helvetica, sans-serif", "size": 12, "color": _LABEL_INK},
        width=900,
        height=520,
        margin={"l": 70, "r": 130, "t": 70, "b": 60},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "left",
            "x": 0.0,
        },
        xaxis={
            "title": {"text": "review budget (% of items)"},
            "showgrid": True,
            "gridcolor": _GRID_COLOR,
            "gridwidth": 1,
            "zeroline": False,
            "range": [0, max(x) * 1.18],
        },
        yaxis={
            "title": {"text": "% of expected loss captured"},
            "showgrid": True,
            "gridcolor": _GRID_COLOR,
            "gridwidth": 1,
            "zeroline": False,
            "range": [0, 105],
        },
        showlegend=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), width=900, height=520, scale=2)

    return {
        "seed": curve["seed"],
        "budget_pct": 10,
        "policy_pct": round(headline["policy_pct"], 2),
        "raw_confidence_pct": round(headline["raw_confidence_pct"], 2),
        "random_pct": round(headline["random_pct"], 2),
        "n_review": headline["n_review"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("docs/headline.png"),
        help="Output PNG path (default: docs/headline.png)",
    )
    args = parser.parse_args(argv)

    curve = compute_curve(args.seed)
    headline = render_chart(curve, args.output)

    print(json.dumps(headline, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())

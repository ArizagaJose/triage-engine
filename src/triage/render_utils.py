"""Rendering helpers shared by every HTML page triage produces.

The single-policy report (`report.py`, docs/interfaces.md §11) and the
policy comparison page (`compare_report.py`, §14) must look and read the
same: one palette, one number format, one figure layout, one Jinja
environment, one base template (`templates/_base.html.j2`). Everything here
is pure and locale-free, so pages stay byte-identical for identical inputs.

Nothing in this module knows about a particular artifact schema beyond the
two small display lookups at the bottom (`segment_cost`, `budget_label`),
which read `policy.json` fields that both pages show.
"""

from typing import Any

import plotly.graph_objects as go
from jinja2 import Environment, PackageLoader, select_autoescape

__all__ = [
    "BG_COLOR",
    "COLOR_CANDIDATE",
    "COLOR_CURRENT",
    "COLOR_POLICY",
    "COLOR_RANDOM",
    "COLOR_RAW",
    "FONT",
    "GRID_COLOR",
    "INK",
    "NEUTRAL",
    "PLOTLY_CONFIG",
    "STATUS_COLOR",
    "base_layout",
    "budget_label",
    "environment",
    "fmt_cost",
    "fmt_count",
    "fmt_ece",
    "fmt_lift",
    "fmt_pct",
    "fmt_ratio",
    "fmt_signed_cost",
    "fmt_signed_count",
    "segment_cost",
    "to_html",
]

# Palette (interfaces.md §11.11.1). Status colours (ok/attention/failed) are
# reserved for verdicts, so no data series uses green, amber or red.
COLOR_POLICY = "#2a78d6"
COLOR_RAW = "#7c5cc4"
COLOR_RANDOM = "#9aa1a9"
# Comparison page (§14): "before" is a lighter, desaturated blue and "after"
# is the policy blue, so the pair reads as the same kind of thing, not as
# two unrelated series (and neither is a verdict colour).
COLOR_CURRENT = "#8fa3bf"
COLOR_CANDIDATE = COLOR_POLICY
STATUS_COLOR = {"ok": "#15803d", "attention": "#d97706", "failed": "#dc2626"}
NEUTRAL = "#64748b"
INK = "#1f2933"
GRID_COLOR = "#e7eaee"
BG_COLOR = "#fcfcfb"

FONT = {
    "family": "-apple-system, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif",
    "size": 12,
    "color": INK,
}

PLOTLY_CONFIG = {"responsive": True, "displaylogo": False}


# ---------------------------------------------------------------------------
# Number formatting (§11.11.3) -- locale-free, pure; registered as Jinja
# filters AND used directly when building sentences in Python, so a number
# reads the same in a sentence as it does in a table cell.
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


def fmt_signed_cost(value: float | None) -> str:
    """Deltas: explicit sign, U+2212 minus, "0" for zero (never "+0"/"-0")."""
    if value is None:
        return "\u2014"
    text = fmt_cost(abs(value))
    if text == "0":
        return "0"
    return ("+" if value > 0 else "\u2212") + text


def fmt_signed_count(value: float | int | None) -> str:
    if value is None:
        return "\u2014"
    text = fmt_count(abs(value))
    if text == "0":
        return "0"
    return ("+" if value > 0 else "\u2212") + text


_FILTERS = {
    "fmt_count": fmt_count,
    "fmt_cost": fmt_cost,
    "fmt_pct": fmt_pct,
    "fmt_ratio": fmt_ratio,
    "fmt_lift": fmt_lift,
    "fmt_ece": fmt_ece,
    "fmt_signed_cost": fmt_signed_cost,
    "fmt_signed_count": fmt_signed_count,
}


def environment(extra_filters: dict[str, Any] | None = None) -> Environment:
    """The Jinja environment every page renders with (§11.11.4): package
    templates, autoescape on (artifact-derived text such as segment names is
    escaped at the template boundary), trailing newline kept, and the shared
    number filters. `extra_filters` adds page-specific filters; it may not
    redefine a shared one (two pages formatting the same number differently
    is exactly what this module exists to prevent)."""
    env = Environment(
        loader=PackageLoader("triage"),
        autoescape=select_autoescape(default=True, default_for_string=True),
        keep_trailing_newline=True,
    )
    for name, fn in _FILTERS.items():
        env.filters[name] = fn
    for name, fn in (extra_filters or {}).items():
        if name in _FILTERS:
            raise ValueError(f"filter {name!r} is shared; it cannot be redefined per page")
        env.filters[name] = fn
    return env


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def base_layout(**overrides: Any) -> dict[str, Any]:
    """No fixed width (§11.11): figures fill their container; only height is
    pinned. `to_html`'s `config={"responsive": True}` makes plotly resize the
    div on the client without changing the serialized figure spec, so output
    stays byte-identical for identical inputs."""
    layout: dict[str, Any] = {
        "autosize": True,
        "plot_bgcolor": BG_COLOR,
        "paper_bgcolor": BG_COLOR,
        "font": dict(FONT),
        "margin": {"l": 60, "r": 110, "t": 40, "b": 50},
        "showlegend": False,
    }
    layout.update(overrides)
    return layout


def to_html(fig: go.Figure, *, div_id: str) -> str:
    """Serialize a figure as an HTML fragment WITHOUT plotly.js: every page
    embeds the bundle exactly once, as a standalone script (§11.4), and an
    explicit `div_id` keeps the fragment deterministic."""
    html: str = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id=div_id,
        config=PLOTLY_CONFIG,
    )
    return html


# ---------------------------------------------------------------------------
# Display lookups on policy.json
# ---------------------------------------------------------------------------


def segment_cost(config_snapshot: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Mirrors `CostModel.costs_for` (§3) for display: exact key, else
    'default', else None (rendered as "—", never a crash -- if an artifact
    reached a renderer its costs already resolved successfully upstream;
    this is a display-only best-effort lookup)."""
    segments = config_snapshot.get("segments", {}) or {}
    found: dict[str, Any] | None = segments.get(name) or segments.get("default")
    return found


# Display names for `policy.allocation.selected_candidate` (opt-in "fill"
# money-budget strategy, docs/interfaces.md §5).
_CANDIDATE_LABELS = {
    "net_benefit": "net-benefit ranking",
    "net_benefit_per_cost": "net-benefit-per-cost ranking",
}


def budget_label(policy_dict: dict[str, Any]) -> str:
    budget = policy_dict.get("budget", {}) or {}
    kind, value = budget.get("type"), budget.get("value")
    if kind == "fraction" and value is not None:
        return f"budget {fmt_pct(value, 0)}"
    if kind == "count" and value is not None:
        return f"budget {fmt_count(value)} items"
    if kind == "money" and value is not None:
        label = f"budget {fmt_cost(value)} cost units"
        # `budget.strategy` is present in policy.json only when non-default,
        # so default-strategy reports keep their label unchanged.
        strategy = budget.get("strategy")
        if strategy is not None:
            allocation = policy_dict.get("allocation") or {}
            candidate = allocation.get("selected_candidate")
            if candidate is not None:
                ranking = _CANDIDATE_LABELS.get(candidate, str(candidate))
                label += f" ({strategy} strategy, {ranking})"
            else:
                label += f" ({strategy} strategy)"
        return label
    return "budget not set"

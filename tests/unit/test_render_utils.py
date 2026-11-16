"""Unit tests for triage.render_utils: the helpers shared by the single-policy
report and the comparison page (docs/interfaces.md §11.11, §14.8)."""

from __future__ import annotations

from typing import Any

import pytest

from triage import render_utils, report
from triage.render_utils import (
    budget_label,
    environment,
    fmt_signed_cost,
    fmt_signed_count,
    segment_cost,
)

MINUS = "\u2212"


class TestSignedFilters:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (42500.4, "+42,500"),
            (-39904.0, f"{MINUS}39,904"),
            (0.0, "0"),
            (-0.0, "0"),
            (0.4, "0"),  # rounds to zero: never "+0"
            (-0.4, "0"),
            (0.6, "+1"),
            (None, "\u2014"),
        ],
    )
    def test_signed_cost(self, value: float | None, expected: str) -> None:
        assert fmt_signed_cost(value) == expected

    @pytest.mark.parametrize(
        "value,expected", [(649, "+649"), (-3, f"{MINUS}3"), (0, "0"), (None, "\u2014")]
    )
    def test_signed_count(self, value: int | None, expected: str) -> None:
        assert fmt_signed_count(value) == expected


class TestEnvironment:
    def test_shared_filters_are_registered(self) -> None:
        env = environment()
        for name in (
            "fmt_count",
            "fmt_cost",
            "fmt_pct",
            "fmt_ratio",
            "fmt_lift",
            "fmt_ece",
            "fmt_signed_cost",
            "fmt_signed_count",
        ):
            assert env.filters[name] is getattr(render_utils, name)

    def test_page_filters_are_added(self) -> None:
        env = environment({"shout": str.upper})
        assert env.from_string("{{ 'x'|shout }}").render() == "X"

    def test_shared_filters_cannot_be_redefined(self) -> None:
        with pytest.raises(ValueError, match="fmt_cost"):
            environment({"fmt_cost": str})

    def test_autoescape_is_on(self) -> None:
        assert environment().from_string("{{ v }}").render(v="<b>") == "&lt;b&gt;"


class TestDisplayLookups:
    def test_segment_cost_exact_then_default_then_none(self) -> None:
        snapshot: dict[str, Any] = {
            "segments": {"a": {"false_accept_cost": 1.0}, "default": {"false_accept_cost": 2.0}}
        }
        assert segment_cost(snapshot, "a") == {"false_accept_cost": 1.0}
        assert segment_cost(snapshot, "zzz") == {"false_accept_cost": 2.0}
        assert segment_cost({"segments": {}}, "a") is None
        assert segment_cost({}, "a") is None

    @pytest.mark.parametrize(
        "budget,label",
        [
            ({"type": "fraction", "value": 0.1}, "budget 10%"),
            ({"type": "count", "value": 1000}, "budget 1,000 items"),
            ({"type": "money", "value": 4000.0}, "budget 4,000 cost units"),
            ({}, "budget not set"),
        ],
    )
    def test_budget_label(self, budget: dict[str, Any], label: str) -> None:
        assert budget_label({"budget": budget}) == label


def test_report_reexports_the_shared_formatters() -> None:
    """The fmt_* filters moved here; `from triage.report import fmt_cost`
    must keep working and mean the same function."""
    for name in ("fmt_count", "fmt_cost", "fmt_pct", "fmt_ratio", "fmt_lift", "fmt_ece"):
        assert getattr(report, name) is getattr(render_utils, name)


def test_base_layout_overrides_and_isolation() -> None:
    a = render_utils.base_layout(height=100)
    b = render_utils.base_layout()
    assert a["height"] == 100 and "height" not in b
    a["font"]["size"] = 99  # a copy: never mutates the shared font
    assert render_utils.FONT["size"] == 12

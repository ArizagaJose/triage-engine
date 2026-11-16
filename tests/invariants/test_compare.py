"""Policy-comparison invariants: docs/test-spec-invariants.md I. Each pair is
two runs in two processes (the shared chain in-process, and a fresh
console-script process), like A/D/G.5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.report_html import table_shapes
from tests.support import (
    Chain,
    html_without_created_at,
    json_text_without_created_at,
    load_json,
    ok,
)
from triage.artifacts import content_hash


@pytest.fixture(scope="module")
def base(chain: Any) -> Chain:
    return chain("overconfident", 42)


@pytest.fixture(scope="module")
def session_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("compare_invariants")


@pytest.fixture(scope="module")
def candidate_policy(base: Chain, cli: Any, session_tmp: Path) -> Path:
    """The same chain planned at a 15% budget."""
    out = session_tmp / "b15" / "policy.json"
    ok(
        cli(
            "plan",
            batch=base.batch,
            calibrator=base.calibrator,
            costs=base.costs,
            budget=0.15,
            o=out,
        )
    )
    return out


def compare_args(base: Chain, candidate_policy: Path, out: Path) -> dict[str, Any]:
    return {
        "current_policy": base.policy,
        "current_calibrator": base.calibrator,
        "candidate_policy": candidate_policy,
        "candidate_calibrator": base.calibrator,
        "history": base.batch,
        "n_bootstrap": 500,
        "o": out,
    }


@pytest.fixture(scope="module")
def comparison_pair(
    base: Chain, candidate_policy: Path, cli: Any, cli_process: Any, session_tmp: Path
) -> tuple[Path, Path]:
    dir_a, dir_b = session_tmp / "cmp_a", session_tmp / "cmp_b"
    ok(cli("compare", **compare_args(base, candidate_policy, dir_a)))
    ok(cli_process("compare", **compare_args(base, candidate_policy, dir_b)))
    return dir_a, dir_b


def test_comparison_identical_across_processes(comparison_pair: tuple[Path, Path]) -> None:
    """I.1: comparison.json (bootstrap included) is identical across processes
    minus `created_at`, and its comparison_hash is its content_hash."""
    dir_a, dir_b = comparison_pair
    text_a = (dir_a / "comparison.json").read_text()
    text_b = (dir_b / "comparison.json").read_text()
    assert json_text_without_created_at(text_a) == json_text_without_created_at(text_b)
    parsed = load_json(dir_a / "comparison.json")
    assert parsed["comparison_hash"] == content_hash(parsed)


def test_comparison_report_identical_across_processes(
    comparison_pair: tuple[Path, Path], cli: Any, cli_process: Any, session_tmp: Path
) -> None:
    """I.2: `report --compare` renders byte-identical HTML minus the created-at line."""
    dir_a, _ = comparison_pair
    html_a, html_b = session_tmp / "a.html", session_tmp / "b.html"
    ok(cli("report", compare=dir_a, o=html_a))
    ok(cli_process("report", compare=dir_a, o=html_b))
    assert html_without_created_at(html_a.read_text()) == html_without_created_at(
        html_b.read_text()
    )


def test_comparison_identities_on_real_chain(comparison_pair: tuple[Path, Path]) -> None:
    """I.3: same history, same basis -> one total loss; deltas are candidate -
    current; the review-queue overlap explains the captured-loss delta."""
    c = load_json(comparison_pair[0] / "comparison.json")
    total = c["history"]["total_true_loss"]
    cur, cand = (c["policies"][s]["outcome"] for s in ("current", "candidate"))
    for out in (cur, cand):
        assert out["residual_loss"] == total - out["captured_loss"]
    for key in ("captured_loss", "residual_loss", "n_review", "review_spend", "total_cost"):
        assert c["delta"][key] == cand[key] - cur[key]
    o = c["overlap"]
    assert abs(
        c["delta"]["captured_loss"]
        - (o["candidate_only"]["captured_loss"] - o["current_only"]["captured_loss"])
    ) <= 1e-9 * max(1.0, total)
    lo, hi = c["uncertainty"]["delta_ci"]["captured_loss"]
    assert lo <= c["delta"]["captured_loss"] <= hi


def test_comparison_report_tables_are_rectangular(
    comparison_pair: tuple[Path, Path], cli: Any, session_tmp: Path
) -> None:
    """I.4: every table row spans the same number of columns as its header."""
    out = session_tmp / "shape.html"
    ok(cli("report", compare=comparison_pair[0], o=out))
    for widths in table_shapes(out.read_text()):
        assert len(set(widths)) == 1, widths

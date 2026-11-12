"""Report invariant tests, driven through the real CLI.

Implements docs/test-spec-invariants.md, section G items 5-9 (Stage 6).
Imitates the tests/test_determinism_backtest.py / tests/test_claims_backtest.py
idiom: subprocess calls to the real `triage` console script, module-scoped
fixtures so the (expensive) CLI chain runs once and is shared across
assertions, and byte/dict comparison after stripping the `created_at` /
`id="created-at"` line.

Chain (spec G preamble, reusing the D/F chain idiom): `synth overconfident
--seed 42` -> `calibrate --seed 42` -> `plan` (emitted costs.yaml, budget
from YAML) -> `backtest --seed 42 --n-random-draws 200` with `--history` =
the scenario's own batch.parquet. Artifacts are written directly as
`calibrator.json` / `policy.json` / `backtest.json` inside a chain directory
so that directory can be passed straight to `triage report --from`.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml

from tests import report_html as rh
from triage.artifacts import content_hash
from triage.exceptions import ContractViolation
from triage.report import ATTENTION_RESIDUAL_RATIO, _build_summary, render

TRIAGE_BIN = shutil.which("triage") or str(Path(__file__).resolve().parents[2] / ".venv/bin/triage")


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the real `triage` console script and capture output."""
    return subprocess.run(
        [TRIAGE_BIN, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _strip_created_at_line(text: str) -> str:
    """Drop the single line containing id="created-at" from an HTML blob,
    asserting there is exactly one such line."""
    lines = text.splitlines()
    matching = [line for line in lines if 'id="created-at"' in line]
    assert len(matching) == 1, f'expected exactly one id="created-at" line, found {len(matching)}'
    return "\n".join(line for line in lines if 'id="created-at"' not in line)


def _run_chain(
    session_dir: Path,
    name: str,
    scenario: str,
    *,
    seed: int = 42,
    max_residual_risk: float | None = None,
    backtest_returncode: int = 0,
) -> Path:
    """Run the full synth -> calibrate -> plan -> backtest chain for
    `scenario`, writing calibrator.json / policy.json / backtest.json
    directly into a chain directory (so it doubles as a `--from` dir).
    Returns the chain directory.

    `max_residual_risk` declares that gate in the emitted costs.yaml before
    `plan` (the CI workflow's exit-code demo, interfaces.md §12); a chain
    that trips it passes `backtest_returncode=2` -- the artifact is still
    written first, so the report can render it (§5.9)."""
    chain_dir = session_dir / name
    data_dir = chain_dir / "data"

    synth_result = run_cli(
        ["synth", "--scenario", scenario, "--seed", str(seed), "-o", str(data_dir)]
    )
    assert synth_result.returncode == 0, synth_result.stderr

    calibrator_path = chain_dir / "calibrator.json"
    calibrate_result = run_cli(
        [
            "calibrate",
            "--golden",
            str(data_dir / "golden.parquet"),
            "--seed",
            str(seed),
            "-o",
            str(calibrator_path),
        ]
    )
    assert calibrate_result.returncode == 0, calibrate_result.stderr

    costs_path = data_dir / "costs.yaml"
    if max_residual_risk is not None:
        costs = yaml.safe_load(costs_path.read_text())
        costs.setdefault("gates", {})["max_residual_risk"] = max_residual_risk
        costs_path.write_text(yaml.safe_dump(costs, sort_keys=True))

    policy_path = chain_dir / "policy.json"
    plan_result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(costs_path),
            "-o",
            str(policy_path),
        ]
    )
    assert plan_result.returncode == 0, plan_result.stderr

    backtest_path = chain_dir / "backtest.json"
    backtest_result = run_cli(
        [
            "backtest",
            "--policy",
            str(policy_path),
            "--calibrator",
            str(calibrator_path),
            "--history",
            str(data_dir / "batch.parquet"),
            "--seed",
            str(seed),
            "--n-random-draws",
            "200",
            "-o",
            str(backtest_path),
        ]
    )
    assert backtest_result.returncode == backtest_returncode, backtest_result.stderr

    return chain_dir


@pytest.fixture(scope="module")
def report_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("report_session")


@pytest.fixture(scope="module")
def full_chain_dir(report_session: Path) -> Path:
    """Full overconfident seed-42 chain (calibrator + policy + backtest)."""
    return _run_chain(report_session, "overconfident_full", "overconfident")


@pytest.fixture(scope="module")
def full_chain_dicts(full_chain_dir: Path) -> dict[str, Any]:
    return {
        "calibrator": json.loads((full_chain_dir / "calibrator.json").read_text()),
        "policy": json.loads((full_chain_dir / "policy.json").read_text()),
        "backtest": json.loads((full_chain_dir / "backtest.json").read_text()),
    }


# --- G.5: report determinism ---


@pytest.fixture(scope="module")
def report_pair(report_session: Path, full_chain_dir: Path) -> tuple[Path, Path]:
    """`report` run twice from the same --from dir into two output paths."""
    out_a = report_session / "report_a.html"
    out_b = report_session / "report_b.html"
    result_a = run_cli(["report", "--from", str(full_chain_dir), "-o", str(out_a)])
    assert result_a.returncode == 0, result_a.stderr
    result_b = run_cli(["report", "--from", str(full_chain_dir), "-o", str(out_b)])
    assert result_b.returncode == 0, result_b.stderr
    return out_a, out_b


def test_report_identical_minus_created_at_line(report_pair: tuple[Path, Path]) -> None:
    """Spec G.5: the two HTML files are identical after dropping the single
    line containing id="created-at" (assert exactly one such line each)."""
    out_a, out_b = report_pair
    stripped_a = _strip_created_at_line(out_a.read_text())
    stripped_b = _strip_created_at_line(out_b.read_text())
    assert stripped_a == stripped_b


# --- G.6: hash chain rendered ---


def test_report_contains_hash_chain(
    report_pair: tuple[Path, Path], full_chain_dicts: dict[str, Any]
) -> None:
    """Spec G.6: the HTML contains the full policy_hash, the calibrator
    content_hash, and the backtest_hash as substrings."""
    out_a, _ = report_pair
    html = out_a.read_text()

    calibrator_hash = content_hash(full_chain_dicts["calibrator"])
    policy_hash = full_chain_dicts["policy"]["policy_hash"]
    backtest_hash = full_chain_dicts["backtest"]["backtest_hash"]

    assert calibrator_hash in html
    assert policy_hash in html
    assert backtest_hash in html


def test_report_contains_pinned_div_ids(report_pair: tuple[Path, Path]) -> None:
    """Spec G.6 (+ H preamble): the four pinned div ids (fig-reliability,
    fig-backtest, fig-coverage, fig-segments) each appear."""
    out_a, _ = report_pair
    html = out_a.read_text()
    assert "fig-reliability" in html
    assert "fig-backtest" in html
    assert "fig-coverage" in html
    assert "fig-segments" in html


def test_report_plotly_embedded_no_network(report_pair: tuple[Path, Path]) -> None:
    """Spec G.6: plotly embedded (no cdn.plot.ly / https:// plotly script
    src anywhere in the file) -- local-first at view time.

    Checks for an actual externally-loaded <script src="..."> tag, not a
    blanket substring search: the embedded plotly.js bundle itself contains
    the literal text "https://cdn.plot.ly/un/" as an inert default config
    value (mapboxAccessToken's placeholder), which is not a network
    reference and must not fail this check.
    """
    out_a, _ = report_pair
    html = out_a.read_text()
    script_srcs = re.findall(r'<script[^>]*\bsrc\s*=\s*["\']([^"\']*)["\']', html, re.IGNORECASE)
    assert script_srcs == [], f"found externally-loaded <script src=...> tag(s): {script_srcs}"


# --- G.7: tamper checks ---


def test_report_tampered_calibrator_exits_1(
    report_session: Path, full_chain_dir: Path, full_chain_dicts: dict[str, Any]
) -> None:
    """Spec G.7: a calibrator.json recalibrated at seed 43 in the --from dir
    -> exit 1, stderr names a hash."""
    tamper_dir = report_session / "tamper_calibrator"
    tamper_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(full_chain_dir / "policy.json", tamper_dir / "policy.json")
    shutil.copy(full_chain_dir / "backtest.json", tamper_dir / "backtest.json")

    data_dir = full_chain_dir / "data"
    tampered_calibrator = tamper_dir / "calibrator.json"
    calibrate_result = run_cli(
        [
            "calibrate",
            "--golden",
            str(data_dir / "golden.parquet"),
            "--seed",
            "43",
            "-o",
            str(tampered_calibrator),
        ]
    )
    assert calibrate_result.returncode == 0, calibrate_result.stderr

    out_path = report_session / "report_tampered.html"
    result = run_cli(["report", "--from", str(tamper_dir), "-o", str(out_path)])
    assert result.returncode == 1, result.stderr
    assert "hash" in result.stderr.lower()


def test_report_missing_policy_exits_1(report_session: Path, full_chain_dir: Path) -> None:
    """Spec G.7: a --from dir missing policy.json -> exit 1."""
    missing_dir = report_session / "missing_policy"
    missing_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(full_chain_dir / "calibrator.json", missing_dir / "calibrator.json")

    out_path = report_session / "report_missing_policy.html"
    result = run_cli(["report", "--from", str(missing_dir), "-o", str(out_path)])
    assert result.returncode == 1, result.stderr


def test_report_tampered_policy_content_raises(full_chain_dicts: dict[str, Any]) -> None:
    """A policy dict whose semantic content (totals) was mutated after hashing,
    with policy_hash left untouched, must be rejected by render() -- the
    declared policy_hash no longer matches content_hash(policy)."""
    calibrator = full_chain_dicts["calibrator"]
    tampered_policy = json.loads(json.dumps(full_chain_dicts["policy"]))
    totals = tampered_policy["totals"]
    key = next(iter(totals))
    totals[key] = (totals[key] or 0) + 1.0
    backtest = full_chain_dicts["backtest"]

    with pytest.raises(ContractViolation, match="policy_hash"):
        render(
            calibrator,
            tampered_policy,
            backtest,
            created_at="2020-01-01T00:00:00Z",
            tool_version="test",
        )


def test_report_tampered_backtest_content_raises(full_chain_dicts: dict[str, Any]) -> None:
    """A backtest dict whose semantic content (total_true_loss) was mutated
    after hashing, with backtest_hash left untouched, must be rejected by
    render() -- the declared backtest_hash no longer matches
    content_hash(backtest)."""
    calibrator = full_chain_dicts["calibrator"]
    policy = full_chain_dicts["policy"]
    tampered_backtest = json.loads(json.dumps(full_chain_dicts["backtest"]))
    tampered_backtest["total_true_loss"] = (tampered_backtest["total_true_loss"] or 0) + 1.0

    with pytest.raises(ContractViolation, match="backtest_hash"):
        render(
            calibrator,
            policy,
            tampered_backtest,
            created_at="2020-01-01T00:00:00Z",
            tool_version="test",
        )


def test_report_valid_triple_still_renders(full_chain_dicts: dict[str, Any]) -> None:
    """A fully valid, untampered triple must still render successfully."""
    html = render(
        full_chain_dicts["calibrator"],
        full_chain_dicts["policy"],
        full_chain_dicts["backtest"],
        created_at="2020-01-01T00:00:00Z",
        tool_version="test",
    )
    assert full_chain_dicts["policy"]["policy_hash"] in html


# --- G.8: backtest-optional ---


def test_report_without_backtest_exits_0_with_note(
    report_session: Path, full_chain_dir: Path
) -> None:
    """Spec G.8: --from dir WITHOUT backtest.json -> exit 0, HTML contains
    the "backtest not run" note and does NOT contain fig-coverage."""
    no_backtest_dir = report_session / "no_backtest"
    no_backtest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(full_chain_dir / "calibrator.json", no_backtest_dir / "calibrator.json")
    shutil.copy(full_chain_dir / "policy.json", no_backtest_dir / "policy.json")

    out_path = report_session / "report_no_backtest.html"
    result = run_cli(["report", "--from", str(no_backtest_dir), "-o", str(out_path)])
    assert result.returncode == 0, result.stderr

    html = out_path.read_text()
    assert "backtest not run" in html.lower()
    assert "fig-coverage" not in html


def test_report_with_backtest_note_absent(report_pair: tuple[Path, Path]) -> None:
    """Spec G.8: with backtest.json present the "backtest not run" note is
    absent."""
    out_a, _ = report_pair
    html = out_a.read_text()
    assert "backtest not run" not in html.lower()


# --- G.9: report is not an exit-code gate ---


# --- Gates panel (docs/interfaces.md §11.9) ---


def test_report_contains_gates_section_with_computed_verdict(
    report_pair: tuple[Path, Path], full_chain_dicts: dict[str, Any]
) -> None:
    """The real CLI-driven report renders a Gates section, and the max_ece
    verdict matches the strict `>` comparison against the actual chain
    values (not hardcoded)."""
    out_a, _ = report_pair
    html = out_a.read_text()
    assert "<h2>Gates</h2>" in html

    gates = full_chain_dicts["policy"]["config_snapshot"]["gates"]
    max_ece = gates.get("max_ece")
    ece_holdout = full_chain_dicts["calibrator"]["ece_holdout"]

    gates_section = rh.section(html, "gates")
    assert "max_ece" in gates_section
    if max_ece is None:
        assert "not set" in gates_section
    else:
        expected_verdict = "FAIL" if ece_holdout > max_ece else "PASS"
        assert f">{expected_verdict}<" in gates_section


@pytest.fixture(scope="module")
def drift_chain_dir(report_session: Path) -> Path:
    """Full drift seed-42 chain, no residual gate declared."""
    return _run_chain(report_session, "drift_full", "drift")


@pytest.fixture(scope="module")
def drift_gated_chain_dir(report_session: Path) -> Path:
    """Drift chain with `max_residual_risk: 200000` declared -- the CI
    workflow's exit-code demo (interfaces.md §12): plan passes (expected
    ~142,706), backtest exits 2 (realized ~336,100) but still writes its
    artifact."""
    return _run_chain(
        report_session,
        "drift_gated_full",
        "drift",
        max_residual_risk=200000.0,
        backtest_returncode=2,
    )


def _report_html(report_session: Path, from_dir: Path, name: str) -> str:
    out_path = report_session / name
    result = run_cli(["report", "--from", str(from_dir), "-o", str(out_path)])
    assert result.returncode == 0, result.stderr
    return out_path.read_text()


def test_report_drift_chain_exits_0(report_session: Path, drift_chain_dir: Path) -> None:
    """Spec G.9: report is not an exit-code gate: the drift chain (realized
    residual > expected) still reports exit 0 -- gating belongs to
    plan/backtest."""
    out_path = report_session / "report_drift.html"
    result = run_cli(["report", "--from", str(drift_chain_dir), "-o", str(out_path)])
    assert result.returncode == 0, result.stderr


# --- H.4: the redesign's status, end to end through the real CLI ---


def test_report_overconfident_chain_is_on_track(report_pair: tuple[Path, Path]) -> None:
    """H.4.1."""
    html = report_pair[0].read_text()
    attrs = rh.main_attrs(html)
    assert (attrs["data-status"], attrs["data-mode"]) == ("ok", "full")
    assert {c["severity"] for c in rh.rendered_checks(html)} <= {"info", "pass"}


def test_report_drift_chain_needs_attention_and_names_the_segment(
    report_session: Path, drift_chain_dir: Path
) -> None:
    """H.4.2: the ratio and the worst segment are recomputed HERE from the
    same artifacts, not hardcoded, then looked for in the rendered check."""
    html = _report_html(report_session, drift_chain_dir, "report_drift_h4.html")
    assert rh.main_attrs(html)["data-status"] == "attention"

    policy = json.loads((drift_chain_dir / "policy.json").read_text())
    backtest = json.loads((drift_chain_dir / "backtest.json").read_text())
    realized_segments = backtest["strategies"]["policy"]["segments"]
    ratio = (
        backtest["strategies"]["policy"]["residual_loss"]
        / backtest["expected"]["expected_residual_loss"]
    )
    assert ratio > ATTENTION_RESIDUAL_RATIO

    def gap(name: str) -> float:
        realized = realized_segments.get(name, {}).get("residual_loss", 0.0)
        expected = policy["segments"].get(name, {}).get("expected_residual_loss", 0.0)
        return float(realized - expected)

    names = sorted(set(policy["segments"]) | set(realized_segments))
    worst = min(names, key=lambda name: (-gap(name), name))
    assert worst == "seg_high"

    check = {c["id"]: c for c in rh.rendered_checks(html)}["plan.vs_actual"]
    assert check["severity"] == "warn" and check["open"]
    block_text = rh.text(rh.check_block(html, "plan.vs_actual"))
    assert f"{ratio:.2f}\u00d7" in block_text
    assert worst in block_text


def test_report_ci_gate_chain_fails_but_report_exits_0(
    report_session: Path, drift_gated_chain_dir: Path
) -> None:
    """H.4.3: backtest exits 2 (asserted while building the chain), yet the
    report renders with exit 0 and a red status -- gating belongs to
    plan/backtest (G.9), the report only tells the truth about it."""
    html = _report_html(report_session, drift_gated_chain_dir, "report_drift_gated.html")
    assert rh.main_attrs(html)["data-status"] == "failed"
    checks = {c["id"]: c for c in rh.rendered_checks(html)}
    assert checks["gate.max_residual_risk.realized"]["severity"] == "fail"
    assert checks["gate.max_residual_risk.realized"]["open"]
    assert checks["gate.max_residual_risk.planned"]["severity"] == "pass"
    assert rh.rendered_checks(html)[0]["id"] == "gate.max_residual_risk.realized"


def test_report_drift_plan_only_is_neutral_not_green(
    report_session: Path, drift_chain_dir: Path
) -> None:
    """H.4.4: the R2-shaped case. The SAME drift plan that is "attention"
    once replayed must not look healthy just because it wasn't checked
    against reality."""
    plan_only_dir = report_session / "drift_plan_only"
    plan_only_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(drift_chain_dir / "calibrator.json", plan_only_dir / "calibrator.json")
    shutil.copy(drift_chain_dir / "policy.json", plan_only_dir / "policy.json")

    html = _report_html(report_session, plan_only_dir, "report_drift_plan_only.html")
    attrs = rh.main_attrs(html)
    assert (attrs["data-status"], attrs["data-mode"]) == ("ok", "plan_only")
    glance = rh.section(html, "at-a-glance")
    assert "status-neutral" in glance
    assert "status-ok" not in glance
    assert "Plan only \u2014 not backtested" in rh.text(glance)


# --- H.4.5: golden summaries ---

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"
GOLDEN_CHAINS = {
    "overconfident": "full_chain_dir",
    "drift": "drift_chain_dir",
    "drift_gated": "drift_gated_chain_dir",
}


def summary_projection(chain_dir: Path) -> dict[str, Any]:
    """What the golden files pin: status, mode, (id, severity) per check in
    order, and the KPI numbers -- NOT raw HTML and NOT check wording, so a
    copy change never touches the goldens and a logic change always does."""
    calibrator, policy, backtest = (
        json.loads((chain_dir / f"{name}.json").read_text())
        for name in ("calibrator", "policy", "backtest")
    )
    summary = _build_summary(calibrator, policy, backtest)
    projection = {
        "status": summary.status,
        "mode": summary.mode,
        "checks": [[c.id, c.severity] for c in summary.checks],
        "kpis": summary.kpis,
    }
    return json.loads(json.dumps(projection, sort_keys=True))


def assert_summary_matches_golden(actual: Any, expected: Any, path: str = "summary") -> None:
    """Compare structure exactly, allowing eight ULPs for report KPI floats.

    Platt fitting uses BLAS: Haswell and Sandybridge kernels reproduce the
    runner's last-bit differences even with identical seeds and dependencies.
    This display-summary check must not impose cross-CPU bitwise equivalence;
    the separate artifact determinism tests still compare exact content.
    """
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key, value in expected.items():
            assert_summary_matches_golden(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for index, value in enumerate(expected):
            assert_summary_matches_golden(actual[index], value, f"{path}[{index}]")
    elif isinstance(expected, float):
        assert math.isfinite(actual) and math.isfinite(expected), path
        assert abs(actual - expected) <= 8 * math.ulp(expected), path
    else:
        assert actual == expected, path


def test_summary_golden_comparison_accepts_only_last_bit_float_noise() -> None:
    expected = {"status": "attention", "count": 1000, "value": 142706.27376309276}
    actual = {**expected, "value": 142706.27376309264}
    assert_summary_matches_golden(actual, expected)
    for changed in ({"status": "ok"}, {"count": 1001}, {"value": expected["value"] + 1e-6}):
        with pytest.raises(AssertionError):
            assert_summary_matches_golden({**expected, **changed}, expected)


@pytest.mark.parametrize("scenario", list(GOLDEN_CHAINS))
def test_report_summary_matches_golden(scenario: str, request: pytest.FixtureRequest) -> None:
    """H.4.5. Regenerate (deliberately, after a reviewed logic change) with:
    `uv run python -m tests.invariants.test_report` -- it rebuilds the three
    chains in a temp dir exactly as these fixtures do."""
    chain_dir = request.getfixturevalue(GOLDEN_CHAINS[scenario])
    golden = json.loads((GOLDEN_DIR / f"report_summary_{scenario}.json").read_text())
    assert_summary_matches_golden(summary_projection(chain_dir), golden)


def _regenerate_golden_summaries(out_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        session = Path(tmp)
        chains = {
            "overconfident": _run_chain(session, "overconfident_full", "overconfident"),
            "drift": _run_chain(session, "drift_full", "drift"),
            "drift_gated": _run_chain(
                session,
                "drift_gated_full",
                "drift",
                max_residual_risk=200000.0,
                backtest_returncode=2,
            ),
        }
        for scenario, chain_dir in chains.items():
            path = out_dir / f"report_summary_{scenario}.json"
            path.write_text(
                json.dumps(summary_projection(chain_dir), sort_keys=True, indent=2) + "\n"
            )
            print(f"wrote {path}")


if __name__ == "__main__":
    _regenerate_golden_summaries(GOLDEN_DIR)

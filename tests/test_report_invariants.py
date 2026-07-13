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
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from triage.artifacts import content_hash

TRIAGE_BIN = shutil.which("triage") or str(Path(__file__).resolve().parents[1] / ".venv/bin/triage")


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


def _run_chain(session_dir: Path, name: str, scenario: str, *, seed: int = 42) -> Path:
    """Run the full synth -> calibrate -> plan -> backtest chain for
    `scenario`, writing calibrator.json / policy.json / backtest.json
    directly into a chain directory (so it doubles as a `--from` dir).
    Returns the chain directory."""
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

    policy_path = chain_dir / "policy.json"
    plan_result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(data_dir / "costs.yaml"),
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
    assert backtest_result.returncode == 0, backtest_result.stderr

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
    """Spec G.6: the three pinned div ids (fig-reliability, fig-backtest,
    fig-coverage) each appear."""
    out_a, _ = report_pair
    html = out_a.read_text()
    assert "fig-reliability" in html
    assert "fig-backtest" in html
    assert "fig-coverage" in html


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


def test_report_drift_chain_exits_0(report_session: Path) -> None:
    """Spec G.9: report is not an exit-code gate: the drift chain (realized
    residual > expected) still reports exit 0 -- gating belongs to
    plan/backtest."""
    drift_chain_dir = _run_chain(report_session, "drift_full", "drift")
    out_path = report_session / "report_drift.html"
    result = run_cli(["report", "--from", str(drift_chain_dir), "-o", str(out_path)])
    assert result.returncode == 0, result.stderr

"""Backtest golden-file invariant test, driven through the real CLI.

Implements docs/test-spec-invariants.md, section E (Stage 5). New file,
matching tests/test_golden.py's pattern: regenerates the end-to-end pipeline
(synth well_calibrated seed 7 -> calibrate seed 7 -> plan with the emitted
costs.yaml -> backtest with --history = the same batch.parquet, seed 7) into
a tmp dir and compares against the committed fixture
tests/golden/backtest.json. The fixture itself is auditor-owned and must not
be regenerated or modified by this test.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from triage.artifacts import content_hash

TRIAGE_BIN = shutil.which("triage") or str(Path(__file__).resolve().parents[1] / ".venv/bin/triage")

GOLDEN_FIXTURE_PATH = Path(__file__).resolve().parent / "golden" / "backtest.json"


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the real `triage` console script and capture output."""
    return subprocess.run(
        [TRIAGE_BIN, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def regenerated_backtest(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Regenerate the golden scenario end-to-end into tmp and return the
    parsed backtest.json."""
    session_tmp = tmp_path_factory.mktemp("golden_backtest_session")
    data_dir = session_tmp / "data"

    synth_result = run_cli(
        ["synth", "--scenario", "well_calibrated", "--seed", "7", "-o", str(data_dir)]
    )
    assert synth_result.returncode == 0, synth_result.stderr

    calibrator_path = session_tmp / "calibrator.json"
    calibrate_result = run_cli(
        [
            "calibrate",
            "--golden",
            str(data_dir / "golden.parquet"),
            "--seed",
            "7",
            "-o",
            str(calibrator_path),
        ]
    )
    assert calibrate_result.returncode == 0, calibrate_result.stderr

    policy_path = session_tmp / "policy.json"
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

    backtest_path = session_tmp / "backtest.json"
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
            "7",
            "-o",
            str(backtest_path),
        ]
    )
    assert backtest_result.returncode == 0, backtest_result.stderr

    result: dict[str, Any] = json.loads(backtest_path.read_text())
    return result


@pytest.fixture(scope="module")
def fixture_backtest() -> dict[str, Any]:
    """Parsed committed golden fixture."""
    result: dict[str, Any] = json.loads(GOLDEN_FIXTURE_PATH.read_text())
    return result


def test_regenerated_matches_fixture(
    regenerated_backtest: dict[str, Any], fixture_backtest: dict[str, Any]
) -> None:
    """Spec E: regenerated backtest equals the committed fixture after
    deleting created_at in both (full dict equality, not just hash
    equality)."""
    regenerated = dict(regenerated_backtest)
    fixture = dict(fixture_backtest)
    regenerated.pop("created_at", None)
    fixture.pop("created_at", None)
    assert regenerated == fixture


def test_fixture_backtest_hash_self_consistent(fixture_backtest: dict[str, Any]) -> None:
    """Spec E: the fixture's backtest_hash matches content_hash of the
    fixture content (self-consistency of the committed artifact)."""
    recomputed = content_hash(fixture_backtest)
    assert fixture_backtest["backtest_hash"] == recomputed

"""Golden-file invariant test, driven through the real CLI.

Implements docs/test-spec-invariants.md, section B. Regenerates the
end-to-end pipeline (synth well_calibrated seed 7 -> calibrate seed 7 -> plan
with the emitted costs.yaml) into a tmp dir and compares against the
committed fixture tests/golden/policy.json. The fixture itself is
auditor-owned and must not be regenerated or modified by this test.
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

GOLDEN_FIXTURE_PATH = Path(__file__).resolve().parent / "golden" / "policy.json"


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the real `triage` console script and capture output."""
    return subprocess.run(
        [TRIAGE_BIN, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def regenerated_policy(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Regenerate the golden scenario end-to-end into tmp and return the
    parsed policy.json."""
    session_tmp = tmp_path_factory.mktemp("golden_session")
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

    result: dict[str, Any] = json.loads(policy_path.read_text())
    return result


@pytest.fixture(scope="module")
def fixture_policy() -> dict[str, Any]:
    """Parsed committed golden fixture."""
    result: dict[str, Any] = json.loads(GOLDEN_FIXTURE_PATH.read_text())
    return result


def test_regenerated_matches_fixture(
    regenerated_policy: dict[str, Any], fixture_policy: dict[str, Any]
) -> None:
    """Spec B: regenerated policy equals the committed fixture after deleting
    created_at in both (full dict equality, not just hash equality)."""
    regenerated = dict(regenerated_policy)
    fixture = dict(fixture_policy)
    regenerated.pop("created_at", None)
    fixture.pop("created_at", None)
    assert regenerated == fixture


def test_fixture_policy_hash_self_consistent(fixture_policy: dict[str, Any]) -> None:
    """Spec B: the fixture's policy_hash matches content_hash of the fixture
    content (self-consistency of the committed artifact)."""
    recomputed = content_hash(fixture_policy)
    assert fixture_policy["policy_hash"] == recomputed

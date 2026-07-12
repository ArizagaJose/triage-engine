"""Determinism invariant tests, driven through the real CLI.

Implements docs/test-spec-invariants.md, section A. Every test docstring cites
its spec item so the auditor can diff test-vs-spec 1:1.

CLI calls are expensive (~5-15s each), so each scenario is invoked exactly
once via session-scoped fixtures and shared across assertions.
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


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the real `triage` console script and capture output."""
    return subprocess.run(
        [TRIAGE_BIN, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _strip_created_at_line(text: str) -> str:
    """Drop any line containing '"created_at"' from a JSON-text blob.

    Used for byte-comparison of artifacts that embed a wall-clock timestamp.
    """
    return "\n".join(line for line in text.splitlines() if '"created_at"' not in line)


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


@pytest.fixture(scope="module")
def session_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("determinism_session")


# --- A.1: synth is byte-identical across two runs ---


@pytest.fixture(scope="module")
def synth_pair(session_tmp: Path) -> tuple[Path, Path]:
    """Run `synth --scenario overconfident --seed 42` into two temp dirs."""
    dir_a = session_tmp / "synth_a"
    dir_b = session_tmp / "synth_b"
    result_a = run_cli(["synth", "--scenario", "overconfident", "--seed", "42", "-o", str(dir_a)])
    assert result_a.returncode == 0, result_a.stderr
    result_b = run_cli(["synth", "--scenario", "overconfident", "--seed", "42", "-o", str(dir_b)])
    assert result_b.returncode == 0, result_b.stderr
    return dir_a, dir_b


@pytest.mark.parametrize(
    "filename", ["golden.parquet", "batch.parquet", "manifest.json", "costs.yaml"]
)
def test_synth_byte_identical(synth_pair: tuple[Path, Path], filename: str) -> None:
    """Spec A.1: synth output files are byte-identical across two runs with the same seed."""
    dir_a, dir_b = synth_pair
    bytes_a = _read_bytes(dir_a / filename)
    bytes_b = _read_bytes(dir_b / filename)
    assert bytes_a == bytes_b


# --- A.2: calibrate is identical (minus created_at) across two runs ---


@pytest.fixture(scope="module")
def calibrator_pair(session_tmp: Path, synth_pair: tuple[Path, Path]) -> tuple[Path, Path]:
    """Run `calibrate` twice on the same golden.parquet (seed 42)."""
    dir_a, _ = synth_pair
    golden = dir_a / "golden.parquet"
    out_a = session_tmp / "calibrator_a.json"
    out_b = session_tmp / "calibrator_b.json"
    result_a = run_cli(["calibrate", "--golden", str(golden), "--seed", "42", "-o", str(out_a)])
    assert result_a.returncode == 0, result_a.stderr
    result_b = run_cli(["calibrate", "--golden", str(golden), "--seed", "42", "-o", str(out_b)])
    assert result_b.returncode == 0, result_b.stderr
    return out_a, out_b


def test_calibrate_identical_minus_created_at(
    calibrator_pair: tuple[Path, Path],
) -> None:
    """Spec A.2: two calibrate runs (same seed) produce identical calibrator.json
    after removing the created_at line; also compare parsed dicts (minus
    created_at) and canonical_dumps strings equal."""
    out_a, out_b = calibrator_pair
    text_a = out_a.read_text()
    text_b = out_b.read_text()

    stripped_a = _strip_created_at_line(text_a)
    stripped_b = _strip_created_at_line(text_b)
    assert stripped_a == stripped_b

    dict_a = json.loads(text_a)
    dict_b = json.loads(text_b)
    dict_a.pop("created_at", None)
    dict_b.pop("created_at", None)
    assert dict_a == dict_b

    from triage.artifacts import canonical_dumps

    assert canonical_dumps(dict_a) == canonical_dumps(dict_b)


# --- A.3: plan is identical (minus created_at) across two runs, and policy_hash
#          matches an independently-recomputed content_hash ---


@pytest.fixture(scope="module")
def policy_pair(
    session_tmp: Path,
    synth_pair: tuple[Path, Path],
    calibrator_pair: tuple[Path, Path],
) -> tuple[Path, Path]:
    """Run `plan` twice on the same batch/calibrator/costs."""
    dir_a, _ = synth_pair
    calibrator_a, _ = calibrator_pair
    batch = dir_a / "batch.parquet"
    costs = dir_a / "costs.yaml"
    out_a = session_tmp / "policy_a.json"
    out_b = session_tmp / "policy_b.json"
    result_a = run_cli(
        [
            "plan",
            "--batch",
            str(batch),
            "--calibrator",
            str(calibrator_a),
            "--costs",
            str(costs),
            "-o",
            str(out_a),
        ]
    )
    assert result_a.returncode == 0, result_a.stderr
    result_b = run_cli(
        [
            "plan",
            "--batch",
            str(batch),
            "--calibrator",
            str(calibrator_a),
            "--costs",
            str(costs),
            "-o",
            str(out_b),
        ]
    )
    assert result_b.returncode == 0, result_b.stderr
    return out_a, out_b


def test_plan_identical_minus_created_at(policy_pair: tuple[Path, Path]) -> None:
    """Spec A.3: two plan runs (same inputs) produce identical policy.json
    after removing the created_at line; parsed dicts (minus created_at) equal."""
    out_a, out_b = policy_pair
    text_a = out_a.read_text()
    text_b = out_b.read_text()

    stripped_a = _strip_created_at_line(text_a)
    stripped_b = _strip_created_at_line(text_b)
    assert stripped_a == stripped_b

    dict_a = json.loads(text_a)
    dict_b = json.loads(text_b)
    dict_a.pop("created_at", None)
    dict_b.pop("created_at", None)
    assert dict_a == dict_b


def test_plan_policy_hash_equal(policy_pair: tuple[Path, Path]) -> None:
    """Spec A.3: policy_hash values equal across the two plan runs."""
    out_a, out_b = policy_pair
    dict_a: dict[str, Any] = json.loads(out_a.read_text())
    dict_b: dict[str, Any] = json.loads(out_b.read_text())
    assert dict_a["policy_hash"] == dict_b["policy_hash"]


def test_plan_policy_hash_matches_recomputed_content_hash(
    policy_pair: tuple[Path, Path],
) -> None:
    """Spec A.3: policy_hash == content_hash(parsed_policy) recomputed
    independently in the test, for both runs."""
    out_a, out_b = policy_pair
    for out_path in (out_a, out_b):
        parsed: dict[str, Any] = json.loads(out_path.read_text())
        recomputed = content_hash(parsed)
        assert parsed["policy_hash"] == recomputed


# --- A.4: calibrate with a different seed produces a different calibrator ---


def test_calibrate_different_seed_differs(
    session_tmp: Path, synth_pair: tuple[Path, Path], calibrator_pair: tuple[Path, Path]
) -> None:
    """Spec A.4: calibrate with seed 43 produces a DIFFERENT calibrator than
    seed 42 (guards against seeds being silently ignored)."""
    dir_a, _ = synth_pair
    golden = dir_a / "golden.parquet"
    out_43 = session_tmp / "calibrator_seed43.json"
    result = run_cli(["calibrate", "--golden", str(golden), "--seed", "43", "-o", str(out_43)])
    assert result.returncode == 0, result.stderr

    calibrator_42_path, _ = calibrator_pair
    dict_42: dict[str, Any] = json.loads(calibrator_42_path.read_text())
    dict_43: dict[str, Any] = json.loads(out_43.read_text())
    dict_42.pop("created_at", None)
    dict_43.pop("created_at", None)
    assert dict_42 != dict_43

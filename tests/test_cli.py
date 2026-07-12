"""End-to-end tests for the `triage` CLI, driving the real console script.

These tests invoke the installed `triage` entry point as a subprocess so that
`app_main()`'s exception-to-exit-code mapping (which lives outside the typer
app) is actually exercised -- CliRunner(app) would bypass it.

Each subprocess invocation pays a fixed ~5s import cost (polars/sklearn/
pyarrow), independent of the work done. To keep the module fast, fixtures are
session-scoped and their CompletedProcess results are reused by multiple
assertions instead of re-invoking the CLI.
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


@pytest.fixture(scope="session")
def session_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A session-scoped scratch directory."""
    return tmp_path_factory.mktemp("cli_session")


@pytest.fixture(scope="session")
def synth_data(session_tmp: Path) -> tuple[Path, subprocess.CompletedProcess[str]]:
    """Generate synthetic data once for the whole test module."""
    out_dir = session_tmp / "data"
    result = run_cli(["synth", "--scenario", "overconfident", "--seed", "42", "-o", str(out_dir)])
    assert result.returncode == 0, result.stderr
    return out_dir, result


@pytest.fixture(scope="session")
def calibrator_result(
    session_tmp: Path, synth_data: tuple[Path, subprocess.CompletedProcess[str]]
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    """Fit a calibrator once for the whole test module (this is the slow step)."""
    data_dir, _ = synth_data
    out_path = session_tmp / "calibrator.json"
    result = run_cli(
        [
            "calibrate",
            "--golden",
            str(data_dir / "golden.parquet"),
            "--seed",
            "42",
            "-o",
            str(out_path),
        ]
    )
    assert result.returncode == 0, result.stderr
    return out_path, result


@pytest.fixture(scope="session")
def calibrator_path(
    calibrator_result: tuple[Path, subprocess.CompletedProcess[str]],
) -> Path:
    return calibrator_result[0]


@pytest.fixture(scope="session")
def policy_result(
    session_tmp: Path,
    synth_data: tuple[Path, subprocess.CompletedProcess[str]],
    calibrator_path: Path,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    """Plan a policy once for the whole test module."""
    data_dir, _ = synth_data
    out_path = session_tmp / "policy.json"
    result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(data_dir / "costs.yaml"),
            "-o",
            str(out_path),
        ]
    )
    assert result.returncode == 0, result.stderr
    return out_path, result


@pytest.fixture(scope="session")
def policy_path(policy_result: tuple[Path, subprocess.CompletedProcess[str]]) -> Path:
    return policy_result[0]


# --- synth ---


def test_synth_success(
    synth_data: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    """Exit 0, four files exist, one-line summary names scenario/seed/dir."""
    out_dir, result = synth_data
    for name in ("golden.parquet", "batch.parquet", "manifest.json", "costs.yaml"):
        assert (out_dir / name).exists(), name
    assert result.returncode == 0
    assert "overconfident" in result.stdout
    assert "42" in result.stdout
    assert str(out_dir) in result.stdout


def test_synth_unknown_scenario_exit_1(session_tmp: Path) -> None:
    out_dir = session_tmp / "synth_bad_scenario"
    result = run_cli(["synth", "--scenario", "not_a_scenario", "-o", str(out_dir)])
    assert result.returncode == 1
    assert "overconfident" in result.stderr
    assert "well_calibrated" in result.stderr


# --- calibrate ---


def test_calibrate_success_artifact_shape(
    calibrator_result: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    calibrator_path, result = calibrator_result
    assert result.returncode == 0
    text = calibrator_path.read_text()
    data = json.loads(text)

    expected_keys = {
        "config",
        "created_at",
        "ece_ci",
        "ece_holdout",
        "ece_raw_holdout",
        "input_sha256",
        "method",
        "n_golden",
        "params",
        "reliability_bins",
        "seed",
        "tool_version",
    }
    assert set(data.keys()) == expected_keys

    # sorted-keys canonical form: first two top-level keys in file text order.
    lines = [line for line in text.splitlines() if line.startswith('  "')]
    first_two_keys = [line.split('"')[1] for line in lines[:2]]
    assert first_two_keys == sorted(expected_keys)[:2]


def test_calibrate_ece_gate_exit_3_artifact_still_written(
    session_tmp: Path, synth_data: tuple[Path, subprocess.CompletedProcess[str]]
) -> None:
    data_dir, _ = synth_data
    out_path = session_tmp / "cal_gate_fail.json"
    result = run_cli(
        [
            "calibrate",
            "--golden",
            str(data_dir / "golden.parquet"),
            "--seed",
            "42",
            "--max-ece",
            "0.0001",
            "-o",
            str(out_path),
        ]
    )
    assert result.returncode == 3, result.stderr
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "ece_holdout" in data


# --- plan ---


def test_plan_success_artifact_shape(
    policy_result: tuple[Path, subprocess.CompletedProcess[str]],
) -> None:
    policy_path, result = policy_result
    assert result.returncode == 0
    data = json.loads(policy_path.read_text())
    expected_keys = {
        "budget",
        "config_snapshot",
        "created_at",
        "input_sha256s",
        "policy_hash",
        "segments",
        "tool_version",
        "totals",
    }
    assert set(data.keys()) == expected_keys
    assert data["policy_hash"] == content_hash(data)


def test_plan_budget_override(
    session_tmp: Path,
    synth_data: tuple[Path, subprocess.CompletedProcess[str]],
    calibrator_path: Path,
) -> None:
    data_dir, _ = synth_data
    out_path = session_tmp / "policy_budget.json"
    result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(data_dir / "costs.yaml"),
            "--budget",
            "0.05",
            "-o",
            str(out_path),
        ]
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out_path.read_text())
    assert data["totals"]["n_review"] == 500
    assert data["config_snapshot"]["budget"]["value"] == 0.05


def test_plan_error_paths(
    session_tmp: Path,
    synth_data: tuple[Path, subprocess.CompletedProcess[str]],
    calibrator_path: Path,
) -> None:
    """Group several distinct error-exit-code checks to bound subprocess count.

    Each still invokes the CLI in isolation (fresh process, real exit code),
    just organized as one test function.
    """
    data_dir, _ = synth_data

    # Invalid --budget for a fraction budget (> 1.0) -> exit 1.
    out_bad_budget = session_tmp / "policy_bad_budget.json"
    result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(data_dir / "costs.yaml"),
            "--budget",
            "1.5",
            "-o",
            str(out_bad_budget),
        ]
    )
    assert result.returncode == 1, result.stderr

    # Missing batch file -> exit 1.
    out_missing_batch = session_tmp / "policy_missing_batch.json"
    result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "does_not_exist.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(data_dir / "costs.yaml"),
            "-o",
            str(out_missing_batch),
        ]
    )
    assert result.returncode == 1, result.stderr

    # Nonexistent calibrator file -> exit 1.
    out_missing_cal = session_tmp / "policy_missing_cal.json"
    result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(session_tmp / "no_such_calibrator.json"),
            "--costs",
            str(data_dir / "costs.yaml"),
            "-o",
            str(out_missing_cal),
        ]
    )
    assert result.returncode == 1, result.stderr


def test_plan_residual_gate_exit_2_artifact_still_written(
    session_tmp: Path,
    synth_data: tuple[Path, subprocess.CompletedProcess[str]],
    calibrator_path: Path,
) -> None:
    data_dir, _ = synth_data
    costs_text = (data_dir / "costs.yaml").read_text()
    strict_costs = costs_text.replace("max_ece: 0.05", "max_ece: 0.05\n  max_residual_risk: 0.01")
    strict_costs_path = session_tmp / "costs_strict.yaml"
    strict_costs_path.write_text(strict_costs)

    out_path = session_tmp / "policy_residual_gate.json"
    result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(strict_costs_path),
            "-o",
            str(out_path),
        ]
    )
    assert result.returncode == 2, result.stderr
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "policy_hash" in data


# --- determinism ---


def test_plan_determinism(
    session_tmp: Path,
    synth_data: tuple[Path, subprocess.CompletedProcess[str]],
    calibrator_path: Path,
    policy_path: Path,
) -> None:
    """Reuses the shared policy_path fixture as run 1; only one extra subprocess call."""
    data_dir, _ = synth_data
    out2 = session_tmp / "det_policy_2.json"
    result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(data_dir / "costs.yaml"),
            "-o",
            str(out2),
        ]
    )
    assert result.returncode == 0, result.stderr

    data1: dict[str, Any] = json.loads(policy_path.read_text())
    data2: dict[str, Any] = json.loads(out2.read_text())
    assert data1["policy_hash"] == data2["policy_hash"]

    def strip_created_at(text: str) -> list[str]:
        return [line for line in text.splitlines() if '"created_at"' not in line]

    assert strip_created_at(policy_path.read_text()) == strip_created_at(out2.read_text())


# --- exit code sanity ---


def test_help_and_bad_flag_exit_codes() -> None:
    help_result = run_cli(["--help"])
    assert help_result.returncode == 0

    bad_flag_result = run_cli(["synth", "--not-a-real-flag", "x"])
    assert bad_flag_result.returncode == 1

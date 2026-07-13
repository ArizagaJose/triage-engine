"""Backtest determinism invariant tests, driven through the real CLI.

Implements docs/test-spec-invariants.md, section D (Stage 5). New file, per
the spec's instruction not to edit tests/test_determinism.py. Imitates that
module's idiom exactly: subprocess calls to the real `triage` console
script, module-scoped fixtures so the (expensive) CLI chain runs once and is
shared across assertions, and byte/dict comparison after stripping the
`created_at` line/key.

Full chain (spec D preamble): `synth overconfident --seed 42` ->
`calibrate --seed 42` -> `plan` (emitted costs.yaml, budget from YAML) ->
`backtest --calibrator ... --seed 42`.
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


@pytest.fixture(scope="module")
def session_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("determinism_backtest_session")


@pytest.fixture(scope="module")
def data_dir(session_tmp: Path) -> Path:
    """`synth --scenario overconfident --seed 42` into a tmp dir."""
    out_dir = session_tmp / "data"
    result = run_cli(["synth", "--scenario", "overconfident", "--seed", "42", "-o", str(out_dir)])
    assert result.returncode == 0, result.stderr
    return out_dir


@pytest.fixture(scope="module")
def calibrator_path(session_tmp: Path, data_dir: Path) -> Path:
    """`calibrate` on golden.parquet, seed 42."""
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
    return out_path


@pytest.fixture(scope="module")
def policy_path(session_tmp: Path, data_dir: Path, calibrator_path: Path) -> Path:
    """`plan` on batch.parquet with the emitted costs.yaml (budget from YAML)."""
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
    return out_path


@pytest.fixture(scope="module")
def policy_dict(policy_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(policy_path.read_text())
    return result


def _run_backtest(
    session_tmp: Path,
    data_dir: Path,
    policy_path: Path,
    calibrator_path: Path,
    *,
    name: str,
    seed: int,
) -> Path:
    out_path = session_tmp / f"{name}.json"
    result = run_cli(
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
            str(out_path),
        ]
    )
    assert result.returncode == 0, result.stderr
    return out_path


# --- D.1: backtest is identical (minus created_at) across two runs, and
#          backtest_hash matches an independently-recomputed content_hash ---


@pytest.fixture(scope="module")
def backtest_pair(
    session_tmp: Path,
    data_dir: Path,
    policy_path: Path,
    calibrator_path: Path,
) -> tuple[Path, Path]:
    """Run `backtest` twice on the same inputs (seed 42)."""
    out_a = _run_backtest(
        session_tmp, data_dir, policy_path, calibrator_path, name="backtest_a", seed=42
    )
    out_b = _run_backtest(
        session_tmp, data_dir, policy_path, calibrator_path, name="backtest_b", seed=42
    )
    return out_a, out_b


def test_backtest_identical_minus_created_at(backtest_pair: tuple[Path, Path]) -> None:
    """Spec D.1: two backtest runs (same inputs, same seed) produce identical
    backtest.json files after removing the created_at line; parsed dicts
    (minus created_at) equal."""
    out_a, out_b = backtest_pair
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


def test_backtest_hash_equal(backtest_pair: tuple[Path, Path]) -> None:
    """Spec D.1: backtest_hash values equal across the two backtest runs."""
    out_a, out_b = backtest_pair
    dict_a: dict[str, Any] = json.loads(out_a.read_text())
    dict_b: dict[str, Any] = json.loads(out_b.read_text())
    assert dict_a["backtest_hash"] == dict_b["backtest_hash"]


def test_backtest_hash_matches_recomputed_content_hash(
    backtest_pair: tuple[Path, Path],
) -> None:
    """Spec D.1: backtest_hash == content_hash(parsed_backtest) recomputed
    independently in the test, for both runs."""
    out_a, out_b = backtest_pair
    for out_path in (out_a, out_b):
        parsed: dict[str, Any] = json.loads(out_path.read_text())
        recomputed = content_hash(parsed)
        assert parsed["backtest_hash"] == recomputed


# --- D.2: backtest with a different seed changes the random baseline only ---


@pytest.fixture(scope="module")
def backtest_seed43(
    session_tmp: Path,
    data_dir: Path,
    policy_path: Path,
    calibrator_path: Path,
) -> Path:
    return _run_backtest(
        session_tmp, data_dir, policy_path, calibrator_path, name="backtest_seed43", seed=43
    )


def test_random_strategy_differs_across_seeds(
    backtest_pair: tuple[Path, Path], backtest_seed43: Path
) -> None:
    """Spec D.2: backtest --seed 43 -> strategies.random.captured_loss_mean
    DIFFERS from the seed-42 run (guards against the seed being ignored)."""
    out_a, _ = backtest_pair
    dict_42: dict[str, Any] = json.loads(out_a.read_text())
    dict_43: dict[str, Any] = json.loads(backtest_seed43.read_text())
    assert (
        dict_42["strategies"]["random"]["captured_loss_mean"]
        != dict_43["strategies"]["random"]["captured_loss_mean"]
    )


def test_policy_and_raw_confidence_strategies_unaffected_by_seed(
    backtest_pair: tuple[Path, Path], backtest_seed43: Path
) -> None:
    """Spec D.2: the `policy` and `raw_confidence` strategy blocks are
    IDENTICAL dicts across seeds (no randomness may leak into them)."""
    out_a, _ = backtest_pair
    dict_42: dict[str, Any] = json.loads(out_a.read_text())
    dict_43: dict[str, Any] = json.loads(backtest_seed43.read_text())
    assert dict_42["strategies"]["policy"] == dict_43["strategies"]["policy"]
    assert dict_42["strategies"]["raw_confidence"] == dict_43["strategies"]["raw_confidence"]


# --- D.3: hash-chain links backtest -> policy -> calibrator ---


def test_hash_chain_policy(backtest_pair: tuple[Path, Path], policy_dict: dict[str, Any]) -> None:
    """Spec D.3: input_sha256s.policy equals the policy.json's own
    policy_hash."""
    out_a, _ = backtest_pair
    backtest_dict: dict[str, Any] = json.loads(out_a.read_text())
    assert backtest_dict["input_sha256s"]["policy"] == policy_dict["policy_hash"]


def test_hash_chain_calibrator(
    backtest_pair: tuple[Path, Path], policy_dict: dict[str, Any]
) -> None:
    """Spec D.3: input_sha256s.calibrator equals
    policy.input_sha256s.calibrator."""
    out_a, _ = backtest_pair
    backtest_dict: dict[str, Any] = json.loads(out_a.read_text())
    assert (
        backtest_dict["input_sha256s"]["calibrator"] == policy_dict["input_sha256s"]["calibrator"]
    )


# --- D.4: tamper check -- mismatched calibrator/policy hash-chain link ---


def test_calibrator_policy_mismatch_exits_1(
    session_tmp: Path, data_dir: Path, policy_path: Path
) -> None:
    """Spec D.4: calibrate --seed 43 into a different calibrator.json, then
    backtest with the seed-42 policy and the seed-43 calibrator -> exit 1,
    stderr names the hash mismatch."""
    calibrator_43_path = session_tmp / "calibrator_seed43.json"
    calibrate_result = run_cli(
        [
            "calibrate",
            "--golden",
            str(data_dir / "golden.parquet"),
            "--seed",
            "43",
            "-o",
            str(calibrator_43_path),
        ]
    )
    assert calibrate_result.returncode == 0, calibrate_result.stderr

    tampered_out = session_tmp / "backtest_tampered.json"
    result = run_cli(
        [
            "backtest",
            "--policy",
            str(policy_path),
            "--calibrator",
            str(calibrator_43_path),
            "--history",
            str(data_dir / "batch.parquet"),
            "--seed",
            "42",
            "-o",
            str(tampered_out),
        ]
    )
    assert result.returncode == 1, result.stderr
    assert "hash" in result.stderr.lower()
    assert not tampered_out.exists()

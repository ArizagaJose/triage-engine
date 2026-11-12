"""Determinism invariants: docs/test-spec-invariants.md A (synth, calibrate,
plan) and D (backtest). Each pair is two runs in two processes: the shared
session chain (in-process) and a fresh console-script process."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.support import Chain, json_text_without_created_at, load_json, ok
from triage.artifacts import canonical_dumps, content_hash


@pytest.fixture(scope="module")
def base(chain: Any) -> Chain:
    return chain("overconfident", 42)


@pytest.fixture(scope="module")
def session_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("determinism")


# --- A.1-A.3: synth, calibrate and plan reproduce across processes ---


@pytest.fixture(scope="module")
def synth_pair(base: Chain, cli_process: Any, session_tmp: Path) -> tuple[Path, Path]:
    """`synth --scenario overconfident --seed 42`: shared chain vs fresh process."""
    dir_b = session_tmp / "synth_b"
    ok(cli_process("synth", scenario="overconfident", seed=42, o=dir_b))
    return base.data_dir, dir_b


@pytest.fixture(scope="module")
def calibrator_pair(base: Chain, cli_process: Any, session_tmp: Path) -> tuple[Path, Path]:
    """`calibrate` (seed 42) on the same golden.parquet, twice."""
    out_b = session_tmp / "calibrator_b.json"
    ok(cli_process("calibrate", golden=base.golden, seed=42, o=out_b))
    return base.calibrator, out_b


@pytest.fixture(scope="module")
def policy_pair(base: Chain, cli_process: Any, session_tmp: Path) -> tuple[Path, Path]:
    """`plan` on the same batch/calibrator/costs, twice."""
    out_b = session_tmp / "policy_b.json"
    ok(cli_process("plan", batch=base.batch, calibrator=base.calibrator, costs=base.costs, o=out_b))
    return base.policy, out_b


@pytest.mark.parametrize(
    "filename", ["golden.parquet", "batch.parquet", "manifest.json", "costs.yaml"]
)
def test_synth_byte_identical(synth_pair: tuple[Path, Path], filename: str) -> None:
    """Spec A.1: synth output files are byte-identical across two runs with the same seed."""
    dir_a, dir_b = synth_pair
    bytes_a = (dir_a / filename).read_bytes()
    bytes_b = (dir_b / filename).read_bytes()
    assert bytes_a == bytes_b


def test_calibrate_identical_minus_created_at(
    calibrator_pair: tuple[Path, Path],
) -> None:
    """Spec A.2: two calibrate runs (same seed) produce identical calibrator.json
    after removing the created_at line; also compare parsed dicts (minus
    created_at) and canonical_dumps strings equal."""
    out_a, out_b = calibrator_pair
    text_a = out_a.read_text()
    text_b = out_b.read_text()

    stripped_a = json_text_without_created_at(text_a)
    stripped_b = json_text_without_created_at(text_b)
    assert stripped_a == stripped_b

    dict_a = json.loads(text_a)
    dict_b = json.loads(text_b)
    dict_a.pop("created_at", None)
    dict_b.pop("created_at", None)
    assert dict_a == dict_b

    assert canonical_dumps(dict_a) == canonical_dumps(dict_b)


def test_plan_identical_minus_created_at(policy_pair: tuple[Path, Path]) -> None:
    """Spec A.3: two plan runs (same inputs) produce identical policy.json
    after removing the created_at line; parsed dicts (minus created_at) equal."""
    out_a, out_b = policy_pair
    text_a = out_a.read_text()
    text_b = out_b.read_text()

    stripped_a = json_text_without_created_at(text_a)
    stripped_b = json_text_without_created_at(text_b)
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
    calibrator_pair: tuple[Path, Path], calibrator_seed43: Path
) -> None:
    """Spec A.4: calibrate with seed 43 produces a DIFFERENT calibrator than
    seed 42 (guards against seeds being silently ignored)."""
    calibrator_42_path, _ = calibrator_pair
    dict_42: dict[str, Any] = json.loads(calibrator_42_path.read_text())
    dict_43: dict[str, Any] = json.loads(calibrator_seed43.read_text())
    dict_42.pop("created_at", None)
    dict_43.pop("created_at", None)
    assert dict_42 != dict_43


# --- D.1-D.3: backtest reproduces across processes; hash-chain links ---


@pytest.fixture(scope="module")
def policy_dict(base: Chain) -> dict[str, Any]:
    return load_json(base.policy)


@pytest.fixture(scope="module")
def backtest_pair(base: Chain, cli_process: Any, session_tmp: Path) -> tuple[Path, Path]:
    """`backtest --seed 42 --n-random-draws 200` on the same inputs, twice."""
    out_b = session_tmp / "backtest_b.json"
    ok(
        cli_process(
            "backtest",
            policy=base.policy,
            calibrator=base.calibrator,
            history=base.batch,
            seed=42,
            n_random_draws=200,
            o=out_b,
        )
    )
    return base.backtest, out_b


@pytest.fixture(scope="module")
def backtest_seed43(base: Chain, cli: Any, session_tmp: Path) -> Path:
    out = session_tmp / "backtest_seed43.json"
    ok(
        cli(
            "backtest",
            policy=base.policy,
            calibrator=base.calibrator,
            history=base.batch,
            seed=43,
            n_random_draws=200,
            o=out,
        )
    )
    return out


def test_backtest_identical_minus_created_at(backtest_pair: tuple[Path, Path]) -> None:
    """Spec D.1: two backtest runs (same inputs, same seed) produce identical
    backtest.json files after removing the created_at line; parsed dicts
    (minus created_at) equal."""
    out_a, out_b = backtest_pair
    text_a = out_a.read_text()
    text_b = out_b.read_text()

    stripped_a = json_text_without_created_at(text_a)
    stripped_b = json_text_without_created_at(text_b)
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
    base: Chain, cli: Any, session_tmp: Path, calibrator_seed43: Path
) -> None:
    """Spec D.4: calibrate --seed 43 into a different calibrator.json, then
    backtest with the seed-42 policy and the seed-43 calibrator -> exit 1,
    stderr names the hash mismatch."""
    tampered_out = session_tmp / "backtest_tampered.json"
    result = cli(
        "backtest",
        policy=base.policy,
        calibrator=calibrator_seed43,
        history=base.batch,
        seed=42,
        o=tampered_out,
    )
    assert result.returncode == 1, result.stderr
    assert "hash" in result.stderr.lower()
    assert not tampered_out.exists()

"""The individual commands: outputs, flags, exit codes, artifact-first gating
(docs/interfaces.md §8). Inputs come from the shared overconfident/42 chain
(ece_holdout 0.0463, ece_ci[1] 0.0820, 1183 replayed reviews for a
1000-item budget)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support import (
    gates_row_cells,
    load_json,
    ok,
    without_created_at,
    write_costs,
)
from triage.artifacts import content_hash


@pytest.fixture(scope="module")
def base(chain: Any) -> Any:
    return chain("overconfident", 42)


@pytest.fixture(scope="module")
def ece_numbers(base: Any) -> tuple[float, float]:
    """(ece_holdout, ece_ci[1]): the point estimate passes 0.05, the upper bound fails."""
    cal = base.load("calibrator")
    point, upper = float(cal["ece_holdout"]), float(cal["ece_ci"][1])
    assert point <= 0.05 < upper, (point, upper)
    return point, upper


# --- synth ---------------------------------------------------------------------


def test_synth_writes_four_files_and_summary(cli: Any, tmp_path: Path) -> None:
    out = tmp_path / "data"
    result = ok(cli("synth", scenario="overconfident", seed=42, o=out))
    for name in ("golden.parquet", "batch.parquet", "manifest.json", "costs.yaml"):
        assert (out / name).exists(), name
    assert "overconfident" in result.stdout
    assert "42" in result.stdout
    assert str(out) in result.stdout


def test_synth_unknown_scenario_exit_1(cli: Any, tmp_path: Path) -> None:
    result = cli("synth", scenario="not_a_scenario", o=tmp_path / "out")
    assert result.returncode == 1
    assert "overconfident" in result.stderr
    assert "well_calibrated" in result.stderr


# --- calibrate -------------------------------------------------------------------


def test_calibrate_artifact_keys_in_canonical_order(base: Any) -> None:
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
    text = base.calibrator.read_text()
    assert set(load_json(base.calibrator)) == expected_keys
    top_level = [line.split('"')[1] for line in text.splitlines() if line.startswith('  "')]
    assert top_level[:2] == sorted(expected_keys)[:2]


def test_calibrate_default_gate_mode_is_point(cli: Any, base: Any, tmp_path: Path) -> None:
    """At 0.05 the default (point) gate passes although the upper bound fails."""
    result = ok(cli("calibrate", golden=base.golden, seed=42, max_ece=0.05, o=tmp_path / "c.json"))
    assert "ece_gate=point ece_holdout=" in result.stdout


@pytest.mark.parametrize(
    ("max_ece", "mode", "checked"),
    [("0.0001", None, "ece_holdout"), ("0.05", "ci_upper", "ece_ci_upper")],
)
def test_calibrate_gate_exit_3_artifact_still_written(
    cli: Any,
    base: Any,
    tmp_path: Path,
    ece_numbers: tuple[float, float],
    max_ece: str,
    mode: str | None,
    checked: str,
) -> None:
    point, upper = ece_numbers
    out = tmp_path / "calibrator.json"
    result = cli(
        "calibrate", golden=base.golden, seed=42, max_ece=max_ece, ece_gate_mode=mode, o=out
    )
    assert result.returncode == 3, result.stderr
    value = point if checked == "ece_holdout" else upper
    assert f"{checked}={value!r}" in result.stderr
    assert f"max_ece={max_ece}" in result.stderr
    assert f"ece_gate_mode={mode or 'point'}" in result.stderr
    assert f"ece_holdout={point!r}" in result.stderr
    # Artifact-first, and the gate is not part of the artifact: identical to
    # the ungated calibrator, so its content_hash is unaffected.
    assert without_created_at(load_json(out)) == without_created_at(base.load("calibrator"))
    assert content_hash(load_json(out)) == content_hash(base.load("calibrator"))


def test_calibrate_explicit_point_passes_where_ci_upper_fails(
    cli: Any, base: Any, tmp_path: Path
) -> None:
    ok(
        cli(
            "calibrate",
            golden=base.golden,
            seed=42,
            max_ece=0.05,
            ece_gate_mode="point",
            o=tmp_path / "c.json",
        )
    )


def test_calibrate_ci_upper_passes_with_looser_threshold(
    cli: Any, base: Any, tmp_path: Path, ece_numbers: tuple[float, float]
) -> None:
    result = ok(
        cli(
            "calibrate",
            golden=base.golden,
            seed=42,
            max_ece=0.09,
            ece_gate_mode="ci_upper",
            o=tmp_path / "c.json",
        )
    )
    assert f"ece_gate=ci_upper ece_ci_upper={ece_numbers[1]:.4f}<=max_ece=0.09" in result.stdout


@pytest.mark.parametrize("mode", ["point", "ci_upper"])
def test_calibrate_equality_at_threshold_passes(
    cli: Any, base: Any, tmp_path: Path, ece_numbers: tuple[float, float], mode: str
) -> None:
    """--max-ece set to the exact checked value (float repr round-trips) passes: strict `>`."""
    threshold = ece_numbers[0] if mode == "point" else ece_numbers[1]
    ok(
        cli(
            "calibrate",
            golden=base.golden,
            seed=42,
            max_ece=repr(threshold),
            ece_gate_mode=mode,
            o=tmp_path / "c.json",
        )
    )


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({"max_ece": "0.05", "ece_gate_mode": "upper"}, "should be 'point' or 'ci_upper'"),
        ({"max_ece": "0.05", "ece_gate_mode": "CI_UPPER"}, "should be 'point' or 'ci_upper'"),
        ({"ece_gate_mode": "ci_upper"}, "requires gates.max_ece"),
        ({"max_ece": "nan", "ece_gate_mode": "ci_upper"}, "max_ece must be in (0, 1]"),
        ({"max_ece": "0", "ece_gate_mode": "ci_upper"}, "max_ece must be in (0, 1]"),
        ({"max_ece": "1.5"}, "max_ece must be in (0, 1]"),
        ({"max_ece": "-0.1"}, "max_ece must be in (0, 1]"),
    ],
)
def test_calibrate_invalid_gate_options_exit_1_before_writing(
    cli: Any, base: Any, tmp_path: Path, options: dict[str, str], expected: str
) -> None:
    out = tmp_path / "calibrator.json"
    result = cli("calibrate", golden=base.golden, seed=42, o=out, **options)
    assert result.returncode == 1, result.stderr
    assert expected in result.stderr
    assert "--ece-gate-mode" in result.stderr
    assert not out.exists()


# --- plan ------------------------------------------------------------------------


def _plan(cli: Any, base: Any, out: Path, *, costs: Path | None = None, **options: Any) -> Any:
    return cli(
        "plan",
        batch=base.batch,
        calibrator=base.calibrator,
        costs=costs or base.costs,
        o=out,
        **options,
    )


def test_plan_artifact_keys_and_self_hash(base: Any) -> None:
    data = base.load("policy")
    assert set(data) == {
        "budget",
        "config_snapshot",
        "created_at",
        "input_sha256s",
        "policy_hash",
        "segments",
        "tool_version",
        "totals",
    }
    assert data["policy_hash"] == content_hash(data)
    # Default gate mode is not serialized (policy_hash stability).
    assert data["config_snapshot"]["gates"] == {"max_ece": 0.05, "max_residual_risk": None}


def test_plan_budget_override(cli: Any, base: Any, tmp_path: Path) -> None:
    out = tmp_path / "policy.json"
    ok(_plan(cli, base, out, budget=0.05))
    data = load_json(out)
    assert data["totals"]["n_review"] == 500
    assert data["config_snapshot"]["budget"]["value"] == 0.05


def test_plan_error_paths_exit_1(cli: Any, base: Any, tmp_path: Path) -> None:
    assert _plan(cli, base, tmp_path / "a.json", budget=1.5).returncode == 1
    missing_batch = cli(
        "plan",
        batch=base.data_dir / "does_not_exist.parquet",
        calibrator=base.calibrator,
        costs=base.costs,
        o=tmp_path / "b.json",
    )
    assert missing_batch.returncode == 1, missing_batch.stderr
    missing_calibrator = cli(
        "plan",
        batch=base.batch,
        calibrator=tmp_path / "no_such_calibrator.json",
        costs=base.costs,
        o=tmp_path / "c.json",
    )
    assert missing_calibrator.returncode == 1, missing_calibrator.stderr


def test_plan_residual_gate_exit_2_artifact_still_written(
    cli: Any, base: Any, tmp_path: Path
) -> None:
    costs = write_costs(base.costs, tmp_path / "costs.yaml", max_residual_risk=0.01)
    out = tmp_path / "policy.json"
    result = _plan(cli, base, out, costs=costs)
    assert result.returncode == 2, result.stderr
    assert "policy_hash" in load_json(out)


def test_plan_ci_upper_exit_3_and_mode_persisted(
    cli: Any, base: Any, tmp_path: Path, ece_numbers: tuple[float, float]
) -> None:
    costs = write_costs(base.costs, tmp_path / "costs.yaml", ece_gate_mode="ci_upper")
    out = tmp_path / "policy.json"
    result = _plan(cli, base, out, costs=costs)
    assert result.returncode == 3, result.stderr
    assert f"ece_ci_upper={ece_numbers[1]!r}" in result.stderr
    policy = load_json(out)  # artifact-first
    assert policy["config_snapshot"]["gates"] == {
        "ece_gate_mode": "ci_upper",
        "max_ece": 0.05,
        "max_residual_risk": None,
    }
    assert policy["policy_hash"] == content_hash(policy)


def test_plan_explicit_point_yields_identical_snapshot(cli: Any, base: Any, tmp_path: Path) -> None:
    costs = write_costs(base.costs, tmp_path / "costs.yaml", ece_gate_mode="point")
    out = tmp_path / "policy.json"
    ok(_plan(cli, base, out, costs=costs))
    explicit, default = load_json(out), base.load("policy")
    for key in ("config_snapshot", "segments", "totals", "budget"):
        assert explicit[key] == default[key], key


def test_plan_ci_upper_passes_with_looser_threshold(cli: Any, base: Any, tmp_path: Path) -> None:
    costs = write_costs(base.costs, tmp_path / "c.yaml", max_ece=0.09, ece_gate_mode="ci_upper")
    ok(_plan(cli, base, tmp_path / "policy.json", costs=costs))


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        ({"ece_gate_mode": "upper"}, "ece_gate_mode"),
        ({"ece_gate_mode": "ci_upper", "max_ece": None}, "requires gates.max_ece"),
    ],
)
def test_plan_invalid_gate_mode_yaml_exit_1(
    cli: Any, base: Any, tmp_path: Path, gates: dict[str, Any], expected: str
) -> None:
    costs = write_costs(base.costs, tmp_path / "costs.yaml", **gates)
    out = tmp_path / "policy.json"
    result = _plan(cli, base, out, costs=costs)
    assert result.returncode == 1, result.stderr
    assert expected in result.stderr
    assert not out.exists()


# --- backtest --------------------------------------------------------------------


def _backtest(cli: Any, base: Any, out: Path, *, policy: Path | None = None, **options: Any) -> Any:
    return cli(
        "backtest",
        policy=policy or base.policy,
        calibrator=base.calibrator,
        history=base.batch,
        o=out,
        **options,
    )


def test_backtest_without_overrun_flag_passes(cli: Any, base: Any, tmp_path: Path) -> None:
    out = tmp_path / "backtest.json"
    ok(_backtest(cli, base, out))
    assert load_json(out)["strategies"]["policy"]["n_review"] == 1183


def test_backtest_budget_overrun_gate_trips_artifact_still_written(
    cli: Any, base: Any, tmp_path: Path
) -> None:
    """Allowed 0.10 * 10000 = 1000, replayed 1183: factor 1.0 -> exit 5."""
    out = tmp_path / "backtest.json"
    result = _backtest(cli, base, out, max_budget_overrun=1.0)
    assert result.returncode == 5, result.stderr
    assert load_json(out)["strategies"]["policy"]["n_review"] == 1183


def test_backtest_budget_overrun_generous_factor_passes(
    cli: Any, base: Any, tmp_path: Path
) -> None:
    out = tmp_path / "backtest.json"
    ok(_backtest(cli, base, out, max_budget_overrun=3.0))
    assert out.exists()


def test_backtest_budget_overrun_factor_below_one_exit_1(
    cli: Any, base: Any, tmp_path: Path
) -> None:
    out = tmp_path / "backtest.json"
    assert _backtest(cli, base, out, max_budget_overrun=0.5).returncode == 1
    assert not out.exists()  # rejected before the replay


# --- ECE gate mode downstream: backtest and report of a ci_upper policy -----------


def test_ci_upper_policy_backtests_and_reports_checked_upper_bound(
    cli: Any, base: Any, tmp_path: Path, ece_numbers: tuple[float, float]
) -> None:
    run_dir = tmp_path / "artifacts"
    run_dir.mkdir()
    (run_dir / "calibrator.json").write_bytes(base.calibrator.read_bytes())
    costs = write_costs(base.costs, tmp_path / "costs.yaml", ece_gate_mode="ci_upper")
    assert _plan(cli, base, run_dir / "policy.json", costs=costs).returncode == 3

    # backtest re-validates config_snapshot into a CostModel: the mode must
    # round-trip (backtest has no ECE gate, so exit 0).
    ok(_backtest(cli, base, run_dir / "backtest.json", policy=run_dir / "policy.json"))

    report = tmp_path / "report.html"
    ok(cli("report", **{"from": run_dir}, o=report))  # informational only: exit 0
    assert gates_row_cells(report.read_text(), "max_ece") == [
        "max_ece",
        "calibrator.ece_ci[1] (upper bound of 95% bootstrap CI; ece_gate_mode=ci_upper)",
        "0.050",
        f"{ece_numbers[1]:.3f}",
        "FAIL",
    ]

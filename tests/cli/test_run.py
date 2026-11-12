"""`triage run` (docs/interfaces.md §8): the documented chain in one command.

The core contract is parity with the manual chain: same per-stage exit codes,
gate messages, artifacts (hashes included) and report. Reference numbers at
seed 42: overconfident ece_holdout 0.0463 / ece_ci[1] 0.0820, 1183 replayed
reviews for a 1000-item budget; drift realized residual ~336100 vs expected
~142706.
"""

from __future__ import annotations

import inspect
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from rich.text import Text
from typer import rich_utils

from tests.support import (
    GOLDEN_DIR,
    CliResult,
    files_under,
    gates_row_cells,
    html_without_created_at,
    load_json,
    ok,
    without_created_at,
    write_costs,
)
from triage import cli as cli_module
from triage import synth
from triage.artifacts import content_hash, file_sha256
from triage.cli import _run_cost_model
from triage.exceptions import ContractViolation

STAGES = ("synth", "calibrate", "plan", "backtest", "report")
ARTIFACTS = ("calibrator.json", "policy.json", "backtest.json")
DATA_FILES = ("golden.parquet", "batch.parquet", "manifest.json", "costs.yaml")
SYNTH_FILES = {
    "costs.yaml",
    "report.html",
    *(f"artifacts/{name}" for name in ARTIFACTS),
    *(f"data/{name}" for name in DATA_FILES),
}


def _error_line(stderr: str) -> str:
    lines = [line for line in stderr.splitlines() if line.startswith("Error:")]
    assert len(lines) == 1, stderr
    return lines[0]


def _run_statuses(stdout: str) -> dict[str, int]:
    summary = [line for line in stdout.splitlines() if line.startswith("run: ")]
    assert len(summary) == 1, stdout
    body = summary[0].removeprefix("run: ").split(" -> ")[0]
    tokens = re.findall(r"(\w+)=(ok|FAIL\(exit (\d)\))", body)
    # Rebuilding the line from the parsed tokens proves nothing was skipped.
    assert " ".join(f"{name}={status}" for name, status, _ in tokens) == body, body
    return {name: int(code) if code else 0 for name, _, code in tokens}


# --- 1. Execution: run vs the manual chain ---------------------------------------


def _check_golden_fixtures(run_dir: Path) -> None:
    """The committed fixtures were produced by the manual chain."""
    for name in ("policy.json", "backtest.json"):
        produced = load_json(run_dir / "artifacts" / name)
        assert without_created_at(produced) == without_created_at(load_json(GOLDEN_DIR / name))


def _check_residual_report_rows(run_dir: Path) -> None:
    html = (run_dir / "report.html").read_text()
    assert gates_row_cells(html, "max_residual_risk (realized)")[-1] == "FAIL"
    assert gates_row_cells(html, "max_residual_risk (planned)")[-1] == "PASS"


def _check_budget_applied(run_dir: Path) -> None:
    policy = load_json(run_dir / "artifacts" / "policy.json")
    assert policy["totals"]["n_review"] == 500
    assert policy["budget"] == {"type": "fraction", "value": 0.05}


@dataclass(frozen=True)
class Case:
    id: str
    scenario: str
    seed: int
    overrides: dict[str, str] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)  # stage -> expected exit code
    stdout_contains: tuple[str, ...] = ()
    stderr_contains: tuple[str, ...] = ()
    check: Callable[[Path], None] | None = None


CASES = [
    Case(
        "all_pass",
        "overconfident",
        42,
        stdout_contains=("ece_gate=point ece_holdout=0.0463<=max_ece=0.05",),
    ),
    Case("golden_fixtures", "well_calibrated", 7, check=_check_golden_fixtures),
    Case(
        "ci_upper",
        "overconfident",
        42,
        {"ece_gate_mode": "ci_upper"},
        {"calibrate": 3, "plan": 3},
        stderr_contains=("ece_ci_upper=0.08199937079701163",),
    ),
    Case(
        "residual",
        "drift",
        42,
        {"max_residual_risk": "200000"},
        {"backtest": 2},
        stderr_contains=("Realized residual loss",),
        check=_check_residual_report_rows,
    ),
    Case(
        "multiple_failures",
        "drift",
        42,
        {"ece_gate_mode": "ci_upper", "max_residual_risk": "200000"},
        {"calibrate": 3, "plan": 3, "backtest": 2},
    ),
    # 1183 replayed / 1000 budgeted = 1.183 exactly: strict `>` at the boundary.
    Case(
        "overrun_fails",
        "overconfident",
        42,
        {"max_budget_overrun": "1.1"},
        {"backtest": 5},
        stderr_contains=("Replayed capacity 1183",),
    ),
    Case("overrun_boundary", "overconfident", 42, {"max_budget_overrun": "1.183"}),
    Case("overrun_passes", "overconfident", 42, {"max_budget_overrun": "1.2"}),
    Case("budget_override", "overconfident", 42, {"budget": "0.05"}, check=_check_budget_applied),
    Case("zero_draws_budget", "well_calibrated", 7, {"n_random_draws": "0", "budget": "0.05"}),
]


def _expected_effective_costs(overrides: dict[str, str]) -> dict[str, Any]:
    """The cost model `run` should use, derived independently of cli.py."""
    cfg: dict[str, Any] = yaml.safe_load(synth.emitted_costs_yaml())
    if "budget" in overrides:
        cfg["budget"]["value"] = float(overrides["budget"])
    for key in ("max_ece", "max_residual_risk"):
        if key in overrides:
            cfg["gates"][key] = float(overrides[key])
    if "ece_gate_mode" in overrides:
        cfg["gates"]["ece_gate_mode"] = overrides["ece_gate_mode"]
    return cfg


def _manual_chain(cli: Any, root: Path, case: Case, costs_file: Path) -> dict[str, CliResult]:
    """The documented commands one by one, replaying the run dir's costs.yaml."""
    data, art = root / "data", root / "artifacts"
    costs = root / "costs.yaml"
    gates = yaml.safe_load(costs_file.read_text())["gates"]
    results = {"synth": cli("synth", scenario=case.scenario, seed=case.seed, o=data)}
    shutil.copyfile(costs_file, costs)
    gate_options: dict[str, Any] = {}
    if gates.get("max_ece") is not None:
        gate_options = {
            "max_ece": repr(gates["max_ece"]),
            "ece_gate_mode": gates.get("ece_gate_mode", "point"),
        }
    results["calibrate"] = cli(
        "calibrate",
        golden=data / "golden.parquet",
        seed=case.seed,
        o=art / "calibrator.json",
        **gate_options,
    )
    results["plan"] = cli(
        "plan",
        batch=data / "batch.parquet",
        calibrator=art / "calibrator.json",
        costs=costs,
        o=art / "policy.json",
    )
    results["backtest"] = cli(
        "backtest",
        policy=art / "policy.json",
        calibrator=art / "calibrator.json",
        history=data / "batch.parquet",
        seed=case.seed,
        n_random_draws=case.overrides.get("n_random_draws"),
        max_budget_overrun=case.overrides.get("max_budget_overrun"),
        o=art / "backtest.json",
    )
    results["report"] = cli("report", **{"from": art}, o=root / "report.html")
    return results


@dataclass(frozen=True)
class Executed:
    case: Case
    run_dir: Path
    run: CliResult
    manual_dir: Path
    manual: dict[str, CliResult]


@pytest.fixture(scope="module", params=CASES, ids=[c.id for c in CASES])
def executed(request: pytest.FixtureRequest, cli: Any, tmp_path_factory: Any) -> Executed:
    case: Case = request.param
    root = tmp_path_factory.mktemp(case.id)
    run_dir, manual_dir = root / "run", root / "manual"
    result = cli("run", scenario=case.scenario, seed=case.seed, o=run_dir, **case.overrides)
    manual = _manual_chain(cli, manual_dir, case, run_dir / "costs.yaml")
    return Executed(case, run_dir, result, manual_dir, manual)


def test_stage_exit_codes_match_manual_chain(executed: Executed) -> None:
    case = executed.case
    manual_codes = {stage: executed.manual[stage].returncode for stage in STAGES}
    assert manual_codes == {stage: case.failures.get(stage, 0) for stage in STAGES}
    assert _run_statuses(executed.run.stdout) == manual_codes
    first_failure = next((code for code in manual_codes.values() if code), 0)
    assert executed.run.returncode == first_failure, executed.run.stderr


def test_gate_messages_match_manual_chain(executed: Executed) -> None:
    case, run = executed.case, executed.run
    for stage, code in case.failures.items():
        manual_message = _error_line(executed.manual[stage].stderr).removeprefix("Error: ")
        assert f"[{stage}] gate failed (exit {code}): {manual_message}" in run.stderr
    if case.failures:
        listed = ", ".join(f"{stage} (exit {code})" for stage, code in case.failures.items())
        first = next(iter(case.failures))
        assert f"gate failure(s) in {listed}; exiting with the first ({first})" in run.stderr
    else:
        assert run.stderr == ""
    for text in case.stderr_contains:
        assert text in run.stderr


def test_outputs_identical_to_manual_chain(executed: Executed) -> None:
    run_dir, manual_dir = executed.run_dir, executed.manual_dir
    for name in DATA_FILES:
        assert (run_dir / "data" / name).read_bytes() == (manual_dir / "data" / name).read_bytes()
    for name in ARTIFACTS:
        run_artifact = load_json(run_dir / "artifacts" / name)
        manual_artifact = load_json(manual_dir / "artifacts" / name)
        assert without_created_at(run_artifact) == without_created_at(manual_artifact), name
    assert html_without_created_at((run_dir / "report.html").read_text()) == (
        html_without_created_at((manual_dir / "report.html").read_text())
    )


def test_layout_and_progress_output(executed: Executed) -> None:
    """Keep-going: every artifact and the report are written even when a gate fails."""
    run_dir, run = executed.run_dir, executed.run
    assert files_under(run_dir) == SYNTH_FILES
    for i, stage in enumerate(STAGES, start=1):
        assert f"[{i}/5] {stage}" in run.stdout
    assert run.stdout.rstrip().endswith(f" -> {run_dir}")
    for text in executed.case.stdout_contains:
        assert text in run.stdout


def test_effective_costs_written_and_used(executed: Executed) -> None:
    run_dir, case = executed.run_dir, executed.case
    costs_text = (run_dir / "costs.yaml").read_text()
    assert yaml.safe_load(costs_text) == _expected_effective_costs(case.overrides)
    policy = load_json(run_dir / "artifacts" / "policy.json")
    assert policy["input_sha256s"]["costs"] == file_sha256(run_dir / "costs.yaml")
    if not case.overrides or set(case.overrides) <= {"n_random_draws", "max_budget_overrun"}:
        # Copied verbatim: the policy records the source file's own hash.
        assert costs_text == synth.emitted_costs_yaml()
        assert policy["input_sha256s"]["costs"] == file_sha256(run_dir / "data" / "costs.yaml")
    mode = case.overrides.get("ece_gate_mode")
    assert policy["config_snapshot"]["gates"].get("ece_gate_mode") == mode


def test_hash_chain_and_seeds(executed: Executed) -> None:
    art = executed.run_dir / "artifacts"
    calibrator, policy, backtest = (load_json(art / name) for name in ARTIFACTS)
    assert policy["input_sha256s"]["calibrator"] == content_hash(calibrator)
    assert policy["policy_hash"] == content_hash(policy)
    assert backtest["input_sha256s"]["policy"] == policy["policy_hash"]
    assert backtest["input_sha256s"]["history"] == file_sha256(
        executed.run_dir / "data" / "batch.parquet"
    )
    assert calibrator["seed"] == backtest["seed"] == executed.case.seed


def test_case_specific_checks(executed: Executed) -> None:
    if executed.case.check is not None:
        executed.case.check(executed.run_dir)


def test_run_is_deterministic_across_processes(cli: Any, cli_process: Any, tmp_path: Path) -> None:
    """In-process vs a fresh console-script process, with a patched cost model."""
    a, b = tmp_path / "a", tmp_path / "b"
    options = {"scenario": "overconfident", "ece_gate_mode": "ci_upper"}
    assert cli("run", o=a, **options).returncode == 3
    assert cli_process("run", o=b, **options).returncode == 3
    for name in (*(f"data/{n}" for n in DATA_FILES), "costs.yaml"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    for name in ARTIFACTS:
        assert without_created_at(load_json(a / "artifacts" / name)) == without_created_at(
            load_json(b / "artifacts" / name)
        )
    assert html_without_created_at((a / "report.html").read_text()) == html_without_created_at(
        (b / "report.html").read_text()
    )


# --- 2. Validation: same verdict as the individual commands ------------------------


def _standalone(
    command: str, base: Any, out: Path, options: dict[str, str]
) -> tuple[str, dict[str, Any]]:
    """(command, options) completed with valid inputs from the base chain."""
    inputs: dict[str, dict[str, Any]] = {
        "synth": {"scenario": "drift"},
        "calibrate": {"golden": base.golden},
        "plan": {"batch": base.batch, "calibrator": base.calibrator, "costs": base.costs},
        "backtest": {
            "policy": base.policy,
            "calibrator": base.calibrator,
            "history": base.batch,
        },
    }
    return command, {**inputs[command], **options, "o": out}


# (run options, expected message, standalone counterpart: (command, options, same_message))
ValidationCase = tuple[dict[str, str], str, tuple[str, dict[str, str], bool] | None]
VALIDATION_CASES: list[ValidationCase] = [
    ({"seed": "-1"}, "seed must be a non-negative integer", ("synth", {"seed": "-1"}, True)),
    ({"seed": "-1"}, "seed must be a non-negative integer", ("calibrate", {"seed": "-1"}, True)),
    ({"seed": "-1"}, "seed must be a non-negative integer", ("backtest", {"seed": "-1"}, True)),
    (
        {"n_random_draws": "-1"},
        "n_random_draws must be an integer >= 0, got -1",
        ("backtest", {"n_random_draws": "-1"}, True),
    ),
    (
        {"n_random_draws": "1.5"},
        "Invalid value for '--n-random-draws'",
        ("backtest", {"n_random_draws": "1.5"}, True),
    ),
    *(
        (
            {"max_budget_overrun": value},
            expected,
            ("backtest", {"max_budget_overrun": value}, True),
        )
        for value, expected in (("0.5", ">= 1.0"), ("nan", "finite"), ("inf", "finite"))
    ),
    (
        {"no_such_flag": "x"},
        "No such option: --no-such-flag",
        ("backtest", {"no_such_flag": "x"}, True),
    ),
    # Gate/budget options: run reports them as cost-model overrides, the
    # commands as flags -- same exit code, same underlying rule.
    *(
        (
            {"max_ece": value},
            "max_ece must be in (0, 1]",
            ("calibrate", {"max_ece": value}, False),
        )
        for value in ("nan", "0", "1.5")
    ),
    (
        {"ece_gate_mode": "upper"},
        "Input should be 'point' or 'ci_upper'",
        ("calibrate", {"max_ece": "0.05", "ece_gate_mode": "upper"}, False),
    ),
    *(
        (
            {"budget": value},
            expected,
            ("plan", {"budget": value}, False),
        )
        for value, expected in (
            ("nan", "budget.value must be > 0 and finite"),
            ("inf", "budget.value must be > 0 and finite"),
            ("0", "budget.value must be > 0 and finite"),
            ("1.5", "must be <= 1.0"),
        )
    ),
    # run-only rules
    ({"max_residual_risk": "-1"}, "must be >= 0", None),
    ({"scenario": "nope"}, "Unknown scenario 'nope'", ("synth", {"scenario": "nope"}, True)),
    ({"scenario": "drift", "costs": "x.yaml"}, "cannot be combined with --costs", None),
    ({"scenario": None, "golden": "g.parquet"}, "missing --batch, --costs", None),
    ({"scenario": None}, "run needs either --scenario", None),
]


@pytest.mark.parametrize(
    ("run_options", "expected", "standalone"),
    VALIDATION_CASES,
    ids=[
        "-".join(f"{k}={v}" for k, v in c[0].items()) + (f"-vs-{c[2][0]}" if c[2] else "")
        for c in VALIDATION_CASES
    ],
)
def test_invalid_option_rejected_like_the_command(
    cli: Any,
    chain: Any,
    tmp_path: Path,
    run_options: dict[str, str],
    expected: str,
    standalone: tuple[str, dict[str, str], bool] | None,
) -> None:
    run_out = tmp_path / "run_out"
    via_run = cli("run", **{"scenario": "overconfident", **run_options}, o=run_out)
    assert via_run.returncode == 1, via_run.stderr
    assert expected in via_run.stderr
    assert not run_out.exists()  # rejected before anything is written
    if standalone is None:
        return
    command, options, same_message = standalone
    standalone_out = tmp_path / "standalone_out"
    name, full_options = _standalone(command, chain("overconfident"), standalone_out, options)
    alone = cli(name, **full_options)
    assert alone.returncode == 1, alone.stderr
    assert expected in alone.stderr
    if same_message:
        assert _error_line(alone.stderr) == _error_line(via_run.stderr)
    assert not standalone_out.exists()


# --- 3. run-only behaviour --------------------------------------------------------


def test_fail_fast_stops_at_first_gate_and_clears_stale_outputs(cli: Any, tmp_path: Path) -> None:
    out = tmp_path / "run"
    (out / "artifacts").mkdir(parents=True)
    for stale in ("artifacts/policy.json", "artifacts/backtest.json", "report.html"):
        (out / stale).write_text("stale")
    (out / "unrelated.txt").write_text("keep me")

    result = cli("run", scenario="overconfident", ece_gate_mode="ci_upper", fail_fast=True, o=out)
    assert result.returncode == 3, result.stderr
    assert "[3/5] plan" not in result.stdout
    assert "run:" not in result.stdout  # stopped before the summary
    assert result.stderr.startswith("Error: Calibration quality gate failed")
    assert files_under(out) == {
        "unrelated.txt",  # not owned by run: untouched
        "costs.yaml",
        "artifacts/calibrator.json",  # artifact-first: written before the gate
        *(f"data/{name}" for name in DATA_FILES),
    }


@pytest.fixture(scope="module")
def own_inputs(chain: Any, tmp_path_factory: Any) -> Path:
    """overconfident/42 data used as "own" files, plus a costs.yaml without gates."""
    d = tmp_path_factory.mktemp("own_inputs")
    for name in DATA_FILES:
        shutil.copy(chain("overconfident").data_dir / name, d / name)
    write_costs(d / "costs.yaml", d / "costs_nogates.yaml", drop_gates=True)
    return d


def _own(d: Path, costs: str = "costs.yaml", history: bool = False) -> dict[str, Any]:
    options: dict[str, Any] = {
        "golden": d / "golden.parquet",
        "batch": d / "batch.parquet",
        "costs": d / costs,
    }
    if history:
        options["history"] = d / "batch.parquet"
    return options


def test_own_data_mode_equals_the_manual_chain(
    cli: Any, chain: Any, own_inputs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    result = ok(cli("run", **_own(own_inputs, history=True), o=out))
    assert "[1/4] calibrate" in result.stdout and "synth" not in result.stdout
    assert files_under(out) == {f for f in SYNTH_FILES if not f.startswith("data/")}
    for name in ARTIFACTS:
        assert without_created_at(load_json(out / "artifacts" / name)) == without_created_at(
            load_json(chain("overconfident").artifacts_dir / name)
        ), name


def test_own_data_without_history_skips_backtest(
    cli: Any, own_inputs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    (out / "artifacts").mkdir(parents=True)
    (out / "artifacts" / "backtest.json").write_text("{}")  # stale, must be removed
    result = ok(cli("run", **_own(own_inputs), o=out))
    assert "run: calibrate=ok plan=ok report=ok" in result.stdout
    assert "[3/3] report" in result.stdout
    assert files_under(out) == {
        "costs.yaml",
        "report.html",
        "artifacts/calibrator.json",
        "artifacts/policy.json",
    }
    assert "Backtest not run" in (out / "report.html").read_text()


def test_own_data_gate_overrides_add_missing_gates_block(
    cli: Any, own_inputs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    result = ok(
        cli(
            "run",
            **_own(own_inputs, costs="costs_nogates.yaml"),
            max_ece=0.09,
            ece_gate_mode="ci_upper",
            o=out,
        )
    )
    assert "ece_gate=ci_upper ece_ci_upper=0.0820<=max_ece=0.09" in result.stdout
    assert yaml.safe_load((out / "costs.yaml").read_text())["gates"] == {
        "ece_gate_mode": "ci_upper",
        "max_ece": 0.09,
    }


def test_own_data_contract_violation_mid_run_exit_1(
    cli: Any, own_inputs: Path, tmp_path: Path
) -> None:
    """A bad history is a contract violation (exit 1), not a gate: run stops."""
    unlabeled = tmp_path / "unlabeled.csv"
    unlabeled.write_text("item_id,segment,score,predicted_value\na,s,0.5,v\n")
    out = tmp_path / "run"
    result = cli("run", **_own(own_inputs), history=unlabeled, o=out)
    assert result.returncode == 1, result.stderr
    assert "true_value" in result.stderr
    assert not (out / "report.html").exists()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"costs": "costs_nogates.yaml", "ece_gate_mode": "ci_upper"}, "requires gates.max_ece"),
        ({"max_budget_overrun": "1.5"}, "needs --history"),
        ({"golden": "missing.parquet"}, "--golden {d}/missing.parquet does not exist"),
        ({"history": "nope.csv"}, "--history {d}/nope.csv does not exist"),
        ({"costs": "."}, "is not a file"),
    ],
)
def test_own_data_invalid_inputs_exit_1_writes_nothing(
    cli: Any, own_inputs: Path, tmp_path: Path, change: dict[str, str], expected: str
) -> None:
    options = _own(own_inputs)
    for key in ("golden", "history", "costs"):
        if key in change:
            options[key] = own_inputs / change[key]
    extra = {k: v for k, v in change.items() if k not in ("golden", "history", "costs")}
    out = tmp_path / "run"
    result = cli("run", **options, **extra, o=out)
    assert result.returncode == 1, result.stderr
    assert expected.format(d=own_inputs) in result.stderr
    assert not out.exists()


def test_invalid_invocation_does_not_touch_previous_run(cli: Any, tmp_path: Path) -> None:
    """Validation happens before stale-output cleanup: a typo must not wipe a run."""
    out = tmp_path / "run"
    ok(cli("run", scenario="overconfident", o=out))
    before = {name: (out / name).read_bytes() for name in files_under(out)}
    assert cli("run", scenario="overconfident", max_ece=2, o=out).returncode == 1
    assert {name: (out / name).read_bytes() for name in files_under(out)} == before


def test_input_that_run_would_overwrite_is_rejected(
    cli: Any, own_inputs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "run"
    out.mkdir()
    shutil.copy(own_inputs / "costs.yaml", out / "costs.yaml")
    result = cli("run", **{**_own(own_inputs), "costs": out / "costs.yaml"}, o=out)
    assert result.returncode == 1
    assert "is a file `run` writes" in result.stderr
    assert (out / "costs.yaml").read_text() == (own_inputs / "costs.yaml").read_text()


def test_output_that_is_a_file_is_rejected(cli: Any, tmp_path: Path) -> None:
    out = tmp_path / "a_file"
    out.write_text("x")
    result = cli("run", scenario="drift", o=out)
    assert result.returncode == 1
    assert "is not a directory" in result.stderr


@pytest.mark.parametrize("force_terminal", [False, True])
def test_run_help_lists_its_options(
    cli: Any, monkeypatch: pytest.MonkeyPatch, force_terminal: bool
) -> None:
    monkeypatch.setattr(rich_utils, "FORCE_TERMINAL", force_terminal)
    result = ok(cli.argv(["run", "--help"]))
    # Rich may style the dashes separately from the option name in CI.
    help_text = Text.from_ansi(result.stdout).plain
    for flag in ("--scenario", "--golden", "--ece-gate-mode", "--fail-fast"):
        assert flag in help_text


# --- 4. Drift guards: every stage parameter is forwarded -----------------------------

# How `run` fills every parameter of every stage command. "run:<name>" means
# forwarded from run's own option of that name; anything else is set by run.
RUN_STAGE_PARAMETERS: dict[str, dict[str, str]] = {
    "synth": {"scenario": "run:scenario", "seed": "run:seed", "out_dir": "<run>/data"},
    "calibrate": {
        "golden": "run:golden or <run>/data/golden.parquet",
        "seed": "run:seed",
        "max_ece": "effective gates.max_ece",
        "ece_gate_mode": "effective gates.ece_gate_mode",
        "output": "<run>/artifacts/calibrator.json",
    },
    "plan": {
        "batch": "run:batch or <run>/data/batch.parquet",
        "calibrator": "<run>/artifacts/calibrator.json",
        "costs": "<run>/costs.yaml (effective)",
        "budget": "None (applied to the effective costs.yaml)",
        "output": "<run>/artifacts/policy.json",
    },
    "backtest": {
        "policy": "<run>/artifacts/policy.json",
        "calibrator": "<run>/artifacts/calibrator.json",
        "history": "run:history or <run>/data/batch.parquet",
        "seed": "run:seed",
        "n_random_draws": "run:n_random_draws",
        "output": "<run>/artifacts/backtest.json",
        "max_budget_overrun": "run:max_budget_overrun",
    },
    "report": {"from_dir": "<run>/artifacts", "output": "<run>/report.html"},
}


@pytest.mark.parametrize("stage", sorted(RUN_STAGE_PARAMETERS))
def test_every_stage_parameter_is_accounted_for(stage: str) -> None:
    """If this fails, a command gained/lost an option: decide how `run`
    forwards it, then update cli.run AND this table."""
    params = set(inspect.signature(getattr(cli_module, stage)).parameters)
    assert params == set(RUN_STAGE_PARAMETERS[stage])


def test_forwarded_options_exist_on_run_with_same_defaults() -> None:
    run_params = inspect.signature(cli_module.run).parameters
    for stage, mapping in RUN_STAGE_PARAMETERS.items():
        stage_params = inspect.signature(getattr(cli_module, stage)).parameters
        for param, source in mapping.items():
            if not source.startswith("run:"):
                continue
            run_name = source.split()[0].removeprefix("run:")
            assert run_name in run_params, (stage, param)
            stage_default = stage_params[param].default.default
            run_default = run_params[run_name].default.default
            if stage_default is not ...:  # `...` = required on the stage
                assert run_default == stage_default, (stage, param, stage_default, run_default)


def test_run_refuses_mismatched_stage_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate a command gaining an option `run` does not forward."""

    def synth_with_new_option(scenario: str, seed: int, out_dir: Path, new_option: int) -> None:
        raise AssertionError("must not be called")

    monkeypatch.setattr(cli_module, "synth", synth_with_new_option)
    defaults = {
        name: p.default.default for name, p in inspect.signature(cli_module.run).parameters.items()
    }
    defaults.update(output=tmp_path / "out", scenario="drift")
    with pytest.raises(RuntimeError, match=r"run passes .* to `synth` but it takes"):
        cli_module.run(**defaults)


# --- 5. _run_cost_model: applying overrides to a cost YAML ---------------------------

_COSTS = synth.emitted_costs_yaml()


def test_cost_model_no_overrides_returns_no_text() -> None:
    model, text = _run_cost_model(
        _COSTS, "src", budget=None, max_ece=None, ece_gate_mode=None, max_residual_risk=None
    )
    assert text is None
    assert model.gates.max_ece == 0.05 and model.gates.ece_gate_mode == "point"


def test_cost_model_overrides_patch_only_their_fields() -> None:
    model, text = _run_cost_model(
        _COSTS, "src", budget=0.2, max_ece=0.07, ece_gate_mode="ci_upper", max_residual_risk=10.0
    )
    assert text is not None
    patched = yaml.safe_load(text)
    original = yaml.safe_load(_COSTS)
    assert patched["segments"] == original["segments"]
    assert patched["budget"] == {"type": "fraction", "value": 0.2}
    assert patched["gates"] == {
        "ece_gate_mode": "ci_upper",
        "max_ece": 0.07,
        "max_residual_risk": 10.0,
    }
    assert model.budget.value == 0.2 and model.gates.ece_gate_mode == "ci_upper"


def test_cost_model_explicit_point_keeps_default_semantics() -> None:
    model, text = _run_cost_model(
        _COSTS, "src", budget=None, max_ece=None, ece_gate_mode="point", max_residual_risk=None
    )
    assert text is not None
    assert model.gates.ece_gate_mode == "point"
    assert "ece_gate_mode" not in model.model_dump()["gates"]


@pytest.mark.parametrize("gates_yaml", ["", "gates:\n", "gates: null\n"])
def test_cost_model_overrides_on_missing_or_null_gates(gates_yaml: str) -> None:
    source = "budget:\n  type: count\n  value: 5\n" + gates_yaml
    source += "segments:\n  default:\n    false_accept_cost: 1.0\n    review_cost: 1.0\n"
    model, _ = _run_cost_model(
        source, "src", budget=None, max_ece=0.1, ece_gate_mode=None, max_residual_risk=None
    )
    assert model.gates.max_ece == 0.1


def test_cost_model_invalid_override_names_flags_and_source() -> None:
    with pytest.raises(ContractViolation) as exc_info:
        _run_cost_model(
            _COSTS, "my.yaml", budget=None, max_ece=None, ece_gate_mode="x", max_residual_risk=None
        )
    message = str(exc_info.value)
    assert "--ece-gate-mode='x'" in message
    assert "my.yaml" in message
    assert "gates.ece_gate_mode" in message

"""CLI entry point and command definitions."""

import inspect
import json
import shutil
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import ValidationError

# typer >= 0.26 vendors click; usage errors are raised as these classes.
from typer._click.exceptions import ClickException

from triage import __version__, artifacts, calibration, contract, optimizer
from triage import backtest as backtest_module
from triage import costs as costs_module
from triage import report as report_module
from triage import synth as synth_module
from triage.calibration import DEFAULT_ECE_GATE_MODE, Calibrator
from triage.costs import Budget, CostModel, Gates
from triage.exceptions import (
    BudgetOverrunExceeded,
    CalibrationGateFailed,
    ContractViolation,
    ResidualRiskExceeded,
)

app = typer.Typer()


@app.command()
def synth(
    scenario: str = typer.Option(
        ..., help="Scenario: 'well_calibrated', 'overconfident', or 'drift'"
    ),
    seed: int = typer.Option(42, help="Random seed"),
    out_dir: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output directory",
    ),
) -> None:
    """Generate synthetic test data."""
    _synth_preflight(scenario, seed)
    synth_module.write(scenario, seed=seed, out_dir=out_dir)
    typer.echo(f"Wrote scenario={scenario!r} seed={seed} to {out_dir}")


# --- option preflights ---------------------------------------------------------
# Every command validates its options through exactly ONE preflight function,
# called before it reads or writes anything. `run` calls the same preflights
# with the exact arguments it later passes to each stage, so `run` and the
# individual commands cannot accept or reject different inputs
# (tests/cli/test_run.py holds the parity tests). `plan`'s only option,
# --budget, needs the YAML's budget.type and is validated by costs.Budget in
# both paths.


def _synth_preflight(scenario: str, seed: int) -> None:
    synth_module.validate_scenario(scenario)
    contract.validate_seed(seed)


def _calibrate_preflight(seed: int, max_ece: float | None, ece_gate_mode: str) -> None:
    """Gate flags use the same costs.Gates rules as the cost YAML's `gates`
    block, so the CLI and YAML gates cannot diverge either."""
    contract.validate_seed(seed)
    try:
        Gates.model_validate({"max_ece": max_ece, "ece_gate_mode": ece_gate_mode})
    except ValidationError as e:
        problems = "; ".join(err["msg"] for err in e.errors())
        raise ContractViolation(
            f"Invalid calibration gate options --max-ece={max_ece!r} "
            f"--ece-gate-mode={ece_gate_mode!r}: {problems}"
        ) from e


def _backtest_preflight(seed: int, n_random_draws: int, max_budget_overrun: float | None) -> None:
    contract.validate_seed(seed)
    backtest_module.validate_n_random_draws(n_random_draws)
    if max_budget_overrun is not None:
        backtest_module.validate_max_budget_overrun(max_budget_overrun)


@app.command()
def calibrate(
    golden: Path = typer.Option(..., help="Path to golden set (parquet or CSV)"),
    seed: int = typer.Option(42, help="Random seed"),
    max_ece: float | None = typer.Option(
        None, help="ECE gate in (0, 1] (optional; exit 3 if exceeded)"
    ),
    ece_gate_mode: str = typer.Option(
        DEFAULT_ECE_GATE_MODE,
        "--ece-gate-mode",
        help="Statistic compared against --max-ece: 'point' (ece_holdout, default) or "
        "'ci_upper' (upper bound of the bootstrap CI, ece_ci[1]). Requires --max-ece.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for calibrator.json",
    ),
) -> None:
    """Calibrate model predictions on golden set."""
    # Validate options before the (slow) fit, and before writing anything.
    _calibrate_preflight(seed, max_ece, ece_gate_mode)

    df = contract.read_input(golden, require_truth=True)
    input_sha = artifacts.file_sha256(golden)
    cal = calibration.fit(df, seed=seed, input_sha256=input_sha)

    artifact_dict = cal.to_dict()
    artifact_dict["created_at"] = artifacts.utc_now_iso()
    artifacts.write_artifact(artifact_dict, output)

    if max_ece is not None:
        calibration.check_ece_gate(cal, max_ece, ece_gate_mode)

    gate_summary = ""
    if max_ece is not None:
        stat = calibration.ece_gate_statistic(cal, ece_gate_mode)
        gate_summary = f" ece_gate={ece_gate_mode} {stat.name}={stat.value:.4f}<=max_ece={max_ece}"
    typer.echo(
        f"method={cal.method} ece_holdout={cal.ece_holdout:.4f} "
        f"ece_ci=({cal.ece_ci[0]:.4f}, {cal.ece_ci[1]:.4f}){gate_summary} -> {output}"
    )


@app.command()
def plan(
    batch: Path = typer.Option(..., help="Path to batch data (parquet or CSV)"),
    calibrator: Path = typer.Option(..., help="Path to calibrator.json"),
    costs: Path = typer.Option(..., help="Path to costs.yaml"),
    budget: float | None = typer.Option(None, help="Override budget.value from YAML"),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for policy.json",
    ),
) -> None:
    """Plan routing policy given calibrator and costs."""
    batch_df = contract.read_input(batch)

    try:
        calibrator_text = calibrator.read_text()
    except OSError as e:
        raise ContractViolation(f"Cannot read calibrator file {calibrator}: {e}") from e
    try:
        calibrator_dict = json.loads(calibrator_text)
    except json.JSONDecodeError as e:
        raise ContractViolation(f"Calibrator file {calibrator} is not valid JSON: {e}") from e
    try:
        cal = Calibrator.from_dict(calibrator_dict)
    except KeyError as e:
        raise ContractViolation(f"Calibrator file {calibrator} is missing key {e}") from e

    cost_model = costs_module.load(costs)

    if budget is not None:
        try:
            new_budget = Budget(type=cost_model.budget.type, value=budget)
        except ValidationError as e:
            raise ContractViolation(f"Invalid --budget override {budget!r}: {e}") from e
        cost_model = cost_model.model_copy(update={"budget": new_budget})

    input_sha256s = {
        "batch": artifacts.file_sha256(batch),
        # Semantic hash, not file bytes: the calibrator artifact embeds its own
        # created_at, so hashing bytes would make policy_hash depend on WHEN the
        # calibrator was fitted rather than on what it computes.
        "calibrator": artifacts.content_hash(calibrator_dict),
        "costs": artifacts.file_sha256(costs),
    }
    policy = optimizer.plan(
        batch_df, cal, cost_model, input_sha256s=input_sha256s, tool_version=__version__
    )

    artifact_dict = policy.to_dict()
    artifact_dict["created_at"] = artifacts.utc_now_iso()
    artifacts.write_artifact(artifact_dict, output)

    if cost_model.gates.max_ece is not None:
        calibration.check_ece_gate(cal, cost_model.gates.max_ece, cost_model.gates.ece_gate_mode)
    optimizer.check_residual_gate(policy, cost_model)

    totals = policy.totals
    typer.echo(
        f"n_review={totals['n_review']}/{totals['n_total']} "
        f"captured={totals['expected_captured_loss']:.1f} "
        f"residual={totals['expected_residual_loss']:.1f} "
        f"policy_hash={policy.policy_hash[:12]} -> {output}"
    )


@app.command()
def backtest(
    policy: Path = typer.Option(..., help="Path to policy.json"),
    calibrator: Path = typer.Option(..., help="Path to calibrator.json"),
    history: Path = typer.Option(..., help="Path to labeled history (parquet or CSV)"),
    seed: int = typer.Option(42, help="Random seed for the random-routing baseline"),
    n_random_draws: int = typer.Option(200, help="Number of random-baseline draws"),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for backtest.json",
    ),
    max_budget_overrun: float | None = typer.Option(
        None,
        "--max-budget-overrun",
        help="Fail (exit 5) if replayed review capacity exceeds this factor times the "
        "budget's allowed capacity on this history. Float >= 1.0. No default (no gate).",
    ),
) -> None:
    """Replay a policy against labeled history and compare it to baselines."""
    # Validate options before reading inputs or computing anything.
    _backtest_preflight(seed, n_random_draws, max_budget_overrun)
    try:
        policy_text = policy.read_text()
    except OSError as e:
        raise ContractViolation(f"Cannot read policy file {policy}: {e}") from e
    try:
        policy_dict = json.loads(policy_text)
    except json.JSONDecodeError as e:
        raise ContractViolation(f"Policy file {policy} is not valid JSON: {e}") from e

    try:
        calibrator_text = calibrator.read_text()
    except OSError as e:
        raise ContractViolation(f"Cannot read calibrator file {calibrator}: {e}") from e
    try:
        calibrator_dict = json.loads(calibrator_text)
    except json.JSONDecodeError as e:
        raise ContractViolation(f"Calibrator file {calibrator} is not valid JSON: {e}") from e
    try:
        cal = Calibrator.from_dict(calibrator_dict)
    except KeyError as e:
        raise ContractViolation(f"Calibrator file {calibrator} is missing key {e}") from e

    # The policy's thresholds are on calibrated p_error; verify the calibrator
    # supplied here is the one that produced them, not merely a compatible one.
    calibrator_hash = artifacts.content_hash(calibrator_dict)
    policy_calibrator_hash = policy_dict.get("input_sha256s", {}).get("calibrator")
    if calibrator_hash != policy_calibrator_hash:
        raise ContractViolation(
            f"Calibrator mismatch: {calibrator} hashes to {calibrator_hash!r}, but policy "
            f"{policy} was planned with calibrator hash {policy_calibrator_hash!r}"
        )

    history_df = contract.read_input(history, require_truth=True)

    input_sha256s = {
        "calibrator": calibrator_hash,
        "history": artifacts.file_sha256(history),
        "policy": artifacts.content_hash(policy_dict),
    }

    result = backtest_module.run(
        policy_dict,
        cal,
        history_df,
        seed=seed,
        n_random_draws=n_random_draws,
        input_sha256s=input_sha256s,
        tool_version=__version__,
    )

    artifact_dict = result.to_dict()
    artifact_dict["created_at"] = artifacts.utc_now_iso()
    artifacts.write_artifact(artifact_dict, output)

    try:
        cost_model = CostModel.model_validate(policy_dict.get("config_snapshot", {}))
    except ValidationError as e:
        raise ContractViolation(f"policy.config_snapshot failed validation: {e}") from e
    backtest_module.check_realized_residual_gate(result, cost_model)
    backtest_module.check_budget_overrun_gate(result, cost_model, max_budget_overrun)

    policy_strategy = result.strategies["policy"]
    typer.echo(
        f"n_review={policy_strategy['n_review']} "
        f"captured={policy_strategy['captured_loss']:.1f} "
        f"residual={policy_strategy['residual_loss']:.1f} "
        f"backtest_hash={result.backtest_hash[:12]} -> {output}"
    )


@app.command()
def report(
    from_dir: Path = typer.Option(
        ...,
        "--from",
        help="Directory containing calibrator.json and policy.json (backtest.json optional)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for report.html",
    ),
) -> None:
    """Render a self-contained HTML report from artifacts in a directory."""
    calibrator_path = from_dir / "calibrator.json"
    policy_path = from_dir / "policy.json"
    backtest_path = from_dir / "backtest.json"

    try:
        calibrator_text = calibrator_path.read_text()
    except OSError as e:
        raise ContractViolation(f"Cannot read calibrator file {calibrator_path}: {e}") from e
    try:
        calibrator_dict = json.loads(calibrator_text)
    except json.JSONDecodeError as e:
        raise ContractViolation(f"Calibrator file {calibrator_path} is not valid JSON: {e}") from e

    try:
        policy_text = policy_path.read_text()
    except OSError as e:
        raise ContractViolation(f"Cannot read policy file {policy_path}: {e}") from e
    try:
        policy_dict = json.loads(policy_text)
    except json.JSONDecodeError as e:
        raise ContractViolation(f"Policy file {policy_path} is not valid JSON: {e}") from e

    backtest_dict: dict[str, Any] | None = None
    if backtest_path.exists():
        try:
            backtest_text = backtest_path.read_text()
        except OSError as e:
            raise ContractViolation(f"Cannot read backtest file {backtest_path}: {e}") from e
        try:
            backtest_dict = json.loads(backtest_text)
        except json.JSONDecodeError as e:
            raise ContractViolation(f"Backtest file {backtest_path} is not valid JSON: {e}") from e

    html = report_module.render(
        calibrator_dict,
        policy_dict,
        backtest_dict,
        created_at=artifacts.utc_now_iso(),
        tool_version=__version__,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)

    included = "calibrator+policy" + (
        "+backtest" if backtest_dict is not None else " (no backtest)"
    )
    typer.echo(f"included={included} -> {output}")


# --- run: the whole chain in one command --------------------------------------

# Gate exceptions a stage raises AFTER writing its artifact, with the exit code
# app_main maps each one to (exit codes are a public API, see CLAUDE.md). Used
# for the run summary only; app_main stays the single place that exits.
_GATE_EXIT_CODES: dict[type[Exception], int] = {
    ResidualRiskExceeded: 2,
    CalibrationGateFailed: 3,
    BudgetOverrunExceeded: 5,
}


@dataclass(frozen=True)
class _RunLayout:
    """Where `run` writes. Every path here is owned by `run`: stale copies are
    removed before a run starts, so one run's outputs never mix with another's
    (e.g. an old backtest.json next to a new policy.json)."""

    out: Path
    data_dir: Path | None  # synthetic mode only
    costs: Path  # the effective cost model actually used by `plan`
    calibrator: Path
    policy: Path
    backtest: Path
    report: Path

    @classmethod
    def for_output(cls, out: Path, *, synthetic: bool) -> "_RunLayout":
        artifacts_dir = out / "artifacts"
        return cls(
            out=out,
            data_dir=out / "data" if synthetic else None,
            costs=out / "costs.yaml",
            calibrator=artifacts_dir / "calibrator.json",
            policy=artifacts_dir / "policy.json",
            backtest=artifacts_dir / "backtest.json",
            report=out / "report.html",
        )

    @property
    def artifacts_dir(self) -> Path:
        return self.calibrator.parent

    def owned_files(self) -> list[Path]:
        files = [self.costs, self.calibrator, self.policy, self.backtest, self.report]
        if self.data_dir is not None:
            files += [
                self.data_dir / name
                for name in ("golden.parquet", "batch.parquet", "manifest.json", "costs.yaml")
            ]
        return files


def _run_cost_model(
    source_text: str,
    source_label: str,
    *,
    budget: float | None,
    max_ece: float | None,
    ece_gate_mode: str | None,
    max_residual_risk: float | None,
) -> tuple[CostModel, str | None]:
    """Apply `run`'s overrides to an already-valid cost YAML.

    Returns the effective CostModel and the effective YAML text, or None for
    the text when no override was given -- the caller then copies the source
    bytes verbatim, so `policy.input_sha256s.costs` is exactly what the
    equivalent manual chain records.
    """
    overrides: dict[str, Any] = {}
    if budget is not None:
        overrides["--budget"] = budget
    if max_ece is not None:
        overrides["--max-ece"] = max_ece
    if ece_gate_mode is not None:
        overrides["--ece-gate-mode"] = ece_gate_mode
    if max_residual_risk is not None:
        overrides["--max-residual-risk"] = max_residual_risk

    raw = yaml.safe_load(source_text)
    if not overrides:
        return CostModel.model_validate(raw), None

    if budget is not None:
        raw["budget"] = {**raw["budget"], "value": budget}
    gates = dict(raw.get("gates") or {})
    if max_ece is not None:
        gates["max_ece"] = max_ece
    if ece_gate_mode is not None:
        gates["ece_gate_mode"] = ece_gate_mode
    if max_residual_risk is not None:
        gates["max_residual_risk"] = max_residual_risk
    raw["gates"] = gates
    try:
        cost_model = CostModel.model_validate(raw)
    except ValidationError as e:
        flags = " ".join(f"{flag}={value!r}" for flag, value in overrides.items())
        problems = "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise ContractViolation(
            f"Invalid run overrides ({flags}) for cost model {source_label}: {problems}"
        ) from e
    return cost_model, yaml.safe_dump(raw, sort_keys=True)


@app.command()
def run(
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Run directory: receives costs.yaml, artifacts/, report.html (and data/ for "
        "--scenario)",
    ),
    scenario: str | None = typer.Option(
        None,
        help="Synthetic mode: generate this scenario ('well_calibrated', 'overconfident', "
        "'drift') and use its batch as the labeled history. Excludes --golden/--batch/"
        "--costs/--history.",
    ),
    seed: int = typer.Option(
        42, help="Seed for every randomized stage: synth, calibrate, backtest baseline"
    ),
    golden: Path | None = typer.Option(None, help="Own-data mode: golden set (with true_value)"),
    batch: Path | None = typer.Option(None, help="Own-data mode: batch to plan"),
    costs: Path | None = typer.Option(None, help="Own-data mode: cost model YAML"),
    history: Path | None = typer.Option(
        None, help="Own-data mode, optional: labeled history to backtest (omit to skip backtest)"
    ),
    budget: float | None = typer.Option(None, help="Override budget.value of the cost model"),
    max_ece: float | None = typer.Option(None, help="Override gates.max_ece, in (0, 1]"),
    ece_gate_mode: str | None = typer.Option(
        None,
        "--ece-gate-mode",
        help="Override gates.ece_gate_mode: 'point' or 'ci_upper' (needs a max_ece)",
    ),
    max_residual_risk: float | None = typer.Option(
        None, help="Override gates.max_residual_risk (>= 0)"
    ),
    max_budget_overrun: float | None = typer.Option(
        None,
        "--max-budget-overrun",
        help="Passed to backtest (exit 5 gate). Float >= 1.0; needs a history.",
    ),
    n_random_draws: int = typer.Option(200, help="Random-baseline draws for backtest (>= 0)"),
    fail_fast: bool = typer.Option(
        False,
        "--fail-fast",
        help="Stop at the first failed gate (like chaining the commands with `set -e`). "
        "Default: keep going so every artifact and the report are written, then exit with "
        "the first failed gate's code.",
    ),
) -> None:
    """Run the whole chain -- [synth] -> calibrate -> plan -> [backtest] -> report --
    with the same code and exit codes as the individual commands."""
    # --- 1. Validate everything before writing anything -----------------------
    own_inputs = {"--golden": golden, "--batch": batch, "--costs": costs, "--history": history}
    given_inputs = [flag for flag, path in own_inputs.items() if path is not None]
    if scenario is not None and given_inputs:
        raise ContractViolation(
            f"--scenario cannot be combined with {', '.join(given_inputs)}: synthetic mode "
            "generates its own golden set, batch, costs and history"
        )
    if scenario is None:
        missing = [flag for flag in ("--golden", "--batch", "--costs") if own_inputs[flag] is None]
        if missing:
            raise ContractViolation(
                "run needs either --scenario (synthetic mode) or --golden, --batch and "
                f"--costs (own-data mode); missing {', '.join(missing)}"
            )
    else:
        _synth_preflight(scenario, seed)
    has_history = scenario is not None or history is not None
    if has_history:
        _backtest_preflight(seed, n_random_draws, max_budget_overrun)
    elif max_budget_overrun is not None:
        raise ContractViolation("--max-budget-overrun gates the backtest, which needs --history")
    for flag, path in own_inputs.items():
        if path is not None and not path.is_file():
            raise ContractViolation(f"{flag} {path} does not exist or is not a file")
    if output.exists() and not output.is_dir():
        raise ContractViolation(f"--output {output} exists and is not a directory")

    layout = _RunLayout.for_output(output, synthetic=scenario is not None)
    owned = {p.resolve() for p in layout.owned_files()}
    for flag, path in own_inputs.items():
        if path is not None and path.resolve() in owned:
            raise ContractViolation(
                f"{flag} {path} is a file `run` writes in {output}; move it or pick another "
                "--output"
            )
    for path in layout.owned_files():
        if path.exists() and not path.is_file():
            raise ContractViolation(f"{path} exists and is not a file; `run` needs to write it")

    if scenario is not None:
        source_text, source_label = synth_module.emitted_costs_yaml(), f"scenario {scenario!r}"
    else:
        assert costs is not None  # guaranteed by the mode checks above
        costs_module.load(costs)  # validates the source file with its usual messages
        source_text, source_label = costs.read_text(), str(costs)
    cost_model, effective_costs_text = _run_cost_model(
        source_text,
        source_label,
        budget=budget,
        max_ece=max_ece,
        ece_gate_mode=ece_gate_mode,
        max_residual_risk=max_residual_risk,
    )
    # The calibrate stage is gated with the effective cost model's gates.
    calibrate_max_ece = cost_model.gates.max_ece
    calibrate_gate_mode: str = cost_model.gates.ece_gate_mode
    _calibrate_preflight(seed, calibrate_max_ece, calibrate_gate_mode)

    # --- 2. Clear stale outputs of a previous run ------------------------------
    for path in layout.owned_files():
        path.unlink(missing_ok=True)

    # --- 3. Run the stages (the real command functions, so `run` cannot drift
    # from the documented chain) ------------------------------------------------
    failures: list[tuple[str, Exception]] = []
    statuses: list[str] = []
    stage_names = (["synth"] if scenario is not None else []) + ["calibrate", "plan"]
    stage_names += (["backtest"] if has_history else []) + ["report"]

    def stage(name: str, fn: Any, **kwargs: Any) -> None:
        # Every parameter of the command must be passed explicitly: an omitted
        # one would silently receive typer's OptionInfo default object. This
        # turns "a command gained an option that run does not forward" into a
        # loud failure instead of a silent drift between run and the command.
        expected = set(inspect.signature(fn).parameters)
        if set(kwargs) != expected:
            raise RuntimeError(
                f"run passes {sorted(kwargs)} to `{name}` but it takes {sorted(expected)}"
            )
        typer.echo(f"[{stage_names.index(name) + 1}/{len(stage_names)}] {name}")
        try:
            fn(**kwargs)
        except (CalibrationGateFailed, ResidualRiskExceeded, BudgetOverrunExceeded) as e:
            if fail_fast:
                raise
            code = _GATE_EXIT_CODES[type(e)]
            typer.echo(f"[{name}] gate failed (exit {code}): {e}", err=True)
            failures.append((name, e))
            statuses.append(f"{name}=FAIL(exit {code})")
            return
        statuses.append(f"{name}=ok")

    if scenario is not None:
        assert layout.data_dir is not None
        stage("synth", synth, scenario=scenario, seed=seed, out_dir=layout.data_dir)
        golden = layout.data_dir / "golden.parquet"
        batch = layout.data_dir / "batch.parquet"
        history = batch  # the synthetic batch is labeled; it doubles as history
        costs = layout.data_dir / "costs.yaml"
    assert golden is not None and batch is not None and costs is not None

    output.mkdir(parents=True, exist_ok=True)
    if effective_costs_text is None:
        shutil.copyfile(costs, layout.costs)
    else:
        layout.costs.write_text(effective_costs_text)

    stage(
        "calibrate",
        calibrate,
        golden=golden,
        seed=seed,
        max_ece=calibrate_max_ece,
        ece_gate_mode=calibrate_gate_mode,
        output=layout.calibrator,
    )
    stage(
        "plan",
        plan,
        batch=batch,
        calibrator=layout.calibrator,
        costs=layout.costs,
        budget=None,  # already applied to the effective costs.yaml
        output=layout.policy,
    )
    if history is not None:
        stage(
            "backtest",
            backtest,
            policy=layout.policy,
            calibrator=layout.calibrator,
            history=history,
            seed=seed,
            n_random_draws=n_random_draws,
            output=layout.backtest,
            max_budget_overrun=max_budget_overrun,
        )
    stage("report", report, from_dir=layout.artifacts_dir, output=layout.report)

    typer.echo(f"run: {' '.join(statuses)} -> {output}")
    if failures:
        first_stage, first_error = failures[0]
        summary = ", ".join(f"{name} (exit {_GATE_EXIT_CODES[type(e)]})" for name, e in failures)
        # Re-raise as the FIRST failure's type: app_main maps it to that exit code.
        raise type(first_error)(
            f"run: gate failure(s) in {summary}; exiting with the first ({first_stage}). "
            f"All artifacts and the report were still written to {output}"
        )


def app_main() -> None:
    """Wrapper function for [project.scripts] entry point.

    Maps exceptions to exit codes:
    - ContractViolation -> 1
    - ResidualRiskExceeded -> 2
    - CalibrationGateFailed -> 3
    - BudgetOverrunExceeded -> 5
    - Other exceptions -> 4
    - Success -> 0
    """
    try:
        app(standalone_mode=False)
    except ContractViolation as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except ResidualRiskExceeded as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except CalibrationGateFailed as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(3)
    except BudgetOverrunExceeded as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(5)
    except typer.Exit as e:
        sys.exit(e.exit_code)
    except ClickException as e:
        # Bad CLI usage is an input contract violation, not an internal error.
        e.show()
        sys.exit(1)
    except typer.Abort:
        typer.echo("Aborted.", err=True)
        sys.exit(4)
    except Exception as e:
        typer.echo(f"Internal error: {e}", err=True)
        traceback.print_exc(file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    app_main()

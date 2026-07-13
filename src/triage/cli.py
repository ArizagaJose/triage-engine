"""CLI entry point and command definitions."""

import json
import sys
import traceback
from pathlib import Path

import typer
from pydantic import ValidationError

# typer >= 0.26 vendors click; usage errors are raised as these classes.
from typer._click.exceptions import ClickException

from triage import __version__, artifacts, calibration, contract, optimizer
from triage import backtest as backtest_module
from triage import costs as costs_module
from triage import synth as synth_module
from triage.calibration import Calibrator
from triage.costs import Budget, CostModel
from triage.exceptions import CalibrationGateFailed, ContractViolation, ResidualRiskExceeded

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
    synth_module.write(scenario, seed=seed, out_dir=out_dir)
    typer.echo(f"Wrote scenario={scenario!r} seed={seed} to {out_dir}")


@app.command()
def calibrate(
    golden: Path = typer.Option(..., help="Path to golden set (parquet or CSV)"),
    seed: int = typer.Option(42, help="Random seed"),
    max_ece: float | None = typer.Option(None, help="ECE gate (optional)"),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for calibrator.json",
    ),
) -> None:
    """Calibrate model predictions on golden set."""
    df = contract.read_input(golden, require_truth=True)
    input_sha = artifacts.file_sha256(golden)
    cal = calibration.fit(df, seed=seed, input_sha256=input_sha)

    artifact_dict = cal.to_dict()
    artifact_dict["created_at"] = artifacts.utc_now_iso()
    artifacts.write_artifact(artifact_dict, output)

    if max_ece is not None:
        calibration.check_ece_gate(cal, max_ece)

    typer.echo(
        f"method={cal.method} ece_holdout={cal.ece_holdout:.4f} "
        f"ece_ci=({cal.ece_ci[0]:.4f}, {cal.ece_ci[1]:.4f}) -> {output}"
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
        calibration.check_ece_gate(cal, cost_model.gates.max_ece)
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
) -> None:
    """Replay a policy against labeled history and compare it to baselines."""
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

    policy_strategy = result.strategies["policy"]
    typer.echo(
        f"n_review={policy_strategy['n_review']} "
        f"captured={policy_strategy['captured_loss']:.1f} "
        f"residual={policy_strategy['residual_loss']:.1f} "
        f"backtest_hash={result.backtest_hash[:12]} -> {output}"
    )


def app_main() -> None:
    """Wrapper function for [project.scripts] entry point.

    Maps exceptions to exit codes:
    - ContractViolation -> 1
    - ResidualRiskExceeded -> 2
    - CalibrationGateFailed -> 3
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

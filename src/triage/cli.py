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
from triage import costs as costs_module
from triage import synth as synth_module
from triage.calibration import Calibrator
from triage.costs import Budget
from triage.exceptions import CalibrationGateFailed, ContractViolation, ResidualRiskExceeded

app = typer.Typer()


@app.command()
def synth(
    scenario: str = typer.Option(..., help="Scenario: 'well_calibrated' or 'overconfident'"),
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
        "calibrator": artifacts.file_sha256(calibrator),
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

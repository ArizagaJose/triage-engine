"""CLI entry point and command definitions."""

import sys
import traceback
from pathlib import Path

import typer

# typer >= 0.26 vendors click; usage errors are raised as these classes.
from typer._click.exceptions import ClickException

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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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

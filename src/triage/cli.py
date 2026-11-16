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
from triage import compare as compare_module
from triage import compare_report as compare_report_module
from triage import costs as costs_module
from triage import report as report_module
from triage import review_queue as review_queue_module
from triage import synth as synth_module
from triage.calibration import DEFAULT_ECE_GATE_MODE, Calibrator
from triage.costs import Budget, CostModel, Gates
from triage.exceptions import (
    BudgetOverrunExceeded,
    CalibrationGateFailed,
    CandidateRegression,
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
# (tests/cli/test_run.py holds the parity tests). `plan`'s --budget needs the
# YAML's budget.type and is validated by costs.Budget in both paths; its
# review-queue options go through `_plan_preflight`.


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


def _queue_manifest_path(queue_csv: Path | None, queue_parquet: Path | None) -> Path | None:
    """Where `plan` writes review_queue.json: next to the queue export(s).

    None when no export was requested. The manifest records file NAMES only,
    so it must sit in the same directory as the files it describes;
    `_plan_preflight` enforces that both exports share that directory.
    """
    first = queue_csv if queue_csv is not None else queue_parquet
    if first is None:
        return None
    return first.parent / review_queue_module.MANIFEST_NAME


def _plan_preflight(
    *,
    batch: Path,
    calibrator: Path,
    costs: Path,
    output: Path,
    queue_csv: Path | None,
    queue_parquet: Path | None,
) -> None:
    """Validate plan's review-queue options before anything is read or written.

    --budget is not checked here: it needs the YAML's budget.type and is
    validated by costs.Budget once the cost model is loaded (same rule in
    `plan` and `run`).
    """
    for flag, path, suffix in (
        ("--queue-csv", queue_csv, ".csv"),
        ("--queue-parquet", queue_parquet, ".parquet"),
    ):
        if path is not None and path.suffix.lower() != suffix:
            raise ContractViolation(f"{flag} {path} must end in {suffix}")
    if (
        queue_csv is not None
        and queue_parquet is not None
        and queue_csv.resolve().parent != queue_parquet.resolve().parent
    ):
        raise ContractViolation(
            f"--queue-csv {queue_csv} and --queue-parquet {queue_parquet} must be in the same "
            f"directory: {review_queue_module.MANIFEST_NAME} is written next to them and "
            "records their file names"
        )
    manifest = _queue_manifest_path(queue_csv, queue_parquet)
    if manifest is None:
        return
    written = {
        "--queue-csv": queue_csv,
        "--queue-parquet": queue_parquet,
        f"the queue manifest ({review_queue_module.MANIFEST_NAME})": manifest,
    }
    protected = {"--batch": batch, "--calibrator": calibrator, "--costs": costs, "--output": output}
    for what, path in written.items():
        if path is None:
            continue
        for flag, other in protected.items():
            if path.resolve() == other.resolve():
                raise ContractViolation(
                    f"{what} {path} would overwrite {flag} {other}; pick another path"
                )
        if path.exists() and not path.is_file():
            raise ContractViolation(f"{what} {path} exists and is not a file")


def _evaluate_plan_gates(
    cal: Calibrator, policy: optimizer.Policy, cost_model: CostModel
) -> list[tuple[str, CalibrationGateFailed | ResidualRiskExceeded]]:
    """Run plan's gates in their pinned order (max_ece, then max_residual_risk)
    WITHOUT stopping at the first failure, and return every failure as
    (gate name, the exception the check raised).

    The checks themselves are the real `check_ece_gate` / `check_residual_gate`
    (single source of gate semantics); only their gate exceptions are caught.
    Anything else -- e.g. a ContractViolation for an unknown ECE mode --
    propagates immediately, as before. The caller records the outcome in the
    review-queue manifest and then raises the FIRST failure, so exit codes
    and messages are exactly those of the short-circuiting chain.
    """
    failures: list[tuple[str, CalibrationGateFailed | ResidualRiskExceeded]] = []
    if cost_model.gates.max_ece is not None:
        try:
            calibration.check_ece_gate(
                cal, cost_model.gates.max_ece, cost_model.gates.ece_gate_mode
            )
        except CalibrationGateFailed as e:
            failures.append(("max_ece", e))
    try:
        optimizer.check_residual_gate(policy, cost_model)
    except ResidualRiskExceeded as e:
        failures.append(("max_residual_risk", e))
    return failures


def _compare_preflight(
    seed: int, n_bootstrap: int, ci_level: float, fail_on_regression: bool
) -> None:
    compare_module.validate_uncertainty_options(seed, n_bootstrap, ci_level)
    if fail_on_regression and n_bootstrap == 0:
        raise ContractViolation(
            "--fail-on-regression gates on the bootstrap CI, which needs --n-bootstrap > 0"
        )


def _read_json_artifact(path: Path, label: str) -> dict[str, Any]:
    """Read one JSON artifact: unreadable, invalid JSON, or not an object is a
    contract violation (exit 1) naming the file."""
    try:
        text = path.read_text()
    except OSError as e:
        raise ContractViolation(f"Cannot read {label} file {path}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ContractViolation(f"{label.capitalize()} file {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ContractViolation(f"{label.capitalize()} file {path} must contain a JSON object")
    return data


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
    budget_strategy: str | None = typer.Option(
        None,
        "--budget-strategy",
        help="Override budget.strategy from YAML: 'prefix' (default; stop at the first item "
        "that does not fit) or 'fill' (money budgets only; skip items that do not fit and "
        "keep the better of two rankings by total expected net benefit).",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for policy.json",
    ),
    queue_csv: Path | None = typer.Option(
        None,
        "--queue-csv",
        help="Also export the review queue (the optimizer's exact selection, in priority "
        "order) as CSV. review_queue.json is written next to it.",
    ),
    queue_parquet: Path | None = typer.Option(
        None,
        "--queue-parquet",
        help="Also export the review queue as Parquet (same directory as --queue-csv if both "
        "are given). review_queue.json is written next to it.",
    ),
) -> None:
    """Plan routing policy given calibrator and costs."""
    _plan_preflight(
        batch=batch,
        calibrator=calibrator,
        costs=costs,
        output=output,
        queue_csv=queue_csv,
        queue_parquet=queue_parquet,
    )
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

    if budget is not None or budget_strategy is not None:
        # Revalidate the whole Budget (type/value/strategy rules interact);
        # a field that is not overridden keeps its YAML value.
        overridden = []
        if budget is not None:
            overridden.append(f"--budget override {budget!r}")
        if budget_strategy is not None:
            overridden.append(f"--budget-strategy override {budget_strategy!r}")
        try:
            new_budget = Budget.model_validate(
                {
                    "type": cost_model.budget.type,
                    "value": budget if budget is not None else cost_model.budget.value,
                    "strategy": (
                        budget_strategy
                        if budget_strategy is not None
                        else cost_model.budget.strategy
                    ),
                }
            )
        except ValidationError as e:
            raise ContractViolation(f"Invalid {' and '.join(overridden)}: {e}") from e
        cost_model = cost_model.model_copy(update={"budget": new_budget})

    input_sha256s = {
        "batch": artifacts.file_sha256(batch),
        # Semantic hash, not file bytes: the calibrator artifact embeds its own
        # created_at, so hashing bytes would make policy_hash depend on WHEN the
        # calibrator was fitted rather than on what it computes.
        "calibrator": artifacts.content_hash(calibrator_dict),
        "costs": artifacts.file_sha256(costs),
    }
    # One allocation, two views: the policy and the review queue behind it. The
    # policy is identical to optimizer.plan()'s, so policy.json (and its
    # policy_hash) does not depend on whether a queue export was requested.
    result = optimizer.plan_with_queue(
        batch_df, cal, cost_model, input_sha256s=input_sha256s, tool_version=__version__
    )
    policy = result.policy

    created_at = artifacts.utc_now_iso()
    artifact_dict = policy.to_dict()
    artifact_dict["created_at"] = created_at
    artifacts.write_artifact(artifact_dict, output)

    # Gates are evaluated before the queue is written so its manifest records
    # the outcome; they are RAISED only after every artifact is on disk
    # (artifact-first, same exit code and message as the first failing check).
    gate_failures = _evaluate_plan_gates(cal, policy, cost_model)

    queue_summary = ""
    manifest_path = _queue_manifest_path(queue_csv, queue_parquet)
    if manifest_path is not None:
        gate_status = (
            review_queue_module.GateStatus(
                status="failed", failed=tuple(name for name, _ in gate_failures)
            )
            if gate_failures
            else review_queue_module.GATE_STATUS_PASS
        )
        manifest = review_queue_module.export(
            result.review_queue,
            csv_path=queue_csv,
            parquet_path=queue_parquet,
            manifest_path=manifest_path,
            policy_hash=policy.policy_hash,
            input_sha256s=policy.input_sha256s,
            totals=policy.totals,
            gate_status=gate_status,
            tool_version=__version__,
            created_at=created_at,
        )
        queue_summary = f" queue={manifest['n_rows']} rows -> {manifest_path}"

    if gate_failures:
        raise gate_failures[0][1]

    totals = policy.totals
    # Only opt-in strategies add to the line; the default summary is unchanged.
    strategy_summary = ""
    if policy.allocation is not None:
        strategy_summary = (
            f"budget_strategy={cost_model.budget.strategy} "
            f"selected_candidate={policy.allocation['selected_candidate']} "
        )
    typer.echo(
        f"n_review={totals['n_review']}/{totals['n_total']} "
        f"captured={totals['expected_captured_loss']:.1f} "
        f"residual={totals['expected_residual_loss']:.1f} "
        f"{strategy_summary}"
        f"policy_hash={policy.policy_hash[:12]} -> {output}{queue_summary}"
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


COMPARISON_FILENAME = "comparison.json"


@app.command()
def compare(
    current_policy: Path = typer.Option(..., help="Path to the CURRENT policy.json"),
    current_calibrator: Path = typer.Option(
        ..., help="Path to the calibrator.json the current policy was planned with"
    ),
    candidate_policy: Path = typer.Option(..., help="Path to the CANDIDATE policy.json"),
    candidate_calibrator: Path = typer.Option(
        ..., help="Path to the calibrator.json the candidate policy was planned with"
    ),
    history: Path = typer.Option(..., help="Path to labeled history (parquet or CSV)"),
    eval_costs: Path | None = typer.Option(
        None,
        help="Explicit evaluation cost model (costs.yaml) that prices BOTH policies. Required "
        "when their own costs differ on a history segment. Its budget and gates are ignored.",
    ),
    seed: int = typer.Option(42, help="Seed for the paired bootstrap CI"),
    n_bootstrap: int = typer.Option(
        compare_module.DEFAULT_N_BOOTSTRAP,
        help="Paired bootstrap resamples for the CI of each delta (0 = no CI)",
    ),
    ci_level: float = typer.Option(
        compare_module.DEFAULT_CI_LEVEL, help="Bootstrap CI level, in (0, 1)"
    ),
    fail_on_regression: bool = typer.Option(
        False,
        "--fail-on-regression",
        help="Exit 6 if the candidate is significantly worse (bootstrap CI): higher total "
        "cost, or worse on captured loss/reviews/spend and better on none. "
        f"{COMPARISON_FILENAME} is written first.",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help=f"Output directory; receives {COMPARISON_FILENAME}",
    ),
) -> None:
    """Compare a current and a candidate policy on the same labeled history."""
    _compare_preflight(seed, n_bootstrap, ci_level, fail_on_regression)
    if output.exists() and not output.is_dir():
        raise ContractViolation(f"--output {output} exists and is not a directory")

    loaded = {
        "current_policy": _read_json_artifact(current_policy, "current policy"),
        "current_calibrator": _read_json_artifact(current_calibrator, "current calibrator"),
        "candidate_policy": _read_json_artifact(candidate_policy, "candidate policy"),
        "candidate_calibrator": _read_json_artifact(candidate_calibrator, "candidate calibrator"),
    }
    history_df = contract.read_input(history, require_truth=True)
    eval_cost_model = costs_module.load(eval_costs) if eval_costs is not None else None

    result = compare_module.run(
        **loaded,
        history=history_df,
        history_sha256=artifacts.file_sha256(history),
        eval_costs=eval_cost_model,
        eval_costs_sha256=artifacts.file_sha256(eval_costs) if eval_costs is not None else None,
        seed=seed,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        tool_version=__version__,
    )

    out_path = output / COMPARISON_FILENAME
    artifact_dict = result.to_dict()
    artifact_dict["created_at"] = artifacts.utc_now_iso()
    artifacts.write_artifact(artifact_dict, out_path)

    if fail_on_regression:
        compare_module.check_regression_gate(result)

    d = result.delta
    ci_verdict = result.verdict["ci"]
    ci_part = f" (ci: {ci_verdict['label']})" if ci_verdict is not None else ""
    typer.echo(
        f"verdict={result.verdict['label']}{ci_part} "
        f"d_captured={d['captured_loss']:+.1f} d_reviews={d['n_review']:+d} "
        f"d_spend={d['review_spend']:+.1f} d_total_cost={d['total_cost']:+.1f} "
        f"basis={result.basis['source']} comparison_hash={result.comparison_hash[:12]} "
        f"-> {out_path}"
    )


@app.command()
def report(
    from_dir: Path | None = typer.Option(
        None,
        "--from",
        help="Single-policy report: directory containing calibrator.json and policy.json "
        "(backtest.json optional)",
    ),
    compare_dir: Path | None = typer.Option(
        None,
        "--compare",
        help=f"Comparison report: directory containing {COMPARISON_FILENAME} "
        "(written by `triage compare`)",
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path for the HTML report",
    ),
) -> None:
    """Render a self-contained HTML report: one policy's chain (--from) or a
    policy comparison (--compare)."""
    if (from_dir is None) == (compare_dir is None):
        raise ContractViolation(
            "report needs exactly one of --from DIR (single-policy report: calibrator.json, "
            f"policy.json, optional backtest.json) or --compare DIR ({COMPARISON_FILENAME})"
        )
    if compare_dir is not None:
        comparison_path = compare_dir / COMPARISON_FILENAME
        comparison_dict = _read_json_artifact(comparison_path, "comparison")
        html = compare_report_module.render(
            comparison_dict, created_at=artifacts.utc_now_iso(), tool_version=__version__
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html)
        typer.echo(f"included=comparison -> {output}")
        return

    assert from_dir is not None
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
    # Review-queue exports (--queue-format). Owned even when not requested, so a
    # run without export removes a previous run's queue instead of leaving it
    # next to a policy it no longer matches.
    queue_csv: Path
    queue_parquet: Path
    queue_manifest: Path

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
            queue_csv=artifacts_dir / "review_queue.csv",
            queue_parquet=artifacts_dir / "review_queue.parquet",
            queue_manifest=artifacts_dir / review_queue_module.MANIFEST_NAME,
        )

    @property
    def artifacts_dir(self) -> Path:
        return self.calibrator.parent

    def owned_files(self) -> list[Path]:
        files = [self.costs, self.calibrator, self.policy, self.backtest, self.report]
        files += [self.queue_csv, self.queue_parquet, self.queue_manifest]
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
    budget_strategy: str | None,
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
    if budget_strategy is not None:
        overrides["--budget-strategy"] = budget_strategy
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
    if budget_strategy is not None:
        raw["budget"] = {**raw["budget"], "strategy": budget_strategy}
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


_QUEUE_FORMATS = ("csv", "parquet")


def _run_queue_formats(queue_format: list[str] | None) -> set[str]:
    """Validate `run --queue-format` values; repeats are harmless."""
    formats = set(queue_format or [])
    unknown = sorted(formats - set(_QUEUE_FORMATS))
    if unknown:
        raise ContractViolation(
            f"--queue-format must be one of {', '.join(_QUEUE_FORMATS)}; got "
            f"{', '.join(repr(u) for u in unknown)}"
        )
    return formats


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
    budget_strategy: str | None = typer.Option(
        None,
        "--budget-strategy",
        help="Override budget.strategy of the cost model: 'prefix' or 'fill' (money budgets only)",
    ),
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
    queue_format: list[str] | None = typer.Option(
        None,
        "--queue-format",
        help="Export the review queue from the plan stage as 'csv' or 'parquet' (repeatable) "
        "to <run>/artifacts/review_queue.<format>, plus review_queue.json. No export by "
        "default.",
    ),
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

    formats = _run_queue_formats(queue_format)
    plan_queue_csv = layout.queue_csv if "csv" in formats else None
    plan_queue_parquet = layout.queue_parquet if "parquet" in formats else None
    if scenario is None:
        assert batch is not None  # guaranteed by the mode checks above
        plan_batch = batch
    else:
        assert layout.data_dir is not None
        plan_batch = layout.data_dir / "batch.parquet"  # what the synth stage writes
    # Same preflight, same arguments as the plan stage below receives.
    _plan_preflight(
        batch=plan_batch,
        calibrator=layout.calibrator,
        costs=layout.costs,
        output=layout.policy,
        queue_csv=plan_queue_csv,
        queue_parquet=plan_queue_parquet,
    )

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
        budget_strategy=budget_strategy,
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
        budget_strategy=None,  # likewise
        output=layout.policy,
        queue_csv=plan_queue_csv,
        queue_parquet=plan_queue_parquet,
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
    stage("report", report, from_dir=layout.artifacts_dir, compare_dir=None, output=layout.report)

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
    - CandidateRegression -> 6
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
    except CandidateRegression as e:
        typer.echo(f"Error: {e}", err=True)
        sys.exit(6)
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

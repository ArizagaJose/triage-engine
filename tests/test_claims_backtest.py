"""Backtest + drift claim-verification invariant tests, driven through the
real CLI.

Implements docs/test-spec-invariants.md, section F (Stage 5). Setup
(session-scoped, one chain per scenario, shared by all tests in this
module): for each of `well_calibrated`, `overconfident`, `drift` --
`synth --seed 42`, `calibrate --seed 42` on golden, `plan` on batch with the
emitted costs.yaml (budget fraction 0.10 from YAML), `backtest --seed 42
--n-random-draws 200` with `--history` = the scenario's own batch.parquet.

Per the spec's "Rules for the implementing agent": no assertion is weakened
and no tolerance is widened. Where a bound in the spec was measured only for
the batch-level quantities (not verified by the auditor against the
golden-level realized rates at n_golden=2000), this file still asserts the
literal bound the spec states and reports the outcome rather than loosening
it.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

TRIAGE_BIN = shutil.which("triage") or str(Path(__file__).resolve().parents[1] / ".venv/bin/triage")

_SCENARIOS = ("well_calibrated", "overconfident", "drift")

# Analytic means of the per-segment Uniform(p_lo, p_hi) true-error-rate
# distributions, per spec C.5 / interfaces.md section 6. Reused here for
# spec F.5's "golden rates still match the UNdrifted means" check.
_ANALYTIC_MEAN_P = {
    "seg_high": 0.155,  # Uniform(0.01, 0.30)
    "seg_mid": 0.21,  # Uniform(0.02, 0.40)
    "seg_low": 0.275,  # Uniform(0.05, 0.50)
}

# drift_factor is fixed at 2.0 for the "drift" scenario (interfaces.md
# section 6 / synth.py); target drifted means per spec F.5.
_DRIFT_FACTOR = 2.0
_DRIFT_TARGET_MEAN_P = {
    seg: min(_DRIFT_FACTOR * mean, 0.95) for seg, mean in _ANALYTIC_MEAN_P.items()
}


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the real `triage` console script and capture output."""
    return subprocess.run(
        [TRIAGE_BIN, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def claims_backtest_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("claims_backtest_session")


@pytest.fixture(scope="module")
def chains(claims_backtest_session: Path) -> dict[str, dict[str, Any]]:
    """Run the full synth -> calibrate -> plan -> backtest chain (seed 42)
    once per scenario and cache the parsed artifacts, since each CLI
    invocation is expensive and every test in this module reads from these
    chains."""
    result: dict[str, dict[str, Any]] = {}
    for scenario in _SCENARIOS:
        scen_dir = claims_backtest_session / scenario
        data_dir = scen_dir / "data"

        synth_result = run_cli(
            ["synth", "--scenario", scenario, "--seed", "42", "-o", str(data_dir)]
        )
        assert synth_result.returncode == 0, synth_result.stderr

        calibrator_path = scen_dir / "calibrator.json"
        calibrate_result = run_cli(
            [
                "calibrate",
                "--golden",
                str(data_dir / "golden.parquet"),
                "--seed",
                "42",
                "-o",
                str(calibrator_path),
            ]
        )
        assert calibrate_result.returncode == 0, calibrate_result.stderr

        policy_path = scen_dir / "policy.json"
        plan_result = run_cli(
            [
                "plan",
                "--batch",
                str(data_dir / "batch.parquet"),
                "--calibrator",
                str(calibrator_path),
                "--costs",
                str(data_dir / "costs.yaml"),
                "-o",
                str(policy_path),
            ]
        )
        assert plan_result.returncode == 0, plan_result.stderr

        backtest_path = scen_dir / "backtest.json"
        backtest_result = run_cli(
            [
                "backtest",
                "--policy",
                str(policy_path),
                "--calibrator",
                str(calibrator_path),
                "--history",
                str(data_dir / "batch.parquet"),
                "--seed",
                "42",
                "--n-random-draws",
                "200",
                "-o",
                str(backtest_path),
            ]
        )
        assert backtest_result.returncode == 0, backtest_result.stderr

        result[scenario] = {
            "data_dir": data_dir,
            "calibrator_path": calibrator_path,
            "policy_path": policy_path,
            "policy_dict": json.loads(policy_path.read_text()),
            "backtest_path": backtest_path,
            "backtest_dict": json.loads(backtest_path.read_text()),
            "manifest": json.loads((data_dir / "manifest.json").read_text()),
        }
    return result


# --- F.1: threshold replay is faithful ---


def test_well_calibrated_replay_exact(chains: dict[str, dict[str, Any]]) -> None:
    """Spec F.1: for well_calibrated (platt selected),
    strategies.policy.n_review == policy totals.n_review exactly (measured
    1000 == 1000)."""
    chain = chains["well_calibrated"]
    replayed = chain["backtest_dict"]["strategies"]["policy"]["n_review"]
    planned = chain["policy_dict"]["totals"]["n_review"]
    assert replayed == planned


def test_overconfident_replay_tie_inflation_bounded(chains: dict[str, dict[str, Any]]) -> None:
    """Spec F.1: for overconfident (isotonic selected), tie inflation is
    expected: planned <= replayed <= 1.3 * planned (measured 1183 vs 1000
    planned)."""
    chain = chains["overconfident"]
    replayed = chain["backtest_dict"]["strategies"]["policy"]["n_review"]
    planned = chain["policy_dict"]["totals"]["n_review"]
    assert planned <= replayed <= 1.3 * planned


# --- F.2: policy beats baselines at equal capacity (overconfident) ---


def test_policy_beats_raw_confidence_with_margin(chains: dict[str, dict[str, Any]]) -> None:
    """Spec F.2: strategies.policy.captured_loss >=
    1.15 * strategies.raw_confidence.captured_loss (measured ratio ~17.9x).
    If this margin fails empirically, report the actual number -- do not
    lower the threshold."""
    strategies = chains["overconfident"]["backtest_dict"]["strategies"]
    policy_captured = strategies["policy"]["captured_loss"]
    raw_captured = strategies["raw_confidence"]["captured_loss"]
    assert policy_captured >= 1.15 * raw_captured, (
        f"margin check failed: policy_captured={policy_captured}, "
        f"raw_captured={raw_captured}, ratio={policy_captured / raw_captured}"
    )


def test_policy_beats_random_baseline(chains: dict[str, dict[str, Any]]) -> None:
    """Spec F.2: strategies.policy.captured_loss >
    strategies.random.captured_loss_mean (measured 144500 vs 34170)."""
    strategies = chains["overconfident"]["backtest_dict"]["strategies"]
    assert strategies["policy"]["captured_loss"] > strategies["random"]["captured_loss_mean"]


def test_policy_capture_rate_floor(chains: dict[str, dict[str, Any]]) -> None:
    """Spec F.2: strategies.policy.capture_rate >= 0.40 (measured 0.5029)."""
    strategies = chains["overconfident"]["backtest_dict"]["strategies"]
    assert strategies["policy"]["capture_rate"] >= 0.40


# --- F.3: random baseline sanity ---


@pytest.mark.parametrize("scenario", ["well_calibrated", "overconfident"])
def test_random_baseline_capture_rate_matches_review_fraction(
    chains: dict[str, dict[str, Any]], scenario: str
) -> None:
    """Spec F.3: strategies.random.capture_rate within +/-0.03 of
    n_review / n_history (uniform sampling captures ~ the review fraction;
    measured 0.1189 vs 0.1183 for overconfident, 0.1004 vs 0.1000 for
    well_calibrated)."""
    backtest_dict = chains[scenario]["backtest_dict"]
    random_strategy = backtest_dict["strategies"]["random"]
    n_history = backtest_dict["n_history"]
    review_fraction = random_strategy["n_review"] / n_history
    assert abs(random_strategy["capture_rate"] - review_fraction) <= 0.03


# --- F.4: drift is exposed ---


def test_drift_residual_ratio_elevated(chains: dict[str, dict[str, Any]]) -> None:
    """Spec F.4: for drift, strategies.policy.residual_loss /
    expected.expected_residual_loss >= 1.8 (measured 2.355 -- the injected
    drift_factor is 2.0)."""
    backtest_dict = chains["drift"]["backtest_dict"]
    ratio = (
        backtest_dict["strategies"]["policy"]["residual_loss"]
        / backtest_dict["expected"]["expected_residual_loss"]
    )
    assert ratio >= 1.8


@pytest.mark.parametrize("scenario", ["well_calibrated", "overconfident"])
def test_undrifted_residual_ratio_in_control_band(
    chains: dict[str, dict[str, Any]], scenario: str
) -> None:
    """Spec F.4 controls: the same residual ratio for well_calibrated and
    overconfident lies in [0.7, 1.4] (measured 1.131 and 0.984)."""
    backtest_dict = chains[scenario]["backtest_dict"]
    ratio = (
        backtest_dict["strategies"]["policy"]["residual_loss"]
        / backtest_dict["expected"]["expected_residual_loss"]
    )
    assert 0.7 <= ratio <= 1.4


# --- F.5: drift injection matches analytic ---


@pytest.mark.parametrize("segment", ["seg_high", "seg_mid", "seg_low"])
def test_drift_realized_batch_error_matches_analytic(
    chains: dict[str, dict[str, Any]], segment: str
) -> None:
    """Spec F.5: per segment, manifest realized_mean_error_batch within
    +/-0.03 of min(drift_factor * mean(Uniform bounds), 0.95): seg_high
    0.31, seg_mid 0.42, seg_low 0.55 (measured 0.297, 0.418, 0.555)."""
    manifest = chains["drift"]["manifest"]
    realized = manifest["segments"][segment]["realized_mean_error_batch"]
    target = _DRIFT_TARGET_MEAN_P[segment]
    assert abs(realized - target) <= 0.03


@pytest.mark.parametrize("segment", ["seg_high", "seg_mid", "seg_low"])
def test_drift_golden_rates_still_undrifted(
    chains: dict[str, dict[str, Any]], segment: str
) -> None:
    """Spec F.5: golden rates still match the UNdrifted means of spec C.5
    (golden is pre-drift): per segment, manifest realized_mean_error_golden
    within +/-0.045 of the analytic Uniform mean. Golden-tier tolerance per
    the auditor amendment of 2026-07-12: n_golden=2000 (~667 rows/segment)
    has sd of the realized mean ~= 0.014, so C.5's +/-0.03 (validated only
    at n_batch=10000) is ~2.1 sigma and too tight; the measured seed-42
    seg_high diff is 0.0332. Drift only touches the batch split, never
    golden."""
    manifest = chains["drift"]["manifest"]
    realized = manifest["segments"][segment]["realized_mean_error_golden"]
    analytic_mean = _ANALYTIC_MEAN_P[segment]
    assert abs(realized - analytic_mean) <= 0.045, (
        f"golden realized_mean_error_golden for {segment!r} = {realized}, "
        f"analytic mean = {analytic_mean}, diff = {abs(realized - analytic_mean)}"
    )


# --- F.6: accounting closes ---


def test_captured_plus_residual_equals_total_for_every_strategy(
    chains: dict[str, dict[str, Any]],
) -> None:
    """Spec F.6: for every strategy, captured + residual == total_true_loss
    exactly (residual is defined as the difference, per interfaces.md
    section 9.9 -- assert equality, it documents the convention)."""
    for scenario in _SCENARIOS:
        backtest_dict = chains[scenario]["backtest_dict"]
        total_true_loss = backtest_dict["total_true_loss"]
        strategies = backtest_dict["strategies"]

        policy = strategies["policy"]
        assert policy["captured_loss"] + policy["residual_loss"] == total_true_loss

        raw_confidence = strategies["raw_confidence"]
        assert raw_confidence["captured_loss"] + raw_confidence["residual_loss"] == total_true_loss

        random_strategy = strategies["random"]
        assert (
            random_strategy["captured_loss_mean"] + random_strategy["residual_loss_mean"]
            == total_true_loss
        )


def test_policy_captured_loss_is_sequential_segment_sum(
    chains: dict[str, dict[str, Any]],
) -> None:
    """Spec F.6: strategies.policy.captured_loss equals the sequential sum
    of its per-segment captured_loss in sorted segment order exactly."""
    for scenario in _SCENARIOS:
        policy = chains[scenario]["backtest_dict"]["strategies"]["policy"]
        seg_names = sorted(policy["segments"])
        sequential_sum = 0.0
        for seg in seg_names:
            sequential_sum += policy["segments"][seg]["captured_loss"]
        assert sequential_sum == policy["captured_loss"]


def test_total_true_loss_matches_independent_fsum(
    chains: dict[str, dict[str, Any]],
) -> None:
    """Spec F.6: total_true_loss equals an independent in-test fsum of
    (predicted != true) * false_accept_cost over the history."""
    for scenario in _SCENARIOS:
        chain = chains[scenario]
        data_dir = chain["data_dir"]
        with (data_dir / "costs.yaml").open() as f:
            costs_yaml = yaml.safe_load(f)
        seg_costs = costs_yaml["segments"]
        default_costs = seg_costs.get("default")

        import polars as pl

        history_df = pl.read_parquet(data_dir / "batch.parquet")
        predicted = history_df["predicted_value"].to_list()
        true = history_df["true_value"].to_list()
        segments = history_df["segment"].to_list()

        losses = []
        for pred, tru, seg in zip(predicted, true, segments):
            if pred == tru:
                continue
            seg_cfg = seg_costs.get(seg, default_costs)
            assert seg_cfg is not None, f"no cost config for segment {seg!r} and no default"
            losses.append(seg_cfg["false_accept_cost"])
        recomputed_total = math.fsum(losses)

        reported_total = chain["backtest_dict"]["total_true_loss"]
        assert math.isclose(reported_total, recomputed_total, rel_tol=1e-9)


def test_n_errors_matches_independent_count(chains: dict[str, dict[str, Any]]) -> None:
    """Spec F.6: n_errors equals the in-test error count."""
    import polars as pl

    for scenario in _SCENARIOS:
        chain = chains[scenario]
        history_df = pl.read_parquet(chain["data_dir"] / "batch.parquet")
        predicted = history_df["predicted_value"].to_list()
        true = history_df["true_value"].to_list()
        recomputed_n_errors = sum(1 for p, t in zip(predicted, true) if p != t)
        assert chain["backtest_dict"]["n_errors"] == recomputed_n_errors


# --- F.7: drift CI gate story ---


def _write_gated_costs(data_dir: Path, dest: Path, *, max_residual_risk: float) -> Path:
    """Write a costs.yaml identical to the emitted one plus
    gates.max_residual_risk, keeping budget/other gates identical."""
    with (data_dir / "costs.yaml").open() as f:
        costs_dict = yaml.safe_load(f)
    costs_dict.setdefault("gates", {})["max_residual_risk"] = max_residual_risk
    dest.write_text(yaml.safe_dump(costs_dict))
    return dest


def test_drift_gate_story(chains: dict[str, dict[str, Any]], claims_backtest_session: Path) -> None:
    """Spec F.7: write a costs.yaml identical to the emitted one plus
    gates.max_residual_risk: 200000. On the drift chain: plan exits 0
    (expected residual ~142706 <= 200000) but backtest exits 2 (realized
    residual ~336100 > 200000), and backtest.json EXISTS afterward
    (artifact-first rule)."""
    chain = chains["drift"]
    data_dir = chain["data_dir"]
    gate_dir = claims_backtest_session / "gate_drift"
    gate_dir.mkdir(parents=True, exist_ok=True)

    gated_costs = _write_gated_costs(
        data_dir, gate_dir / "costs_gated.yaml", max_residual_risk=200000.0
    )

    gated_policy_path = gate_dir / "policy.json"
    plan_result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(chain["calibrator_path"]),
            "--costs",
            str(gated_costs),
            "-o",
            str(gated_policy_path),
        ]
    )
    assert plan_result.returncode == 0, plan_result.stderr

    gated_backtest_path = gate_dir / "backtest.json"
    backtest_result = run_cli(
        [
            "backtest",
            "--policy",
            str(gated_policy_path),
            "--calibrator",
            str(chain["calibrator_path"]),
            "--history",
            str(data_dir / "batch.parquet"),
            "--seed",
            "42",
            "-o",
            str(gated_backtest_path),
        ]
    )
    assert backtest_result.returncode == 2, backtest_result.stderr
    assert gated_backtest_path.exists(), (
        "artifact-first rule: backtest.json must exist even on gate failure"
    )


def test_well_calibrated_gate_story_both_pass(
    chains: dict[str, dict[str, Any]], claims_backtest_session: Path
) -> None:
    """Spec F.7: on the well_calibrated chain with the same gate
    (max_residual_risk: 200000), both plan and backtest exit 0 (realized
    residual ~161350 <= 200000)."""
    chain = chains["well_calibrated"]
    data_dir = chain["data_dir"]
    gate_dir = claims_backtest_session / "gate_well_calibrated"
    gate_dir.mkdir(parents=True, exist_ok=True)

    gated_costs = _write_gated_costs(
        data_dir, gate_dir / "costs_gated.yaml", max_residual_risk=200000.0
    )

    gated_policy_path = gate_dir / "policy.json"
    plan_result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(chain["calibrator_path"]),
            "--costs",
            str(gated_costs),
            "-o",
            str(gated_policy_path),
        ]
    )
    assert plan_result.returncode == 0, plan_result.stderr

    gated_backtest_path = gate_dir / "backtest.json"
    backtest_result = run_cli(
        [
            "backtest",
            "--policy",
            str(gated_policy_path),
            "--calibrator",
            str(chain["calibrator_path"]),
            "--history",
            str(data_dir / "batch.parquet"),
            "--seed",
            "42",
            "-o",
            str(gated_backtest_path),
        ]
    )
    assert backtest_result.returncode == 0, backtest_result.stderr


# --- F.8: history without labels ---


def test_backtest_history_missing_true_value_exits_1(
    chains: dict[str, dict[str, Any]], claims_backtest_session: Path
) -> None:
    """Spec F.8: history without labels -> backtest exits 1 (drop
    true_value from the history parquet)."""
    import polars as pl

    chain = chains["overconfident"]
    data_dir = chain["data_dir"]

    unlabeled_dir = claims_backtest_session / "unlabeled"
    unlabeled_dir.mkdir(parents=True, exist_ok=True)
    unlabeled_history = unlabeled_dir / "history_no_truth.parquet"

    history_df = pl.read_parquet(data_dir / "batch.parquet")
    history_df.drop("true_value").write_parquet(unlabeled_history)

    out_path = unlabeled_dir / "backtest.json"
    result = run_cli(
        [
            "backtest",
            "--policy",
            str(chain["policy_path"]),
            "--calibrator",
            str(chain["calibrator_path"]),
            "--history",
            str(unlabeled_history),
            "--seed",
            "42",
            "-o",
            str(out_path),
        ]
    )
    assert result.returncode == 1, result.stderr

"""Capacity-sweep curves claim-verification invariant tests, driven through
the real CLI.

Implements docs/test-spec-invariants.md, section G items 1-4 (Stage 6).
Imitates the tests/test_claims_backtest.py idiom: module-scoped fixture that
runs the full synth -> calibrate -> plan -> backtest chain once per scenario
(seed 42, budget 0.10 from the emitted costs.yaml, backtest --seed 42
--n-random-draws 200, history = the scenario's own batch.parquet) and shares
the parsed artifacts across all tests in this module.

Per the spec's "Rules for the implementing agent": no assertion is weakened
and no tolerance is widened. If a claim fails empirically the test stays
failing and the discrepancy is reported rather than the bound being loosened.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from triage import costs as costs_module
from triage.calibration import Calibrator

TRIAGE_BIN = shutil.which("triage") or str(Path(__file__).resolve().parents[1] / ".venv/bin/triage")

_SCENARIOS = ("well_calibrated", "overconfident", "drift")


def run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the real `triage` console script and capture output."""
    return subprocess.run(
        [TRIAGE_BIN, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture(scope="module")
def claims_curves_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("claims_curves_session")


@pytest.fixture(scope="module")
def chains(claims_curves_session: Path) -> dict[str, dict[str, Any]]:
    """Run the full synth -> calibrate -> plan -> backtest chain (seed 42)
    once per scenario and cache the parsed artifacts and paths needed for
    independent recomputation."""
    result: dict[str, dict[str, Any]] = {}
    for scenario in _SCENARIOS:
        scen_dir = claims_curves_session / scenario
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
        }
    return result


# --- G.1: curves schema ---


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_curves_grid_exact(chains: dict[str, dict[str, Any]], scenario: str) -> None:
    """Spec G.1: grid == [i/100 for i in range(51)] exactly."""
    curves = chains[scenario]["backtest_dict"]["curves"]
    assert curves["grid"] == [i / 100 for i in range(51)]


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_curves_series_have_51_entries(chains: dict[str, dict[str, Any]], scenario: str) -> None:
    """Spec G.1: the three series each have 51 entries."""
    curves = chains[scenario]["backtest_dict"]["curves"]
    assert len(curves["policy"]) == 51
    assert len(curves["raw_confidence"]) == 51
    assert len(curves["random"]) == 51


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_curves_random_exact(chains: dict[str, dict[str, Any]], scenario: str) -> None:
    """Spec G.1: random == [floor(f*n)/n for f in grid] exactly."""
    backtest_dict = chains[scenario]["backtest_dict"]
    curves = backtest_dict["curves"]
    n = backtest_dict["n_history"]
    expected = [math.floor(f * n) / n for f in curves["grid"]]
    assert curves["random"] == expected


@pytest.mark.parametrize("scenario", _SCENARIOS)
def test_curves_policy_and_raw_start_at_zero(
    chains: dict[str, dict[str, Any]], scenario: str
) -> None:
    """Spec G.1: policy[0] == raw_confidence[0] == 0.0."""
    curves = chains[scenario]["backtest_dict"]["curves"]
    assert curves["policy"][0] == 0.0
    assert curves["raw_confidence"][0] == 0.0


@pytest.mark.parametrize("scenario", _SCENARIOS)
@pytest.mark.parametrize("series", ["policy", "raw_confidence", "random"])
def test_curves_monotone_and_bounded(
    chains: dict[str, dict[str, Any]], scenario: str, series: str
) -> None:
    """Spec G.1: every series is monotone non-decreasing and bounded by 1.0."""
    values = chains[scenario]["backtest_dict"]["curves"][series]
    assert all(v >= 0.0 for v in values)
    assert all(v <= 1.0 for v in values)
    for prev, cur in zip(values, values[1:]):
        assert cur >= prev, f"{series} not monotone non-decreasing: {prev} -> {cur}"


# --- G.2: headline reproducibility (overconfident chain) ---


def test_headline_policy_at_f10(chains: dict[str, dict[str, Any]]) -> None:
    """Spec G.2: curves.policy at f=0.10 in [0.42, 0.46] (measured 0.4420)."""
    curves = chains["overconfident"]["backtest_dict"]["curves"]
    value = curves["policy"][10]
    assert 0.42 <= value <= 0.46, f"curves.policy[10]={value}"


def test_headline_raw_confidence_at_f10(chains: dict[str, dict[str, Any]]) -> None:
    """Spec G.2: curves.raw_confidence at f=0.10 in [0.015, 0.035]
    (measured 0.0236)."""
    curves = chains["overconfident"]["backtest_dict"]["curves"]
    value = curves["raw_confidence"][10]
    assert 0.015 <= value <= 0.035, f"curves.raw_confidence[10]={value}"


def test_headline_policy_beats_raw_confidence_10x(chains: dict[str, dict[str, Any]]) -> None:
    """Spec G.2: curves.policy at f=0.10 >= 10 * curves.raw_confidence at
    f=0.10 (measured ~18.7x)."""
    curves = chains["overconfident"]["backtest_dict"]["curves"]
    policy_val = curves["policy"][10]
    raw_val = curves["raw_confidence"][10]
    assert policy_val >= 10 * raw_val, (
        f"policy[10]={policy_val}, raw_confidence[10]={raw_val}, "
        f"ratio={policy_val / raw_val if raw_val else float('inf')}"
    )


# --- G.3: independent recomputation of curves.policy[10] ---


def test_curves_policy_f10_independent_recomputation(
    chains: dict[str, dict[str, Any]],
) -> None:
    """Spec G.3: for the overconfident chain, recompute curves.policy[10]
    in-test from batch.parquet + the calibrator artifact + costs.yaml
    (net_benefit ranking, top-1000, fsum of true loss / total) and assert
    exact equality with the artifact value (float-identical)."""
    chain = chains["overconfident"]
    data_dir = chain["data_dir"]

    batch = pl.read_parquet(data_dir / "batch.parquet")
    item_ids = batch["item_id"].to_list()
    segments = batch["segment"].to_list()
    scores = batch["score"].to_numpy()
    predicted = batch["predicted_value"].to_list()
    truth = batch["true_value"].to_list()
    n_history = len(item_ids)

    calibrator_dict = json.loads(chain["calibrator_path"].read_text())
    calibrator = Calibrator.from_dict(calibrator_dict)
    p_error = calibrator.predict(scores)

    cost_model = costs_module.load(data_dir / "costs.yaml")

    import numpy as np

    false_accept_costs = np.empty(n_history, dtype=np.float64)
    review_costs = np.empty(n_history, dtype=np.float64)
    for i, seg in enumerate(segments):
        seg_costs = cost_model.costs_for(seg)
        false_accept_costs[i] = seg_costs.false_accept_cost
        review_costs[i] = seg_costs.review_cost

    is_error = np.array(
        [1.0 if pv != tv else 0.0 for pv, tv in zip(predicted, truth)], dtype=np.float64
    )
    true_loss = is_error * false_accept_costs
    total_true_loss = math.fsum(true_loss.tolist())

    net_benefit = np.asarray(p_error, dtype=np.float64) * false_accept_costs - review_costs

    order = sorted(range(n_history), key=lambda i: item_ids[i])
    order.sort(key=lambda i: -net_benefit[i])

    k = 1000
    top_k = order[:k]
    recomputed = math.fsum(float(true_loss[i]) for i in top_k) / total_true_loss

    artifact_value = chain["backtest_dict"]["curves"]["policy"][10]
    assert recomputed == artifact_value, (
        f"recomputed curves.policy[10]={recomputed!r} vs artifact value={artifact_value!r}"
    )


# --- G.4: divergence is documented behavior ---


def test_capture_rate_at_or_above_curve_value(chains: dict[str, dict[str, Any]]) -> None:
    """Spec G.4: overconfident strategies.policy.capture_rate (measured
    0.5029) minus curves.policy[10] (0.4420) is positive (tie inflation
    reviews more); assert capture_rate >= curves.policy[10]."""
    backtest_dict = chains["overconfident"]["backtest_dict"]
    capture_rate = backtest_dict["strategies"]["policy"]["capture_rate"]
    curve_value = backtest_dict["curves"]["policy"][10]
    assert capture_rate >= curve_value, (
        f"capture_rate={capture_rate}, curves.policy[10]={curve_value}"
    )

"""Claim-verification invariant tests, driven through the real CLI.

Implements docs/test-spec-invariants.md, section C. Setup (session-scoped,
single chain, shared by all tests in this module): `synth overconfident
--seed 42`, `calibrate` on golden (seed 42), `plan` on batch with the emitted
costs.yaml (budget fraction 0.10). Downstream claims are recomputed
independently in-test from the raw artifacts (batch.parquet, calibrator.json,
policy.json, manifest.json) rather than trusted from the CLI's own numbers,
per the spec's "computed in-test" requirement for claim 3.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from triage.calibration import Calibrator
from triage.costs import CostModel
from triage.costs import load as load_costs

TRIAGE_BIN = shutil.which("triage") or str(Path(__file__).resolve().parents[1] / ".venv/bin/triage")

# Documented fixed seed for the 100 random-selection draws in claim C.3(a).
_RANDOM_BASELINE_SEED = 20260712
_N_RANDOM_DRAWS = 100

# Analytic means of the per-segment Uniform(p_lo, p_hi) true-error-rate
# distributions, per spec C.5 / interfaces.md section 6.
_ANALYTIC_MEAN_P = {
    "seg_high": 0.155,  # Uniform(0.01, 0.30)
    "seg_mid": 0.21,  # Uniform(0.02, 0.40)
    "seg_low": 0.275,  # Uniform(0.05, 0.50)
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
def claims_session(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("claims_session")


@pytest.fixture(scope="module")
def data_dir(claims_session: Path) -> Path:
    """`synth --scenario overconfident --seed 42` into a tmp dir."""
    out_dir = claims_session / "data"
    result = run_cli(["synth", "--scenario", "overconfident", "--seed", "42", "-o", str(out_dir)])
    assert result.returncode == 0, result.stderr
    return out_dir


@pytest.fixture(scope="module")
def manifest(data_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads((data_dir / "manifest.json").read_text())
    return result


@pytest.fixture(scope="module")
def calibrator_path(claims_session: Path, data_dir: Path) -> Path:
    """`calibrate` on golden.parquet, seed 42."""
    out_path = claims_session / "calibrator.json"
    result = run_cli(
        [
            "calibrate",
            "--golden",
            str(data_dir / "golden.parquet"),
            "--seed",
            "42",
            "-o",
            str(out_path),
        ]
    )
    assert result.returncode == 0, result.stderr
    return out_path


@pytest.fixture(scope="module")
def calibrator_dict(calibrator_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(calibrator_path.read_text())
    return result


@pytest.fixture(scope="module")
def calibrator_obj(calibrator_dict: dict[str, Any]) -> Calibrator:
    """The calibrator, deserialized independently from the artifact JSON
    (sklearn-free predict path), per spec C.4."""
    return Calibrator.from_dict(calibrator_dict)


@pytest.fixture(scope="module")
def policy_path(claims_session: Path, data_dir: Path, calibrator_path: Path) -> Path:
    """`plan` on batch.parquet with the emitted costs.yaml (budget fraction 0.10)."""
    out_path = claims_session / "policy.json"
    result = run_cli(
        [
            "plan",
            "--batch",
            str(data_dir / "batch.parquet"),
            "--calibrator",
            str(calibrator_path),
            "--costs",
            str(data_dir / "costs.yaml"),
            "-o",
            str(out_path),
        ]
    )
    assert result.returncode == 0, result.stderr
    return out_path


@pytest.fixture(scope="module")
def policy_dict(policy_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(policy_path.read_text())
    return result


@pytest.fixture(scope="module")
def cost_model(data_dir: Path) -> CostModel:
    return load_costs(data_dir / "costs.yaml")


@pytest.fixture(scope="module")
def batch_df(data_dir: Path) -> pl.DataFrame:
    return pl.read_parquet(data_dir / "batch.parquet")


@pytest.fixture(scope="module")
def batch_computed(
    batch_df: pl.DataFrame, calibrator_obj: Calibrator, cost_model: CostModel
) -> dict[str, np.ndarray]:
    """Per-item arrays recomputed independently in-test: p_error, segment,
    false_accept_cost, review_cost, true_error, item_id -- all aligned by row
    position."""
    item_ids = batch_df["item_id"].to_list()
    segments = batch_df["segment"].to_list()
    scores = batch_df["score"].to_numpy()
    predicted_value = batch_df["predicted_value"].to_list()
    true_value = batch_df["true_value"].to_list()

    p_error = np.asarray(calibrator_obj.predict(scores), dtype=np.float64)

    false_accept_cost = np.empty(len(item_ids), dtype=np.float64)
    review_cost = np.empty(len(item_ids), dtype=np.float64)
    for i, seg in enumerate(segments):
        seg_costs = cost_model.costs_for(seg)
        false_accept_cost[i] = seg_costs.false_accept_cost
        review_cost[i] = seg_costs.review_cost

    true_error = np.array([pv != tv for pv, tv in zip(predicted_value, true_value)], dtype=bool)

    return {
        "item_id": np.asarray(item_ids),
        "segment": np.asarray(segments),
        "score": scores,
        "p_error": p_error,
        "false_accept_cost": false_accept_cost,
        "review_cost": review_cost,
        "true_error": true_error,
    }


@pytest.fixture(scope="module")
def routed_mask(
    batch_computed: dict[str, np.ndarray], cost_model: CostModel, policy_dict: dict[str, Any]
) -> np.ndarray:
    """Reconstruct the policy's exact review set by replicating the pinned
    greedy allocation algorithm from interfaces.md section 5, points 1-6
    (sort by net_benefit DESC, tie-break item_id ASC; route while budget
    remains and net_benefit > 0), rather than the lossy "apply a policy
    later" threshold rule from point 7.

    The threshold rule (review iff p_error >= p_error_threshold) is NOT used
    here: isotonic calibration produces plateaus (many items sharing the
    exact same p_error), so thresholding over-includes tied items beyond
    what the tie-broken greedy selection actually routed. Verified
    empirically: for this scenario, the threshold rule over-selects
    seg_high (1183 reconstructed vs. 1000 actually routed per policy.json).
    """
    item_ids = batch_computed["item_id"]
    p_error = batch_computed["p_error"]
    false_accept_cost = batch_computed["false_accept_cost"]
    review_cost = batch_computed["review_cost"]
    n_batch = len(item_ids)

    expected_loss = p_error * false_accept_cost
    net_benefit = expected_loss - review_cost

    order = sorted(range(n_batch), key=lambda i: item_ids[i])
    order.sort(key=lambda i: -net_benefit[i])

    budget = cost_model.budget
    assert budget.type == "fraction"
    n_capacity = math.floor(budget.value * n_batch)

    mask = np.zeros(n_batch, dtype=bool)
    n_routed = 0
    for i in order:
        if n_routed >= n_capacity:
            break
        if net_benefit[i] <= 0:
            break
        mask[i] = True
        n_routed += 1

    # Self-check: reconstructed totals and per-segment counts must match
    # policy.json's own numbers -- this is the load-bearing assumption
    # behind claim C.3's "policy review set".
    assert n_routed == policy_dict["totals"]["n_review"]
    segments = batch_computed["segment"]
    for seg, seg_stats in policy_dict["segments"].items():
        seg_sel = segments == seg
        assert int((mask & seg_sel).sum()) == seg_stats["n_review"], (
            f"reconstructed routed count for segment {seg!r} does not match policy.json n_review"
        )
    return mask


def _captured_true_loss(mask: np.ndarray, batch_computed: dict[str, np.ndarray]) -> float:
    """Sum of false_accept_cost over items in `mask` that are actually
    errors (captured TRUE loss, per spec C.3)."""
    true_error = batch_computed["true_error"]
    false_accept_cost = batch_computed["false_accept_cost"]
    sel = mask & true_error
    return float(false_accept_cost[sel].sum())


# --- C.1: calibration works ---


def test_calibration_improves_ece(calibrator_dict: dict[str, Any]) -> None:
    """Spec C.1: ece_holdout (calibrated) < ece_raw_holdout (raw score)."""
    assert calibrator_dict["ece_holdout"] < calibrator_dict["ece_raw_holdout"]


def test_calibration_meets_gate(calibrator_dict: dict[str, Any]) -> None:
    """Spec C.1: ece_holdout <= 0.05 (the emitted gate value)."""
    assert calibrator_dict["ece_holdout"] <= 0.05


def test_raw_ece_is_large_for_overconfident_scenario(
    calibrator_dict: dict[str, Any],
) -> None:
    """Spec C.1: for the overconfident scenario, ece_raw_holdout >= 0.10
    (auditor measured 0.1329 at seed 42; the model under-reports error by
    ~2/3 of p)."""
    assert calibrator_dict["ece_raw_holdout"] >= 0.10


# --- C.2: budget respected ---


def test_budget_respected(policy_dict: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Spec C.2: totals.n_review <= floor(0.10 * n_batch)."""
    n_batch = manifest["n_batch"]
    cap = math.floor(0.10 * n_batch)
    assert policy_dict["totals"]["n_review"] <= cap


def test_budget_fully_used_for_this_scenario(
    policy_dict: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Spec C.2: for the overconfident scenario at seed 42, equality holds
    (expected-loss ranking at 10% budget stays well above review costs, so
    the positive-net-benefit rule does not stop the allocation early)."""
    n_batch = manifest["n_batch"]
    cap = math.floor(0.10 * n_batch)
    assert policy_dict["totals"]["n_review"] == cap


# --- C.3: policy beats naive orderings on expected loss captured at equal spend ---


def test_policy_beats_random_selection(
    batch_computed: dict[str, np.ndarray],
    routed_mask: np.ndarray,
    policy_dict: dict[str, Any],
) -> None:
    """Spec C.3(a): captured TRUE loss of the policy review set strictly
    greater than captured true loss of a random selection of the same size
    (mean over 100 seeded draws, rng seed documented as
    _RANDOM_BASELINE_SEED = 20260712)."""
    n_review = policy_dict["totals"]["n_review"]
    n_batch = len(batch_computed["item_id"])
    policy_captured = _captured_true_loss(routed_mask, batch_computed)

    rng = np.random.default_rng(_RANDOM_BASELINE_SEED)
    true_error = batch_computed["true_error"]
    false_accept_cost = batch_computed["false_accept_cost"]
    draws_captured = np.empty(_N_RANDOM_DRAWS, dtype=np.float64)
    for d in range(_N_RANDOM_DRAWS):
        idx = rng.choice(n_batch, size=n_review, replace=False)
        sel_mask = np.zeros(n_batch, dtype=bool)
        sel_mask[idx] = True
        sel = sel_mask & true_error
        draws_captured[d] = false_accept_cost[sel].sum()
    mean_random_captured = float(draws_captured.mean())

    assert policy_captured > mean_random_captured


def test_policy_beats_raw_confidence_selection(
    batch_computed: dict[str, np.ndarray],
    routed_mask: np.ndarray,
    policy_dict: dict[str, Any],
) -> None:
    """Spec C.3(b): captured TRUE loss of the policy review set strictly
    greater than captured true loss of a top-(1-score) raw-confidence
    selection of the same size, with margin policy >= 1.15 x raw (auditor
    measured 44.2% vs 2.36% of total true loss at seed 42 -- ~18x margin to
    spare). If this margin fails empirically, report the actual number; do
    not lower the threshold."""
    n_review = policy_dict["totals"]["n_review"]
    n_batch = len(batch_computed["item_id"])
    policy_captured = _captured_true_loss(routed_mask, batch_computed)

    item_ids = batch_computed["item_id"]
    scores = batch_computed["score"]
    # Rank by raw error estimate (1 - score) descending == score ascending;
    # stable tie-break on item_id ascending.
    order = sorted(range(n_batch), key=lambda i: item_ids[i])
    order.sort(key=lambda i: scores[i])
    raw_idx = np.array(order[:n_review], dtype=np.int64)
    raw_mask = np.zeros(n_batch, dtype=bool)
    raw_mask[raw_idx] = True
    raw_captured = _captured_true_loss(raw_mask, batch_computed)

    assert policy_captured > raw_captured
    assert policy_captured >= 1.15 * raw_captured, (
        f"margin check failed: policy_captured={policy_captured}, "
        f"raw_captured={raw_captured}, ratio={policy_captured / raw_captured}"
    )


# --- C.4: residual accounting closes ---


def test_residual_accounting_closes_total(
    batch_computed: dict[str, np.ndarray], policy_dict: dict[str, Any]
) -> None:
    """Spec C.4: totals.expected_captured_loss + totals.expected_residual_loss
    == sum over all items of p_error * false_accept_cost (recomputed
    independently via Calibrator.from_dict), within 1e-6 relative tolerance."""
    p_error = batch_computed["p_error"]
    false_accept_cost = batch_computed["false_accept_cost"]
    recomputed_total = float(np.sum(p_error * false_accept_cost))

    reported_total = (
        policy_dict["totals"]["expected_captured_loss"]
        + policy_dict["totals"]["expected_residual_loss"]
    )

    rel_tol = 1e-6
    assert math.isclose(reported_total, recomputed_total, rel_tol=rel_tol)


def test_per_segment_numbers_sum_to_totals(policy_dict: dict[str, Any]) -> None:
    """Spec C.4: per-segment numbers sum to totals (exact float sums,
    deterministic order: segment names sorted ascending)."""
    segments = policy_dict["segments"]
    seg_names = sorted(segments)

    captured_sum = sum(segments[seg]["expected_captured_loss"] for seg in seg_names)
    residual_sum = sum(segments[seg]["expected_residual_loss"] for seg in seg_names)
    n_review_sum = sum(segments[seg]["n_review"] for seg in seg_names)
    n_total_sum = sum(segments[seg]["n_total"] for seg in seg_names)

    assert captured_sum == policy_dict["totals"]["expected_captured_loss"]
    assert residual_sum == policy_dict["totals"]["expected_residual_loss"]
    assert n_review_sum == policy_dict["totals"]["n_review"]
    assert n_total_sum == policy_dict["totals"]["n_total"]


# --- C.5: realized error rates match injected ---


@pytest.mark.parametrize("segment", ["seg_high", "seg_mid", "seg_low"])
def test_realized_error_rate_matches_injected(manifest: dict[str, Any], segment: str) -> None:
    """Spec C.5: per segment, |realized_batch_error_rate - analytic mean of
    Uniform bounds| <= 0.03."""
    realized = manifest["segments"][segment]["realized_mean_error_batch"]
    analytic_mean = _ANALYTIC_MEAN_P[segment]
    assert abs(realized - analytic_mean) <= 0.03

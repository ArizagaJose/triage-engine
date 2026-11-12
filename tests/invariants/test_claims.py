"""Claim-verification invariants: docs/test-spec-invariants.md C (plan claims,
overconfident/42), F (backtest + drift claims) and G.1-4 (capacity-sweep
curves). Claims are recomputed in-test from the raw artifacts rather than
trusted from the CLI's own numbers. Bounds and their measured values come
from the spec; if one fails, report the number instead of loosening it."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml

from tests.support import Chain, load_json, write_costs
from triage import costs as costs_module
from triage.calibration import Calibrator
from triage.costs import CostModel
from triage.costs import load as load_costs

_SCENARIOS = ("well_calibrated", "overconfident", "drift")


_RANDOM_BASELINE_SEED = 20260712


_N_RANDOM_DRAWS = 100


_ANALYTIC_MEAN_P = {
    "seg_high": 0.155,  # Uniform(0.01, 0.30)
    "seg_mid": 0.21,  # Uniform(0.02, 0.40)
    "seg_low": 0.275,  # Uniform(0.05, 0.50)
}


_DRIFT_FACTOR = 2.0


_DRIFT_TARGET_MEAN_P = {
    seg: min(_DRIFT_FACTOR * mean, 0.95) for seg, mean in _ANALYTIC_MEAN_P.items()
}


# --- C: setup (overconfident/42: synth, calibrate, plan with emitted costs) ---


@pytest.fixture(scope="module")
def base(chain: Any) -> Chain:
    return chain("overconfident", 42)


@pytest.fixture(scope="module")
def data_dir(base: Chain) -> Path:
    return base.data_dir


@pytest.fixture(scope="module")
def manifest(base: Chain) -> dict[str, Any]:
    return base.load("manifest")


@pytest.fixture(scope="module")
def calibrator_dict(base: Chain) -> dict[str, Any]:
    return base.load("calibrator")


@pytest.fixture(scope="module")
def policy_dict(base: Chain) -> dict[str, Any]:
    return base.load("policy")


@pytest.fixture(scope="module")
def calibrator_obj(calibrator_dict: dict[str, Any]) -> Calibrator:
    """The calibrator, deserialized independently from the artifact JSON
    (sklearn-free predict path), per spec C.4."""
    return Calibrator.from_dict(calibrator_dict)


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


@pytest.mark.parametrize("segment", ["seg_high", "seg_mid", "seg_low"])
def test_realized_error_rate_matches_injected(manifest: dict[str, Any], segment: str) -> None:
    """Spec C.5: per segment, |realized_batch_error_rate - analytic mean of
    Uniform bounds| <= 0.03."""
    realized = manifest["segments"][segment]["realized_mean_error_batch"]
    analytic_mean = _ANALYTIC_MEAN_P[segment]
    assert abs(realized - analytic_mean) <= 0.03


# --- F, G: one chain per scenario (seed 42, backtest --n-random-draws 200,
#     history = the scenario's own batch.parquet) ---


@pytest.fixture(scope="module")
def chains(chain: Any) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for scenario in _SCENARIOS:
        c = chain(scenario, 42)
        result[scenario] = {
            "data_dir": c.data_dir,
            "calibrator_path": c.calibrator,
            "policy_path": c.policy,
            "policy_dict": load_json(c.policy),
            "backtest_path": c.backtest,
            "backtest_dict": load_json(c.backtest),
            "manifest": c.load("manifest"),
        }
    return result


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
    for scenario in _SCENARIOS:
        chain = chains[scenario]
        history_df = pl.read_parquet(chain["data_dir"] / "batch.parquet")
        predicted = history_df["predicted_value"].to_list()
        true = history_df["true_value"].to_list()
        recomputed_n_errors = sum(1 for p, t in zip(predicted, true) if p != t)
        assert chain["backtest_dict"]["n_errors"] == recomputed_n_errors


# --- F.7: drift CI gate story ---


def _gate_story(cli: Any, chain: dict[str, Any], gate_dir: Path) -> tuple[Any, Any, Path]:
    """plan + backtest (seed 42) with the emitted costs plus
    gates.max_residual_risk: 200000 (budget and other gates identical)."""
    data_dir = chain["data_dir"]
    gated_costs = write_costs(
        data_dir / "costs.yaml", gate_dir / "costs_gated.yaml", max_residual_risk=200000.0
    )
    gated_policy_path = gate_dir / "policy.json"
    plan_result = cli(
        "plan",
        batch=data_dir / "batch.parquet",
        calibrator=chain["calibrator_path"],
        costs=gated_costs,
        o=gated_policy_path,
    )
    gated_backtest_path = gate_dir / "backtest.json"
    backtest_result = cli(
        "backtest",
        policy=gated_policy_path,
        calibrator=chain["calibrator_path"],
        history=data_dir / "batch.parquet",
        seed=42,
        o=gated_backtest_path,
    )
    return plan_result, backtest_result, gated_backtest_path


def test_drift_gate_story(chains: dict[str, dict[str, Any]], cli: Any, tmp_path: Path) -> None:
    """Spec F.7: on the drift chain: plan exits 0 (expected residual ~142706
    <= 200000) but backtest exits 2 (realized residual ~336100 > 200000), and
    backtest.json EXISTS afterward (artifact-first rule)."""
    plan_result, backtest_result, gated_backtest_path = _gate_story(cli, chains["drift"], tmp_path)
    assert plan_result.returncode == 0, plan_result.stderr
    assert backtest_result.returncode == 2, backtest_result.stderr
    assert gated_backtest_path.exists(), (
        "artifact-first rule: backtest.json must exist even on gate failure"
    )


def test_well_calibrated_gate_story_both_pass(
    chains: dict[str, dict[str, Any]], cli: Any, tmp_path: Path
) -> None:
    """Spec F.7: on the well_calibrated chain with the same gate, both plan and
    backtest exit 0 (realized residual ~161350 <= 200000)."""
    plan_result, backtest_result, _ = _gate_story(cli, chains["well_calibrated"], tmp_path)
    assert plan_result.returncode == 0, plan_result.stderr
    assert backtest_result.returncode == 0, backtest_result.stderr


# --- F.8: history without labels ---


def test_backtest_history_missing_true_value_exits_1(
    chains: dict[str, dict[str, Any]], cli: Any, tmp_path: Path
) -> None:
    """Spec F.8: history without labels -> backtest exits 1 (drop
    true_value from the history parquet)."""
    chain = chains["overconfident"]
    unlabeled_history = tmp_path / "history_no_truth.parquet"
    pl.read_parquet(chain["data_dir"] / "batch.parquet").drop("true_value").write_parquet(
        unlabeled_history
    )
    result = cli(
        "backtest",
        policy=chain["policy_path"],
        calibrator=chain["calibrator_path"],
        history=unlabeled_history,
        seed=42,
        o=tmp_path / "backtest.json",
    )
    assert result.returncode == 1, result.stderr


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

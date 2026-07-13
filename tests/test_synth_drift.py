"""Unit tests for the drift scenario (docs/interfaces.md §6)."""

import polars as pl
import pytest

from triage.exceptions import ContractViolation
from triage.synth import generate

SEGMENT_BOUNDS = {
    "seg_high": (0.01, 0.30),
    "seg_mid": (0.02, 0.40),
    "seg_low": (0.05, 0.50),
}


def test_drift_golden_equals_well_calibrated_golden_same_seed() -> None:
    drift = generate("drift", seed=42)
    well = generate("well_calibrated", seed=42)
    drift_golden = drift["golden"]
    well_golden = well["golden"]
    assert isinstance(drift_golden, pl.DataFrame)
    assert isinstance(well_golden, pl.DataFrame)
    assert drift_golden.equals(well_golden)


def test_drift_batch_differs_from_well_calibrated_batch() -> None:
    # Same rng consumption order and same score formula, but drifted error
    # probabilities mean the error/predicted_value columns (and hence the
    # frame) diverge on the batch split.
    drift = generate("drift", seed=42)
    well = generate("well_calibrated", seed=42)
    drift_batch = drift["batch"]
    well_batch = well["batch"]
    assert isinstance(drift_batch, pl.DataFrame)
    assert isinstance(well_batch, pl.DataFrame)
    assert not drift_batch.equals(well_batch)
    # But item_id and score columns are identical (rng order + score formula unchanged).
    assert drift_batch["item_id"].to_list() == well_batch["item_id"].to_list()
    assert drift_batch["score"].to_list() == well_batch["score"].to_list()
    assert drift_batch["segment"].to_list() == well_batch["segment"].to_list()


def test_drift_batch_error_rate_approx_double_undrifted_mean() -> None:
    # Undrifted analytic means: seg_high (0.01+0.30)/2=0.155, seg_mid
    # (0.02+0.40)/2=0.21, seg_low (0.05+0.50)/2=0.275. Doubled (drift_factor=2.0,
    # capped at 0.95, which none of these hit): 0.31 / 0.42 / 0.55.
    expected_doubled = {"seg_high": 0.31, "seg_mid": 0.42, "seg_low": 0.55}

    data = generate("drift", seed=42)
    batch = data["batch"]
    assert isinstance(batch, pl.DataFrame)

    err = batch.with_columns(
        (pl.col("predicted_value") != pl.col("true_value")).cast(pl.Float64).alias("error")
    )
    per_seg = err.group_by("segment").agg(pl.col("error").mean().alias("mean_error"))
    rates = dict(zip(per_seg["segment"].to_list(), per_seg["mean_error"].to_list(), strict=True))

    for seg, expected in expected_doubled.items():
        assert abs(rates[seg] - expected) < 0.03, (seg, rates[seg], expected)


def test_manifest_has_drift_factor_2_for_drift_scenario() -> None:
    data = generate("drift", seed=42)
    manifest = data["manifest"]
    assert isinstance(manifest, dict)
    assert manifest["drift_factor"] == 2.0
    assert manifest["scenario"] == "drift"


@pytest.mark.parametrize("scenario", ["well_calibrated", "overconfident"])
def test_manifest_has_null_drift_factor_for_non_drift_scenarios(scenario: str) -> None:
    data = generate(scenario, seed=42)
    manifest = data["manifest"]
    assert isinstance(manifest, dict)
    assert "drift_factor" in manifest
    assert manifest["drift_factor"] is None


def test_unknown_scenario_still_raises_contract_violation() -> None:
    with pytest.raises(ContractViolation):
        generate("not_a_real_scenario", seed=42)

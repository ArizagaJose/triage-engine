"""Synthetic scenarios, including drift (docs/interfaces.md §6)."""

import json
from pathlib import Path

import polars as pl
import pytest

from triage import contract, costs, synth
from triage.exceptions import ContractViolation
from triage.synth import generate, write

SCENARIOS = ["well_calibrated", "overconfident"]

EXPECTED_SCHEMA = {
    "item_id": pl.String,
    "segment": pl.String,
    "score": pl.Float64,
    "predicted_value": pl.String,
    "true_value": pl.String,
}

SEGMENT_BOUNDS = {
    "seg_high": (0.01, 0.30),
    "seg_mid": (0.02, 0.40),
    "seg_low": (0.05, 0.50),
}


def _files(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir())


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_write_determinism_same_seed(tmp_path: Path, scenario: str) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    write(scenario, seed=42, out_dir=out1)
    write(scenario, seed=42, out_dir=out2)

    assert (
        _files(out1)
        == _files(out2)
        == [
            "batch.parquet",
            "costs.yaml",
            "golden.parquet",
            "manifest.json",
        ]
    )
    for name in _files(out1):
        b1 = (out1 / name).read_bytes()
        b2 = (out2 / name).read_bytes()
        assert b1 == b2, f"{name} differs between two runs with the same seed"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_write_different_seed_differs(tmp_path: Path, scenario: str) -> None:
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    write(scenario, seed=42, out_dir=out1)
    write(scenario, seed=43, out_dir=out2)

    assert (out1 / "golden.parquet").read_bytes() != (out2 / "golden.parquet").read_bytes()
    assert (out1 / "batch.parquet").read_bytes() != (out2 / "batch.parquet").read_bytes()
    # costs.yaml is scenario/seed-independent, must be identical.
    assert (out1 / "costs.yaml").read_bytes() == (out2 / "costs.yaml").read_bytes()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_schema_and_basic_shape(scenario: str) -> None:
    data = generate(scenario, seed=42)
    golden = data["golden"]
    batch = data["batch"]
    assert isinstance(golden, pl.DataFrame)
    assert isinstance(batch, pl.DataFrame)

    for df, n_expected in ((golden, 2000), (batch, 10000)):
        assert df.height == n_expected
        assert dict(df.schema) == EXPECTED_SCHEMA
        assert df["item_id"].n_unique() == n_expected
        assert df["item_id"].null_count() == 0
        assert df["segment"].null_count() == 0
        assert df["predicted_value"].null_count() == 0
        assert df["true_value"].null_count() == 0
        assert df["score"].null_count() == 0
        assert df["score"].min() >= 0.0  # type: ignore[operator]
        assert df["score"].max() <= 1.0  # type: ignore[operator]
        assert set(df["segment"].unique().to_list()) <= set(SEGMENT_BOUNDS)
        # ids are zero-padded and rows are sorted by item_id
        assert df["item_id"].to_list() == sorted(df["item_id"].to_list())


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_item_ids_prefixed(scenario: str) -> None:
    data = generate(scenario, seed=42)
    golden = data["golden"]
    batch = data["batch"]
    assert isinstance(golden, pl.DataFrame)
    assert isinstance(batch, pl.DataFrame)
    assert golden["item_id"].str.starts_with("g-").all()
    assert batch["item_id"].str.starts_with("b-").all()
    assert golden["item_id"][0] == "g-000001"
    assert batch["item_id"][0] == "b-000001"


def test_statistical_sanity_error_rate_matches_uniform_mean() -> None:
    # Use the batch split (10k rows, ~3.3k/segment) rather than golden (2k rows,
    # ~666/segment): at 2k rows the binomial + uniform-p sampling noise alone can
    # exceed a 0.03 tolerance for some seeds, which would make this a flaky check
    # of RNG luck rather than of the generator's correctness.
    data = generate("well_calibrated", seed=42)
    golden = data["batch"]
    assert isinstance(golden, pl.DataFrame)

    err = golden.with_columns(
        (pl.col("predicted_value") != pl.col("true_value")).cast(pl.Float64).alias("error")
    )
    per_seg = err.group_by("segment").agg(pl.col("error").mean().alias("mean_error"))
    rates = dict(zip(per_seg["segment"].to_list(), per_seg["mean_error"].to_list(), strict=True))

    for seg, (lo, hi) in SEGMENT_BOUNDS.items():
        analytic_mean = (lo + hi) / 2
        assert abs(rates[seg] - analytic_mean) < 0.03, (seg, rates[seg], analytic_mean)


def test_statistical_sanity_well_calibrated_score_tracks_error() -> None:
    # See note above: use batch for a tighter, non-flaky statistical check.
    data = generate("well_calibrated", seed=42)
    golden = data["batch"]
    assert isinstance(golden, pl.DataFrame)

    df = golden.with_columns(
        (pl.col("predicted_value") != pl.col("true_value")).cast(pl.Float64).alias("error"),
        (1.0 - pl.col("score")).alias("one_minus_score"),
    )
    per_seg = df.group_by("segment").agg(
        pl.col("error").mean().alias("mean_error"),
        pl.col("one_minus_score").mean().alias("mean_one_minus_score"),
    )
    for row in per_seg.iter_rows(named=True):
        assert abs(row["mean_one_minus_score"] - row["mean_error"]) < 0.03


def test_statistical_sanity_overconfident_score_is_one_third() -> None:
    # See note above: use batch for a tighter, non-flaky statistical check.
    data = generate("overconfident", seed=42)
    golden = data["batch"]
    assert isinstance(golden, pl.DataFrame)

    df = golden.with_columns(
        (pl.col("predicted_value") != pl.col("true_value")).cast(pl.Float64).alias("error"),
        (1.0 - pl.col("score")).alias("one_minus_score"),
    )
    per_seg = df.group_by("segment").agg(
        pl.col("error").mean().alias("mean_error"),
        pl.col("one_minus_score").mean().alias("mean_one_minus_score"),
    )
    for row in per_seg.iter_rows(named=True):
        expected = row["mean_error"] / 3.0
        assert abs(row["mean_one_minus_score"] - expected) < 0.02


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_manifest_contents(scenario: str) -> None:
    data = generate(scenario, seed=42)
    manifest = data["manifest"]
    assert isinstance(manifest, dict)

    assert manifest["scenario"] == scenario
    assert manifest["seed"] == 42
    assert manifest["n_golden"] == 2000
    assert manifest["n_batch"] == 10000
    assert manifest["noise_sigma"] == 0.01
    assert isinstance(manifest["segment_assignment_scheme"], str)
    assert manifest["score_formula"] in ("one_minus_p", "one_minus_p_over_3")

    for seg, (lo, hi) in SEGMENT_BOUNDS.items():
        seg_manifest = manifest["segments"][seg]
        assert seg_manifest["p_min"] == lo
        assert seg_manifest["p_max"] == hi
        assert 0.0 <= seg_manifest["realized_mean_error_golden"] <= 1.0
        assert 0.0 <= seg_manifest["realized_mean_error_batch"] <= 1.0


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_manifest_realized_rates_match_recomputed_from_parquet(
    tmp_path: Path, scenario: str
) -> None:
    out_dir = tmp_path / "out"
    write(scenario, seed=42, out_dir=out_dir)

    manifest = json.loads((out_dir / "manifest.json").read_text())
    golden = pl.read_parquet(out_dir / "golden.parquet")
    batch = pl.read_parquet(out_dir / "batch.parquet")

    def recompute(df: pl.DataFrame) -> dict[str, float]:
        err = df.with_columns(
            (pl.col("predicted_value") != pl.col("true_value")).cast(pl.Float64).alias("error")
        )
        per_seg = err.group_by("segment").agg(pl.col("error").mean().alias("mean_error"))
        return dict(zip(per_seg["segment"].to_list(), per_seg["mean_error"].to_list(), strict=True))

    golden_rates = recompute(golden)
    batch_rates = recompute(batch)

    for seg in SEGMENT_BOUNDS:
        assert manifest["segments"][seg]["realized_mean_error_golden"] == golden_rates[seg]
        assert manifest["segments"][seg]["realized_mean_error_batch"] == batch_rates[seg]


def test_manifest_is_canonical_json_sorted_keys(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    write("overconfident", seed=42, out_dir=out_dir)
    raw = (out_dir / "manifest.json").read_text()
    parsed = json.loads(raw)
    reserialized = json.dumps(parsed, sort_keys=True, indent=2, allow_nan=False) + "\n"
    assert raw == reserialized


def test_unknown_scenario_raises_contract_violation() -> None:
    with pytest.raises(ContractViolation) as exc_info:
        generate("not_a_real_scenario", seed=42)
    message = str(exc_info.value)
    assert "well_calibrated" in message
    assert "overconfident" in message
    assert "drift" in message


def test_unknown_scenario_raises_on_write(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation):
        write("bogus", seed=42, out_dir=tmp_path / "out")


@pytest.mark.parametrize("seed", [-1, -42])
def test_negative_seed_is_contract_violation(seed: int) -> None:
    """np.random.default_rng rejects negative seeds with ValueError (exit 4);
    it must surface as an input contract violation (exit 1) instead."""
    with pytest.raises(ContractViolation, match="seed must be a non-negative integer"):
        generate("drift", seed=seed)


def test_unknown_scenario_checked_before_seed() -> None:
    with pytest.raises(ContractViolation, match="Unknown scenario 'nope'"):
        generate("nope", seed=-1)


# --- drift scenario (docs/interfaces.md §6) ---


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


# --- generated data satisfies the input contract and the cost model ---


@pytest.mark.parametrize("scenario", ["well_calibrated", "overconfident"])
def test_synth_output_passes_contract(scenario: str) -> None:
    data = synth.generate(scenario, seed=42)
    for key in ("golden", "batch"):
        df = data[key]
        assert contract.validate(df, require_truth=True) is not None  # type: ignore[arg-type]


def test_emitted_costs_yaml_loads_and_files_round_trip(tmp_path: Path) -> None:
    synth.write("overconfident", seed=42, out_dir=tmp_path)
    cost_model = costs.load(tmp_path / "costs.yaml")
    assert cost_model.budget.type == "fraction"
    assert cost_model.budget.value == 0.10
    assert cost_model.gates.max_ece == 0.05
    for name in ("golden.parquet", "batch.parquet"):
        df = contract.read_input(tmp_path / name, require_truth=True)
        assert df.height > 0

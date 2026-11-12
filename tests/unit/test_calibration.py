"""Calibration: ECE, isotonic/Platt fit, bootstrap CI, ECE gate (docs/interfaces.md §4)."""

import json

import numpy as np
import polars as pl
import pytest

from triage import synth
from triage.artifacts import canonical_dumps
from triage.calibration import (
    DEFAULT_ECE_GATE_MODE,
    ECE_GATE_MODES,
    CalibrationConfig,
    Calibrator,
    check_ece_gate,
    ece,
    ece_gate_statistic,
    fit,
)
from triage.exceptions import CalibrationGateFailed, ContractViolation


def _make_golden(n: int, *, seed: int, score_fn: str = "well_calibrated") -> pl.DataFrame:
    """Build a synthetic golden frame with score exactly tracking 1 - P(error)
    (no noise), for the perfect-calibration sanity test."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.5, size=n)
    error = rng.binomial(1, p, size=n)
    score = 1.0 - p  # exact, no noise: perfectly calibrated raw score
    item_id = [f"g-{i:06d}" for i in range(n)]
    segment = np.full(n, "seg_a")
    true_value = np.full(n, "v", dtype=object)
    predicted_value = np.where(error == 1, "x", "v")
    return pl.DataFrame(
        {
            "item_id": item_id,
            "segment": segment.tolist(),
            "score": score.tolist(),
            "predicted_value": predicted_value.tolist(),
            "true_value": true_value.tolist(),
        },
        schema={
            "item_id": pl.String,
            "segment": pl.String,
            "score": pl.Float64,
            "predicted_value": pl.String,
            "true_value": pl.String,
        },
    )


# ---------------------------------------------------------------------------
# ece() unit tests — hand-computed
# ---------------------------------------------------------------------------


def test_ece_hand_computed_two_bins() -> None:
    # 4 points, 2 bins (edges at 0, 0.5, 1.0).
    # p_pred = [0.1, 0.2, 0.6, 0.9], y = [0, 1, 1, 1]
    # bin_idx = floor(p*2), clipped to [0,1]:
    #   0.1 -> floor(0.2)=0 -> bin 0
    #   0.2 -> floor(0.4)=0 -> bin 0
    #   0.6 -> floor(1.2)=1 -> bin 1
    #   0.9 -> floor(1.8)=1 -> bin 1
    # bin 0: preds=[0.1,0.2] mean_pred=0.15, y=[0,1] mean_obs=0.5
    #   contribution = |0.15-0.5| * (2/4) = 0.35 * 0.5 = 0.175
    # bin 1: preds=[0.6,0.9] mean_pred=0.75, y=[1,1] mean_obs=1.0
    #   contribution = |0.75-1.0| * (2/4) = 0.25 * 0.5 = 0.125
    # total = 0.175 + 0.125 = 0.3
    p_pred = np.array([0.1, 0.2, 0.6, 0.9])
    y = np.array([0.0, 1.0, 1.0, 1.0])
    result = ece(p_pred, y, n_bins=2)
    assert result == pytest.approx(0.3, abs=1e-12)


def test_ece_perfect_calibration_is_zero() -> None:
    p_pred = np.array([0.0, 0.0, 1.0, 1.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert ece(p_pred, y, n_bins=2) == pytest.approx(0.0, abs=1e-12)


def test_ece_boundary_value_one_maps_to_last_bin() -> None:
    # p_pred == 1.0 exactly must land in the last bin, not overflow.
    p_pred = np.array([1.0])
    y = np.array([1.0])
    result = ece(p_pred, y, n_bins=10)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_ece_empty_bins_ignored() -> None:
    # All points in one bin; other bins empty and contribute nothing.
    p_pred = np.array([0.05, 0.05])
    y = np.array([1.0, 1.0])
    # bin 0: mean_pred=0.05, mean_obs=1.0, weight=2/2=1 -> |0.05-1.0|*1 = 0.95
    result = ece(p_pred, y, n_bins=10)
    assert result == pytest.approx(0.95, abs=1e-12)


# ---------------------------------------------------------------------------
# Perfect-calibration sanity
# ---------------------------------------------------------------------------


def test_perfect_calibration_sanity() -> None:
    golden = _make_golden(6000, seed=1)
    config = CalibrationConfig(holdout_fraction=0.3, n_bins=10, n_bootstrap=200)
    cal = fit(golden, seed=1, config=config)

    assert cal.ece_holdout < cal.ece_raw_holdout + 0.01
    assert cal.ece_holdout <= 0.03


# ---------------------------------------------------------------------------
# Overconfident correction (real synth scenario)
# ---------------------------------------------------------------------------


def test_overconfident_correction() -> None:
    data = synth.generate("overconfident", seed=42)
    golden = data["golden"]
    assert isinstance(golden, pl.DataFrame)

    cal = fit(golden, seed=42)

    assert cal.ece_holdout < cal.ece_raw_holdout

    if cal.method == "isotonic":
        # Monotonic consistency: higher raw score -> lower or equal p_error.
        scores = np.linspace(0.0, 1.0, 200)
        preds = cal.predict(scores)
        diffs = np.diff(preds)
        assert np.all(diffs <= 1e-12)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_determinism_two_fits_identical() -> None:
    golden = _make_golden(2000, seed=7)
    c1 = fit(golden, seed=7)
    c2 = fit(golden, seed=7)

    assert c1.to_dict() == c2.to_dict()
    assert canonical_dumps(c1.to_dict()) == canonical_dumps(c2.to_dict())


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_predict_bit_identical() -> None:
    golden = _make_golden(2000, seed=3)
    cal = fit(golden, seed=3)
    restored = Calibrator.from_dict(cal.to_dict())

    x = np.array([0.0, 1.0, -5.0, 2.0, 0.3, 0.7, 0.999999])
    original_pred = cal.predict(x)
    restored_pred = restored.predict(x)
    assert np.array_equal(original_pred, restored_pred)


def test_round_trip_to_dict_json_safe() -> None:

    golden = _make_golden(1500, seed=11)
    cal = fit(golden, seed=11)
    d = cal.to_dict()
    # Must serialize without error (no numpy scalars/arrays left over).
    json.dumps(d, allow_nan=False)

    expected_keys = {
        "method",
        "params",
        "ece_holdout",
        "ece_ci",
        "ece_raw_holdout",
        "n_golden",
        "seed",
        "input_sha256",
        "tool_version",
        "config",
        "reliability_bins",
    }
    assert set(d.keys()) == expected_keys
    assert isinstance(d["ece_ci"], list)
    assert len(d["ece_ci"]) == 2


# ---------------------------------------------------------------------------
# check_ece_gate
# ---------------------------------------------------------------------------


def test_check_ece_gate_raises_when_above() -> None:
    golden = _make_golden(500, seed=5)
    cal = fit(golden, seed=5)
    too_strict = cal.ece_holdout - 1e-6
    if too_strict < 0:
        too_strict = 0.0
    with pytest.raises(CalibrationGateFailed) as exc_info:
        check_ece_gate(cal, max_ece=too_strict if cal.ece_holdout > 0 else -1.0)
    message = str(exc_info.value)
    assert "ece_holdout" in message


def test_check_ece_gate_does_not_raise_when_within() -> None:
    golden = _make_golden(500, seed=5)
    cal = fit(golden, seed=5)
    check_ece_gate(cal, max_ece=cal.ece_holdout + 0.5)


def test_check_ece_gate_message_contains_both_numbers() -> None:
    golden = _make_golden(500, seed=5)
    cal = fit(golden, seed=5)
    max_ece = -1.0  # guaranteed to trigger the gate
    with pytest.raises(CalibrationGateFailed) as exc_info:
        check_ece_gate(cal, max_ece=max_ece)
    message = str(exc_info.value)
    assert str(cal.ece_holdout) in message
    assert str(max_ece) in message


# ---------------------------------------------------------------------------
# check_ece_gate modes: "point" (default) vs "ci_upper"
# ---------------------------------------------------------------------------

# The exact top-level key set of a calibrator.json written before ece gate
# modes existed (interfaces.md §7). Gate modes add NO key to the calibrator
# artifact, so this is also the current key set; the handcrafted artifacts
# below are therefore "old" artifacts in every respect.
_PINNED_CALIBRATOR_KEYS = {
    "config",
    "created_at",
    "ece_ci",
    "ece_holdout",
    "ece_raw_holdout",
    "input_sha256",
    "method",
    "n_golden",
    "params",
    "reliability_bins",
    "seed",
    "tool_version",
}


def _calibrator_artifact(
    *, ece_holdout: float, ece_ci: tuple[float, float], ci_level: float = 0.95
) -> dict[str, object]:
    """A handcrafted calibrator.json dict with chosen ECE numbers, so gate
    tests do not depend on what a fit happens to produce."""
    return {
        "config": {
            "ci_level": ci_level,
            "holdout_fraction": 0.3,
            "n_bins": 10,
            "n_bootstrap": 1000,
        },
        "created_at": "2026-07-01T00:00:00+00:00",
        "ece_ci": [ece_ci[0], ece_ci[1]],
        "ece_holdout": ece_holdout,
        "ece_raw_holdout": 0.13,
        "input_sha256": "a" * 64,
        "method": "platt",
        "n_golden": 2000,
        "params": {"a": -3.0, "b": 1.0},
        "reliability_bins": {"after": [], "before": []},
        "seed": 42,
        "tool_version": "0.1.0",
    }


def _gate_calibrator(
    *, ece_holdout: float = 0.04, ece_ci: tuple[float, float] = (0.02, 0.08)
) -> Calibrator:
    return Calibrator.from_dict(_calibrator_artifact(ece_holdout=ece_holdout, ece_ci=ece_ci))


def test_ece_gate_modes_and_default() -> None:
    assert ECE_GATE_MODES == ("point", "ci_upper")
    assert DEFAULT_ECE_GATE_MODE == "point"


def test_point_passes_but_ci_upper_fails() -> None:
    """The case the mode exists for: point estimate under the threshold, the
    upper CI bound over it."""
    cal = _gate_calibrator(ece_holdout=0.04, ece_ci=(0.02, 0.08))
    check_ece_gate(cal, 0.05, "point")  # 0.04 <= 0.05
    with pytest.raises(CalibrationGateFailed) as exc_info:
        check_ece_gate(cal, 0.05, "ci_upper")  # 0.08 > 0.05
    message = str(exc_info.value)
    assert "ece_gate_mode=ci_upper" in message
    assert "ece_ci_upper=0.08" in message
    assert "max_ece=0.05" in message
    assert "95% bootstrap CI" in message
    # The point estimate is reported for context, not as the checked value.
    assert "ece_holdout=0.04" in message


def test_default_mode_is_point_estimate() -> None:
    """Omitting `mode` keeps the original semantics: only ece_holdout is
    compared, a failing upper bound is ignored."""
    cal = _gate_calibrator(ece_holdout=0.04, ece_ci=(0.02, 0.08))
    check_ece_gate(cal, 0.05)  # would fail under ci_upper
    with pytest.raises(CalibrationGateFailed) as exc_info:
        check_ece_gate(cal, 0.039)
    message = str(exc_info.value)
    assert "ece_gate_mode=point" in message
    assert "ece_holdout=0.04" in message


def test_both_modes_pass_when_upper_bound_within() -> None:
    cal = _gate_calibrator(ece_holdout=0.04, ece_ci=(0.02, 0.08))
    check_ece_gate(cal, 0.09, "point")
    check_ece_gate(cal, 0.09, "ci_upper")


def test_both_modes_fail_when_point_above() -> None:
    cal = _gate_calibrator(ece_holdout=0.04, ece_ci=(0.02, 0.08))
    for mode in ECE_GATE_MODES:
        with pytest.raises(CalibrationGateFailed):
            check_ece_gate(cal, 0.03, mode)


@pytest.mark.parametrize("mode", ECE_GATE_MODES)
def test_gate_equality_at_threshold_passes(mode: str) -> None:
    """Strict `>`: a checked value exactly equal to max_ece passes; a max_ece
    one ULP below the checked value fails."""
    cal = _gate_calibrator(ece_holdout=0.04, ece_ci=(0.02, 0.08))
    checked = cal.ece_holdout if mode == "point" else cal.ece_ci[1]
    check_ece_gate(cal, checked, mode)
    with pytest.raises(CalibrationGateFailed):
        check_ece_gate(cal, float(np.nextafter(checked, 0.0)), mode)


def test_ci_upper_checks_the_upper_bound_not_the_lower() -> None:
    cal = _gate_calibrator(ece_holdout=0.04, ece_ci=(0.02, 0.08))
    # Between lower bound and point estimate: the lower bound would pass,
    # the upper bound must not.
    with pytest.raises(CalibrationGateFailed):
        check_ece_gate(cal, 0.03, "ci_upper")


@pytest.mark.parametrize("mode", ["upper", "CI_UPPER", "", "ci_lower", "Point"])
def test_unknown_gate_mode_is_contract_violation(mode: str) -> None:
    cal = _gate_calibrator()
    with pytest.raises(ContractViolation) as exc_info:
        check_ece_gate(cal, 0.05, mode)
    message = str(exc_info.value)
    assert repr(mode) in message
    assert "point" in message and "ci_upper" in message


def test_ece_gate_statistic_reports_field_and_ci_level() -> None:
    cal = Calibrator.from_dict(
        _calibrator_artifact(ece_holdout=0.04, ece_ci=(0.02, 0.08), ci_level=0.9)
    )
    point = ece_gate_statistic(cal, "point")
    assert (point.name, point.value) == ("ece_holdout", 0.04)
    upper = ece_gate_statistic(cal, "ci_upper")
    assert (upper.name, upper.value) == ("ece_ci_upper", 0.08)
    assert "90% bootstrap CI" in upper.description
    assert "ece_ci[1]" in upper.description


def test_old_calibrator_artifact_supports_both_modes() -> None:
    """Backwards compatibility: a calibrator.json with exactly the pinned
    pre-existing key set loads and gates in both modes, and re-serializes to
    the same key set (gate modes add nothing to the calibrator artifact, so
    its content_hash -- and every policy_hash built on it -- is unaffected)."""
    artifact = _calibrator_artifact(ece_holdout=0.04, ece_ci=(0.02, 0.08))
    assert set(artifact) == _PINNED_CALIBRATOR_KEYS
    cal = Calibrator.from_dict(artifact)
    check_ece_gate(cal, 0.05, "point")
    with pytest.raises(CalibrationGateFailed):
        check_ece_gate(cal, 0.05, "ci_upper")
    assert set(cal.to_dict()) | {"created_at"} == _PINNED_CALIBRATOR_KEYS


def test_ci_upper_on_real_fit_uses_fitted_interval() -> None:
    """On a real fit, ci_upper gates exactly at the fitted ece_ci[1]."""
    golden = _make_golden(500, seed=5)
    cal = fit(golden, seed=5)
    assert cal.ece_ci[1] > cal.ece_holdout  # sanity for the next assertions
    check_ece_gate(cal, cal.ece_ci[1], "ci_upper")
    with pytest.raises(CalibrationGateFailed):
        check_ece_gate(cal, cal.ece_holdout, "ci_upper")
    check_ece_gate(cal, cal.ece_holdout, "point")


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def test_bootstrap_ci_contains_point_estimate_and_ordered() -> None:
    golden = _make_golden(3000, seed=9)
    config = CalibrationConfig(n_bootstrap=500)
    cal = fit(golden, seed=9, config=config)

    lo, hi = cal.ece_ci
    assert lo < hi
    assert lo <= cal.ece_holdout <= hi


def test_bootstrap_ci_deterministic() -> None:
    golden = _make_golden(2000, seed=13)
    config = CalibrationConfig(n_bootstrap=300)
    c1 = fit(golden, seed=13, config=config)
    c2 = fit(golden, seed=13, config=config)
    assert c1.ece_ci == c2.ece_ci


# ---------------------------------------------------------------------------
# Seed sensitivity
# ---------------------------------------------------------------------------


def test_seed_sensitivity_changes_output() -> None:
    golden = _make_golden(2000, seed=21)
    c1 = fit(golden, seed=1)
    c2 = fit(golden, seed=2)
    assert c1.to_dict() != c2.to_dict()


# ---------------------------------------------------------------------------
# Degenerate single-class calibration split (regression: Platt would raise)
# ---------------------------------------------------------------------------


def _make_constant_golden(n: int, *, all_errors: bool) -> pl.DataFrame:
    """Golden frame where every row is an error (or none is)."""
    predicted = "x" if all_errors else "v"
    return pl.DataFrame(
        {
            "item_id": [f"g-{i:04d}" for i in range(n)],
            "segment": ["s"] * n,
            "score": [0.4 + 0.002 * i for i in range(n)],
            "predicted_value": [predicted] * n,
            "true_value": ["v"] * n,
        },
        schema={
            "item_id": pl.String,
            "segment": pl.String,
            "score": pl.Float64,
            "predicted_value": pl.String,
            "true_value": pl.String,
        },
    )


def test_zero_error_golden_selects_isotonic_constant_zero() -> None:
    golden = _make_constant_golden(100, all_errors=False)
    config = CalibrationConfig(n_bootstrap=50)
    cal = fit(golden, seed=42, config=config)

    assert cal.method == "isotonic"
    preds = cal.predict(np.array([0.0, 0.3, 0.5, 0.7, 1.0]))
    assert np.allclose(preds, 0.0, atol=1e-9)

    # Serialization round-trip still works on the degenerate calibrator.
    restored = Calibrator.from_dict(cal.to_dict())
    x = np.array([0.0, 0.5, 1.0, -1.0, 2.0])
    assert np.array_equal(cal.predict(x), restored.predict(x))
    canonical_dumps(cal.to_dict())  # must not raise (JSON-safe, no NaN/inf)


def test_all_error_golden_selects_isotonic_constant_one() -> None:
    golden = _make_constant_golden(100, all_errors=True)
    config = CalibrationConfig(n_bootstrap=50)
    cal = fit(golden, seed=42, config=config)

    assert cal.method == "isotonic"
    preds = cal.predict(np.array([0.0, 0.3, 0.5, 0.7, 1.0]))
    assert np.allclose(preds, 1.0, atol=1e-9)

    restored = Calibrator.from_dict(cal.to_dict())
    x = np.array([0.0, 0.5, 1.0])
    assert np.array_equal(cal.predict(x), restored.predict(x))
    canonical_dumps(cal.to_dict())


# ---------------------------------------------------------------------------
# Degenerate input: empty splits
# ---------------------------------------------------------------------------


def test_fit_one_row_golden_empty_holdout() -> None:
    """1-row golden set with default holdout_fraction=0.3 yields empty holdout."""
    golden = _make_golden(1, seed=99)
    with pytest.raises(ContractViolation) as exc_info:
        fit(golden, seed=99)
    message = str(exc_info.value)
    assert "n_golden=1" in message
    assert "n_calibration" in message
    assert "n_holdout" in message


def test_fit_zero_holdout_fraction_empty_holdout() -> None:
    """100-row golden set with holdout_fraction=0.0 yields empty holdout."""
    golden = _make_golden(100, seed=100)
    config = CalibrationConfig(holdout_fraction=0.0)
    with pytest.raises(ContractViolation) as exc_info:
        fit(golden, seed=100, config=config)
    message = str(exc_info.value)
    assert "n_golden=100" in message
    assert "n_calibration" in message
    assert "n_holdout" in message


def test_fit_one_holdout_fraction_empty_calibration() -> None:
    """100-row golden set with holdout_fraction=1.0 yields empty calibration."""
    golden = _make_golden(100, seed=101)
    config = CalibrationConfig(holdout_fraction=1.0)
    with pytest.raises(ContractViolation) as exc_info:
        fit(golden, seed=101, config=config)
    message = str(exc_info.value)
    assert "n_golden=100" in message
    assert "n_calibration=0" in message
    assert "n_holdout=100" in message


def test_fit_negative_seed_is_contract_violation() -> None:
    with pytest.raises(ContractViolation, match="seed must be a non-negative integer"):
        fit(_make_golden(50, seed=1), seed=-1)

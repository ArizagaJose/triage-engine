"""Synthetic data generator with injected, controlled error rates.

Three scenarios ("well_calibrated", "overconfident", "drift") share the same
generative process (3 segments, per-segment Uniform bounds on the true error
probability ``p``, Bernoulli errors). "well_calibrated" and "overconfident"
differ only in the score formula. "drift" reuses the well_calibrated golden
process unchanged, but draws batch/deployment errors from a doubled
probability while keeping the score formula on the original (undrifted) p —
simulating a policy that is honest at calibration time and stale at
deployment time. See docs/interfaces.md §6 for the pinned contract.
"""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from triage.artifacts import write_artifact
from triage.exceptions import ContractViolation

# Segment order is fixed and used everywhere randomness is consumed, so that
# the sequence of rng calls (and therefore the generated data) is a pure
# function of (scenario, seed).
_SEGMENT_ORDER: tuple[str, ...] = ("seg_high", "seg_mid", "seg_low")

# Per-segment Uniform(lo, hi) bounds on the true error probability p.
_P_BOUNDS: dict[str, tuple[float, float]] = {
    "seg_high": (0.01, 0.30),
    "seg_mid": (0.02, 0.40),
    "seg_low": (0.05, 0.50),
}

_NOISE_SIGMA = 0.01

_N_GOLDEN = 2000
_N_BATCH = 10000

_VALID_SCENARIOS: tuple[str, ...] = ("drift", "overconfident", "well_calibrated")

_DRIFT_FACTOR = 2.0

# Segment assignment scheme, recorded verbatim in the manifest.
_SEGMENT_ASSIGNMENT_SCHEME = (
    "np.random.default_rng(seed).choice(['seg_high', 'seg_mid', 'seg_low'], "
    "size=n) with uniform probability (equal thirds), consumed before the "
    "per-segment p draws"
)

_SCORE_FORMULAS: dict[str, str] = {
    "well_calibrated": "one_minus_p",
    "overconfident": "one_minus_p_over_3",
    "drift": "one_minus_p",
}

_COSTS_YAML = """\
budget:
  type: fraction
  value: 0.10
gates:
  max_ece: 0.05
segments:
  seg_high:
    false_accept_cost: 500.0
    review_cost: 4.0
  seg_mid:
    false_accept_cost: 50.0
    review_cost: 3.0
  seg_low:
    false_accept_cost: 10.0
    review_cost: 2.0
  default:
    false_accept_cost: 50.0
    review_cost: 3.0
"""


def _generate_split(
    rng: np.random.Generator, *, n: int, prefix: str, scenario: str, drift: bool = False
) -> pl.DataFrame:
    """Generate one split (golden or batch) of synthetic rows.

    Consumes rng state in a fixed order: segment assignment, then per-segment
    p draws (in `_SEGMENT_ORDER`), then Bernoulli errors, then score noise.

    `drift=True` (used for the drift scenario's batch/deployment split only)
    draws errors from `min(p * _DRIFT_FACTOR, 0.95)` instead of `p`, while
    keeping rng consumption order and the score formula (on the ORIGINAL p)
    unchanged.
    """
    segments = rng.choice(np.array(_SEGMENT_ORDER), size=n)

    p = np.empty(n, dtype=np.float64)
    for seg in _SEGMENT_ORDER:
        mask = segments == seg
        lo, hi = _P_BOUNDS[seg]
        p[mask] = rng.uniform(lo, hi, size=int(mask.sum()))

    error_p = np.minimum(p * _DRIFT_FACTOR, 0.95) if drift else p
    error = rng.binomial(1, error_p, size=n)
    noise = rng.normal(0.0, _NOISE_SIGMA, size=n)

    if scenario in ("well_calibrated", "drift"):
        raw_score = 1.0 - p
    else:
        raw_score = 1.0 - p / 3.0
    score = np.clip(raw_score + noise, 0.0, 1.0)

    true_value = np.full(n, "v", dtype=object)
    predicted_value = np.where(error == 1, "x", "v")

    item_id = [f"{prefix}-{i:06d}" for i in range(1, n + 1)]

    df = pl.DataFrame(
        {
            "item_id": item_id,
            "segment": segments.tolist(),
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
    return df.sort("item_id")


def _segment_error_rates(df: pl.DataFrame) -> dict[str, float]:
    """Realized mean error rate per segment, computed from predicted vs true value."""
    result: dict[str, float] = {}
    per_seg = (
        df.with_columns(
            (pl.col("predicted_value") != pl.col("true_value")).cast(pl.Float64).alias("_error")
        )
        .group_by("segment")
        .agg(pl.col("_error").mean().alias("mean_error"))
    )
    rates = {row["segment"]: row["mean_error"] for row in per_seg.iter_rows(named=True)}
    for seg in _SEGMENT_ORDER:
        result[seg] = float(rates.get(seg, 0.0))
    return result


def generate(scenario: str, *, seed: int) -> dict[str, pl.DataFrame | dict[str, Any]]:
    """Generate synthetic data for a scenario.

    Args:
        scenario: "well_calibrated", "overconfident", or "drift"
        seed: Random seed for reproducibility

    Returns:
        dict with keys "golden", "batch", "manifest"
        - "golden": pl.DataFrame with item_id, segment, score, predicted_value, true_value
        - "batch": pl.DataFrame with same schema
        - "manifest": dict with scenario, seed, n rows, p distributions, etc.
    """
    if scenario not in _VALID_SCENARIOS:
        raise ContractViolation(
            f"Unknown scenario {scenario!r}. Valid scenarios: {', '.join(_VALID_SCENARIOS)}"
        )

    # Drift golden is generated by the IDENTICAL process as well_calibrated
    # (score = 1 - p + noise, error ~ Bernoulli(p)); only the batch/deployment
    # split draws errors from the drifted probability.
    golden_scenario = "well_calibrated" if scenario == "drift" else scenario

    rng = np.random.default_rng(seed)
    golden = _generate_split(rng, n=_N_GOLDEN, prefix="g", scenario=golden_scenario)
    batch = _generate_split(
        rng, n=_N_BATCH, prefix="b", scenario=scenario, drift=(scenario == "drift")
    )

    golden_rates = _segment_error_rates(golden)
    batch_rates = _segment_error_rates(batch)

    segments_manifest = {
        seg: {
            "p_min": _P_BOUNDS[seg][0],
            "p_max": _P_BOUNDS[seg][1],
            "realized_mean_error_golden": golden_rates[seg],
            "realized_mean_error_batch": batch_rates[seg],
        }
        for seg in _SEGMENT_ORDER
    }

    manifest: dict[str, Any] = {
        "scenario": scenario,
        "seed": seed,
        "n_golden": _N_GOLDEN,
        "n_batch": _N_BATCH,
        "segments": segments_manifest,
        "score_formula": _SCORE_FORMULAS[scenario],
        "noise_sigma": _NOISE_SIGMA,
        "segment_assignment_scheme": _SEGMENT_ASSIGNMENT_SCHEME,
        "drift_factor": _DRIFT_FACTOR if scenario == "drift" else None,
    }

    return {"golden": golden, "batch": batch, "manifest": manifest}


def write(scenario: str, *, seed: int, out_dir: Path) -> None:
    """Write synthetic scenario to directory.

    Creates:
    - golden.parquet
    - batch.parquet
    - manifest.json
    - costs.yaml
    """
    data = generate(scenario, seed=seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    golden_df = data["golden"]
    batch_df = data["batch"]
    manifest = data["manifest"]
    assert isinstance(golden_df, pl.DataFrame)
    assert isinstance(batch_df, pl.DataFrame)
    assert isinstance(manifest, dict)

    golden_df.write_parquet(out_dir / "golden.parquet")
    batch_df.write_parquet(out_dir / "batch.parquet")
    write_artifact(manifest, out_dir / "manifest.json")
    (out_dir / "costs.yaml").write_text(_COSTS_YAML)

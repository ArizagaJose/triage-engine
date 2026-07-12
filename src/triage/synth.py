"""Synthetic data generator with injected, controlled error rates."""

from pathlib import Path
from typing import Any

import polars as pl


def generate(scenario: str, *, seed: int) -> dict[str, pl.DataFrame | dict[str, Any]]:
    """Generate synthetic data for a scenario.

    Args:
        scenario: "well_calibrated" or "overconfident"
        seed: Random seed for reproducibility

    Returns:
        dict with keys "golden", "batch", "manifest"
        - "golden": pl.DataFrame with true_value, predicted_value, score, segment, item_id
        - "batch": pl.DataFrame with same schema
        - "manifest": dict with scenario, seed, n_rows, p distributions, etc.
    """
    raise NotImplementedError


def write(scenario: str, *, seed: int, out_dir: Path) -> None:
    """Write synthetic scenario to directory.

    Creates:
    - golden.parquet
    - batch.parquet
    - manifest.json
    - costs.yaml
    """
    raise NotImplementedError

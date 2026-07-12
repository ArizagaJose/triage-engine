"""Input contract validation."""

from pathlib import Path

import polars as pl


def validate(df: pl.DataFrame, *, require_truth: bool = False) -> pl.DataFrame:
    """Validate input dataframe against contract.

    Checks:
    - Required columns: item_id, segment, score, predicted_value
    - Dtypes: item_id/segment/predicted_value as pl.String, score as pl.Float64
    - item_id: non-null, unique
    - segment, predicted_value: non-null
    - score: non-null, finite, in [0.0, 1.0]
    - true_value: optional, pl.String; required if require_truth=True
    - Non-empty dataframe

    Collects ALL violations and raises ContractViolation with full list.
    Returns validated dataframe (with score upcasted if needed) otherwise.
    """
    raise NotImplementedError


def read_input(path: Path, *, require_truth: bool = False) -> pl.DataFrame:
    """Read CSV or Parquet file and validate against contract.

    Accepts .parquet or .csv. CSV is read with try_parse_dates=False
    and explicit schema_overrides. Parse errors are ContractViolation.
    """
    raise NotImplementedError

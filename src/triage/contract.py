"""Input contract validation."""

from pathlib import Path

import polars as pl

from triage.exceptions import ContractViolation

_REQUIRED_STRING_COLUMNS = ("item_id", "segment", "predicted_value")
_REQUIRED_COLUMNS = ("item_id", "segment", "score", "predicted_value")

_CSV_SCHEMA_OVERRIDES: dict[str, type[pl.DataType]] = {
    "item_id": pl.String,
    "segment": pl.String,
    "score": pl.Float64,
    "predicted_value": pl.String,
    "true_value": pl.String,
}


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
    violations: list[str] = []
    working = df

    for column in _REQUIRED_COLUMNS:
        if column not in df.columns:
            violations.append(f"{column}: required column is missing")

    if df.height == 0:
        violations.append("dataframe: no rows (empty input is a violation)")

    for column in _REQUIRED_STRING_COLUMNS:
        if column in df.columns and df.schema[column] != pl.String:
            violations.append(f"{column}: expected dtype String, got {df.schema[column]}")

    if "score" in df.columns:
        score_dtype = df.schema["score"]
        if score_dtype == pl.Float32:
            working = working.with_columns(pl.col("score").cast(pl.Float64))
        elif score_dtype != pl.Float64:
            violations.append(
                f"score: expected dtype Float64 (Float32 accepted and upcast), got {score_dtype}"
            )

    if "item_id" in df.columns:
        item_id = df["item_id"]
        null_count = item_id.null_count()
        if null_count > 0:
            violations.append(f"item_id: contains {null_count} null value(s)")
        non_null_ids = item_id.drop_nulls()
        if non_null_ids.len() != non_null_ids.n_unique():
            dup_values = sorted(
                non_null_ids.filter(non_null_ids.is_duplicated()).unique().to_list()
            )
            violations.append(f"item_id: duplicate values found: {dup_values}")

    for column in ("segment", "predicted_value"):
        if column in df.columns:
            null_count = df[column].null_count()
            if null_count > 0:
                violations.append(f"{column}: contains {null_count} null value(s)")

    if "score" in working.columns and working.schema["score"] in (pl.Float64, pl.Float32):
        score = working["score"]
        null_count = score.null_count()
        if null_count > 0:
            violations.append(f"score: contains {null_count} null value(s)")
        non_null_scores = score.drop_nulls()
        if non_null_scores.len() > 0:
            finite_mask = non_null_scores.is_finite()
            n_non_finite = int((~finite_mask).sum())
            if n_non_finite > 0:
                violations.append(f"score: contains {n_non_finite} non-finite value(s) (NaN/inf)")
            finite_scores = non_null_scores.filter(finite_mask)
            out_of_range = finite_scores.filter((finite_scores < 0.0) | (finite_scores > 1.0))
            if out_of_range.len() > 0:
                violations.append(
                    f"score: {out_of_range.len()} value(s) outside the [0.0, 1.0] range"
                )

    if "true_value" in df.columns:
        if df.schema["true_value"] != pl.String:
            violations.append(f"true_value: expected dtype String, got {df.schema['true_value']}")
        if require_truth:
            null_count = df["true_value"].null_count()
            if null_count > 0:
                violations.append(
                    f"true_value: contains {null_count} null value(s) but require_truth=True"
                )
    elif require_truth:
        violations.append("true_value: required column is missing (require_truth=True)")

    if violations:
        raise ContractViolation("\n".join(violations))

    return working


def read_input(path: Path, *, require_truth: bool = False) -> pl.DataFrame:
    """Read CSV or Parquet file and validate against contract.

    Accepts .parquet or .csv. CSV is read with try_parse_dates=False
    and explicit schema_overrides. Parse errors are ContractViolation.
    """
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df = pl.read_parquet(path)
        except Exception as exc:
            raise ContractViolation(f"{path}: failed to read parquet file: {exc}") from exc
    elif suffix == ".csv":
        try:
            df = pl.read_csv(
                path,
                try_parse_dates=False,
                schema_overrides=_CSV_SCHEMA_OVERRIDES,
            )
        except Exception as exc:
            raise ContractViolation(f"{path}: failed to parse CSV file: {exc}") from exc
    else:
        raise ContractViolation(
            f"{path}: unsupported file extension {suffix!r}; expected .parquet or .csv"
        )

    return validate(df, require_truth=require_truth)

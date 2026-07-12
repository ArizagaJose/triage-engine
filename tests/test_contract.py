"""Tests for triage.contract: validate() and read_input()."""

from pathlib import Path

import polars as pl
import pytest

from triage.contract import read_input, validate
from triage.exceptions import ContractViolation


def _valid_df(**overrides: pl.Series) -> pl.DataFrame:
    """Build a minimal, fully valid input dataframe, with optional column overrides."""
    data: dict[str, pl.Series] = {
        "item_id": pl.Series("item_id", ["a", "b", "c"], dtype=pl.String),
        "segment": pl.Series("segment", ["seg1", "seg1", "seg2"], dtype=pl.String),
        "score": pl.Series("score", [0.1, 0.5, 0.9], dtype=pl.Float64),
        "predicted_value": pl.Series("predicted_value", ["v1", "v2", "v3"], dtype=pl.String),
    }
    data.update(overrides)
    return pl.DataFrame(list(data.values()))


# --------------------------------------------------------------------------
# validate(): happy paths
# --------------------------------------------------------------------------


def test_validate_happy_path_returns_equal_frame() -> None:
    df = _valid_df()
    result = validate(df)
    assert result.equals(df)


def test_validate_happy_path_with_true_value_no_require_truth() -> None:
    df = _valid_df(
        true_value=pl.Series("true_value", ["v1", "x", "v3"], dtype=pl.String),
    )
    result = validate(df)
    assert result.equals(df)


def test_validate_true_value_present_and_non_null_with_require_truth() -> None:
    df = _valid_df(
        true_value=pl.Series("true_value", ["v1", "x", "v3"], dtype=pl.String),
    )
    result = validate(df, require_truth=True)
    assert result.equals(df)


# --------------------------------------------------------------------------
# validate(): the only permitted coercion
# --------------------------------------------------------------------------


def test_validate_float32_score_is_upcast_to_float64() -> None:
    df = _valid_df(score=pl.Series("score", [0.1, 0.5, 0.9], dtype=pl.Float32))
    result = validate(df)
    assert result.schema["score"] == pl.Float64
    assert result["score"].to_list() == pytest.approx([0.1, 0.5, 0.9])


def test_validate_other_dtypes_are_not_coerced() -> None:
    """item_id as Int64 must be rejected, not silently cast to String."""
    df = _valid_df(item_id=pl.Series("item_id", [1, 2, 3], dtype=pl.Int64))
    with pytest.raises(ContractViolation, match="item_id"):
        validate(df)


# --------------------------------------------------------------------------
# validate(): missing columns
# --------------------------------------------------------------------------


@pytest.mark.parametrize("column", ["item_id", "segment", "score", "predicted_value"])
def test_validate_missing_required_column(column: str) -> None:
    df = _valid_df().drop(column)
    with pytest.raises(ContractViolation, match=rf"{column}: required column is missing"):
        validate(df)


# --------------------------------------------------------------------------
# validate(): dtype violations
# --------------------------------------------------------------------------


def test_validate_segment_wrong_dtype() -> None:
    df = _valid_df(segment=pl.Series("segment", [1, 2, 3], dtype=pl.Int64))
    with pytest.raises(ContractViolation, match="segment: expected dtype String"):
        validate(df)


def test_validate_predicted_value_wrong_dtype() -> None:
    df = _valid_df(predicted_value=pl.Series("predicted_value", [1, 2, 3], dtype=pl.Int64))
    with pytest.raises(ContractViolation, match="predicted_value: expected dtype String"):
        validate(df)


def test_validate_score_wrong_dtype() -> None:
    df = _valid_df(score=pl.Series("score", [1, 2, 3], dtype=pl.Int64))
    with pytest.raises(ContractViolation, match="score: expected dtype Float64"):
        validate(df)


def test_validate_true_value_wrong_dtype() -> None:
    df = _valid_df(true_value=pl.Series("true_value", [1, 2, 3], dtype=pl.Int64))
    with pytest.raises(ContractViolation, match="true_value: expected dtype String"):
        validate(df)


# --------------------------------------------------------------------------
# validate(): item_id null / uniqueness
# --------------------------------------------------------------------------


def test_validate_null_item_id() -> None:
    df = _valid_df(item_id=pl.Series("item_id", ["a", None, "c"], dtype=pl.String))
    with pytest.raises(ContractViolation, match=r"item_id: contains 1 null value\(s\)"):
        validate(df)


def test_validate_duplicate_item_id() -> None:
    df = _valid_df(item_id=pl.Series("item_id", ["a", "a", "c"], dtype=pl.String))
    with pytest.raises(ContractViolation, match="item_id: duplicate values found"):
        validate(df)


# --------------------------------------------------------------------------
# validate(): segment / predicted_value null
# --------------------------------------------------------------------------


def test_validate_null_segment() -> None:
    df = _valid_df(segment=pl.Series("segment", ["seg1", None, "seg2"], dtype=pl.String))
    with pytest.raises(ContractViolation, match=r"segment: contains 1 null value\(s\)"):
        validate(df)


def test_validate_null_predicted_value() -> None:
    df = _valid_df(
        predicted_value=pl.Series("predicted_value", ["v1", None, "v3"], dtype=pl.String)
    )
    with pytest.raises(ContractViolation, match=r"predicted_value: contains 1 null value\(s\)"):
        validate(df)


# --------------------------------------------------------------------------
# validate(): score range / null / finiteness
# --------------------------------------------------------------------------


def test_validate_score_null() -> None:
    df = _valid_df(score=pl.Series("score", [0.1, None, 0.9], dtype=pl.Float64))
    with pytest.raises(ContractViolation, match=r"score: contains 1 null value\(s\)"):
        validate(df)


def test_validate_score_nan() -> None:
    df = _valid_df(score=pl.Series("score", [0.1, float("nan"), 0.9], dtype=pl.Float64))
    with pytest.raises(ContractViolation, match=r"score: contains 1 non-finite value\(s\)"):
        validate(df)


def test_validate_score_infinite() -> None:
    df = _valid_df(score=pl.Series("score", [0.1, float("inf"), 0.9], dtype=pl.Float64))
    with pytest.raises(ContractViolation, match=r"score: contains 1 non-finite value\(s\)"):
        validate(df)


def test_validate_score_below_range() -> None:
    df = _valid_df(score=pl.Series("score", [-0.1, 0.5, 0.9], dtype=pl.Float64))
    with pytest.raises(ContractViolation, match=r"score: 1 value\(s\) outside the \[0.0, 1.0\]"):
        validate(df)


def test_validate_score_above_range() -> None:
    df = _valid_df(score=pl.Series("score", [0.1, 0.5, 1.1], dtype=pl.Float64))
    with pytest.raises(ContractViolation, match=r"score: 1 value\(s\) outside the \[0.0, 1.0\]"):
        validate(df)


def test_validate_score_boundary_values_are_valid() -> None:
    df = _valid_df(score=pl.Series("score", [0.0, 0.5, 1.0], dtype=pl.Float64))
    result = validate(df)
    assert result.equals(df)


# --------------------------------------------------------------------------
# validate(): empty dataframe
# --------------------------------------------------------------------------


def test_validate_empty_dataframe() -> None:
    df = pl.DataFrame(
        {
            "item_id": pl.Series("item_id", [], dtype=pl.String),
            "segment": pl.Series("segment", [], dtype=pl.String),
            "score": pl.Series("score", [], dtype=pl.Float64),
            "predicted_value": pl.Series("predicted_value", [], dtype=pl.String),
        }
    )
    with pytest.raises(ContractViolation, match="dataframe: no rows"):
        validate(df)


# --------------------------------------------------------------------------
# validate(): true_value / require_truth
# --------------------------------------------------------------------------


def test_validate_require_truth_missing_column() -> None:
    df = _valid_df()
    with pytest.raises(ContractViolation, match=r"true_value: required column is missing"):
        validate(df, require_truth=True)


def test_validate_require_truth_null_values() -> None:
    df = _valid_df(
        true_value=pl.Series("true_value", ["v1", None, "v3"], dtype=pl.String),
    )
    with pytest.raises(ContractViolation, match=r"true_value: contains 1 null value\(s\)"):
        validate(df, require_truth=True)


# --------------------------------------------------------------------------
# validate(): multi-violation aggregation
# --------------------------------------------------------------------------


def test_validate_collects_all_violations_in_one_exception() -> None:
    df = pl.DataFrame(
        {
            "item_id": pl.Series("item_id", ["a", "a", None], dtype=pl.String),
            "segment": pl.Series("segment", ["seg1", None, "seg2"], dtype=pl.String),
            "score": pl.Series("score", [-0.1, 0.5, float("nan")], dtype=pl.Float64),
            "predicted_value": pl.Series("predicted_value", ["v1", "v2", "v3"], dtype=pl.String),
        }
    )
    with pytest.raises(ContractViolation) as exc_info:
        validate(df)

    message = str(exc_info.value)
    assert "item_id: contains 1 null value(s)" in message
    assert "item_id: duplicate values found" in message
    assert "segment: contains 1 null value(s)" in message
    assert "score: contains 1 non-finite value(s)" in message
    assert "score: 1 value(s) outside the [0.0, 1.0]" in message
    # each violation is on its own line
    assert len(message.splitlines()) == 5


def test_validate_missing_columns_reported_together() -> None:
    df = pl.DataFrame({"score": pl.Series("score", [0.5], dtype=pl.Float64)})
    with pytest.raises(ContractViolation) as exc_info:
        validate(df)
    message = str(exc_info.value)
    assert "item_id: required column is missing" in message
    assert "segment: required column is missing" in message
    assert "predicted_value: required column is missing" in message


# --------------------------------------------------------------------------
# read_input(): happy paths (parquet + csv round trip)
# --------------------------------------------------------------------------


def test_read_input_parquet_round_trip(tmp_path: Path) -> None:
    df = _valid_df()
    path = tmp_path / "batch.parquet"
    df.write_parquet(path)

    result = read_input(path)
    assert result.sort("item_id").equals(df.sort("item_id"))


def test_read_input_csv_round_trip(tmp_path: Path) -> None:
    df = _valid_df()
    path = tmp_path / "batch.csv"
    df.write_csv(path)

    result = read_input(path)
    assert result["item_id"].to_list() == df["item_id"].to_list()
    assert result["segment"].to_list() == df["segment"].to_list()
    assert result["score"].to_list() == pytest.approx(df["score"].to_list())
    assert result["predicted_value"].to_list() == df["predicted_value"].to_list()
    assert result.schema["score"] == pl.Float64


def test_read_input_csv_with_true_value_require_truth(tmp_path: Path) -> None:
    df = _valid_df(
        true_value=pl.Series("true_value", ["v1", "x", "v3"], dtype=pl.String),
    )
    path = tmp_path / "golden.csv"
    df.write_csv(path)

    result = read_input(path, require_truth=True)
    assert result["true_value"].to_list() == df["true_value"].to_list()


# --------------------------------------------------------------------------
# read_input(): errors
# --------------------------------------------------------------------------


def test_read_input_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    path.write_text("{}")
    with pytest.raises(ContractViolation, match="unsupported file extension"):
        read_input(path)


def test_read_input_csv_parse_failure_is_contract_violation(tmp_path: Path) -> None:
    path = tmp_path / "batch.csv"
    path.write_text("item_id,segment,score,predicted_value\na,seg1,not_a_number,v1\n")
    with pytest.raises(ContractViolation, match="failed to parse CSV file"):
        read_input(path)


def test_read_input_propagates_validation_violations(tmp_path: Path) -> None:
    """A structurally readable file that fails contract checks still raises ContractViolation."""
    df = _valid_df(score=pl.Series("score", [-0.1, 0.5, 0.9], dtype=pl.Float64))
    path = tmp_path / "batch.parquet"
    df.write_parquet(path)

    with pytest.raises(ContractViolation, match=r"score: 1 value\(s\) outside"):
        read_input(path)


def test_read_input_require_truth_missing_from_csv(tmp_path: Path) -> None:
    df = _valid_df()
    path = tmp_path / "golden.csv"
    df.write_csv(path)

    with pytest.raises(ContractViolation, match="true_value: required column is missing"):
        read_input(path, require_truth=True)

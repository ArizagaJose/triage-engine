"""Unit tests for `src/triage/review_queue.py`: CSV/Parquet export plus the
`review_queue.json` sidecar manifest. Queue frames come from the real
optimizer (`plan_with_queue`) wherever the content matters, so these tests
exercise the actual writer against the actual selection, not a hand-built
stand-in."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from tests.support import IdentityCalibrator, batch_frame
from triage.artifacts import file_sha256
from triage.costs import Budget, CostModel, SegmentCosts
from triage.optimizer import REVIEW_QUEUE_COLUMNS, REVIEW_QUEUE_SCHEMA, plan_with_queue
from triage.review_queue import (
    GATE_STATUS_PASS,
    GateStatus,
    build_manifest,
    export,
    queue_hash,
    write_csv,
    write_parquet,
)

CALIBRATOR = IdentityCalibrator()

_SHA = "a" * 64


def _cost_model(budget: Budget) -> CostModel:
    return CostModel(
        budget=budget,
        segments={
            "segA": SegmentCosts(false_accept_cost=100.0, review_cost=10.0),
            "segB": SegmentCosts(false_accept_cost=50.0, review_cost=5.0),
        },
    )


def _rows() -> list[tuple[str, str, float]]:
    # Same fixture as tests/unit/test_optimizer.py's main case.
    return [
        ("i1", "segA", 0.90),
        ("i2", "segA", 0.50),
        ("i3", "segA", 0.05),
        ("i4", "segB", 0.80),
        ("i5", "segB", 0.30),
        ("i6", "segB", 0.05),
    ]


def _result(budget: Budget) -> Any:
    return plan_with_queue(
        batch_frame(_rows()),
        CALIBRATOR,
        _cost_model(budget),
        input_sha256s={"batch": _SHA, "calibrator": "b" * 64, "costs": "c" * 64},
        tool_version="0.1.0",
    )


# --- write_csv / write_parquet: exact round trips --------------------------------


def test_write_csv_round_trips_exactly(tmp_path: Path) -> None:
    queue = _result(Budget(type="count", value=3)).review_queue
    out = tmp_path / "q.csv"
    write_csv(queue, out)
    back = pl.read_csv(out, schema_overrides=dict(REVIEW_QUEUE_SCHEMA))
    assert back.equals(queue)


def test_write_parquet_round_trips_exactly(tmp_path: Path) -> None:
    queue = _result(Budget(type="count", value=3)).review_queue
    out = tmp_path / "q.parquet"
    write_parquet(queue, out)
    back = pl.read_parquet(out)
    assert back.equals(queue)


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    queue = _result(Budget(type="count", value=3)).review_queue
    csv_out = tmp_path / "nested" / "dir" / "q.csv"
    parquet_out = tmp_path / "other" / "dir" / "q.parquet"
    write_csv(queue, csv_out)
    write_parquet(queue, parquet_out)
    assert csv_out.exists() and parquet_out.exists()


def test_queue_columns_match_schema() -> None:
    queue = _result(Budget(type="count", value=3)).review_queue
    assert queue.columns == list(REVIEW_QUEUE_COLUMNS)
    assert queue.schema == REVIEW_QUEUE_SCHEMA


# --- empty selection: still a valid, correctly-typed export ----------------------


def test_empty_selection_writes_header_only_csv(tmp_path: Path) -> None:
    queue = _result(Budget(type="fraction", value=0.001)).review_queue
    assert queue.height == 0
    out = tmp_path / "q.csv"
    write_csv(queue, out)
    lines = out.read_text().splitlines()
    assert lines == [",".join(REVIEW_QUEUE_COLUMNS)]


def test_empty_selection_writes_typed_zero_row_parquet(tmp_path: Path) -> None:
    queue = _result(Budget(type="fraction", value=0.001)).review_queue
    out = tmp_path / "q.parquet"
    write_parquet(queue, out)
    back = pl.read_parquet(out)
    assert back.height == 0
    assert back.schema == REVIEW_QUEUE_SCHEMA


# --- queue_hash: format-independent, content-sensitive ---------------------------


def test_queue_hash_matches_across_csv_and_parquet_round_trip(tmp_path: Path) -> None:
    queue = _result(Budget(type="count", value=3)).review_queue
    csv_path, parquet_path = tmp_path / "q.csv", tmp_path / "q.parquet"
    write_csv(queue, csv_path)
    write_parquet(queue, parquet_path)
    from_csv = pl.read_csv(csv_path, schema_overrides=dict(REVIEW_QUEUE_SCHEMA))
    from_parquet = pl.read_parquet(parquet_path)
    assert queue_hash(queue) == queue_hash(from_csv) == queue_hash(from_parquet)


def test_queue_hash_changes_with_any_cell() -> None:
    queue = _result(Budget(type="count", value=3)).review_queue
    tampered = queue.with_columns(pl.col("p_error") * 1.0000001)
    assert queue_hash(queue) != queue_hash(tampered)


def test_queue_hash_is_order_sensitive() -> None:
    queue = _result(Budget(type="count", value=3)).review_queue
    reversed_queue = queue.reverse()
    assert queue_hash(queue) != queue_hash(reversed_queue)


def test_queue_hash_stable_across_calls() -> None:
    queue = _result(Budget(type="count", value=3)).review_queue
    assert queue_hash(queue) == queue_hash(queue.clone())


def test_queue_hash_empty_is_deterministic() -> None:
    empty = _result(Budget(type="fraction", value=0.001)).review_queue
    assert queue_hash(empty) == queue_hash(empty.clone())


# --- GateStatus --------------------------------------------------------------------


class TestGateStatus:
    def test_pass_with_no_failed_gates(self) -> None:
        assert GateStatus(status="pass").to_dict() == {"status": "pass", "failed": []}
        assert GATE_STATUS_PASS.to_dict() == {"status": "pass", "failed": []}

    def test_failed_names_the_gates(self) -> None:
        status = GateStatus(status="failed", failed=("max_ece",))
        assert status.to_dict() == {"status": "failed", "failed": ["max_ece"]}

    def test_pass_with_failed_names_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GateStatus(status="pass", failed=("max_ece",))

    def test_failed_without_names_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            GateStatus(status="failed")

    def test_unknown_status_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="'pass' or 'failed'"):
            GateStatus(status="warn")


# --- build_manifest ----------------------------------------------------------------


def _manifest_inputs(budget: Budget, tmp_path: Path) -> dict[str, Any]:
    result = _result(budget)
    csv_path, parquet_path = tmp_path / "q.csv", tmp_path / "q.parquet"
    write_csv(result.review_queue, csv_path)
    write_parquet(result.review_queue, parquet_path)
    return {
        "review_queue": result.review_queue,
        "policy_hash": result.policy.policy_hash,
        "input_sha256s": result.policy.input_sha256s,
        "totals": result.policy.totals,
        "files": {"csv": csv_path, "parquet": parquet_path},
    }


def test_build_manifest_shape(tmp_path: Path) -> None:
    inputs = _manifest_inputs(Budget(type="count", value=3), tmp_path)
    manifest = build_manifest(
        inputs["review_queue"],
        policy_hash=inputs["policy_hash"],
        input_sha256s=inputs["input_sha256s"],
        totals=inputs["totals"],
        files=inputs["files"],
        gate_status=GATE_STATUS_PASS,
        tool_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
    )
    assert set(manifest) == {
        "columns",
        "created_at",
        "files",
        "gates",
        "input_sha256s",
        "n_rows",
        "policy_hash",
        "queue_hash",
        "schema_version",
        "tool_version",
        "totals",
    }
    assert manifest["columns"] == list(REVIEW_QUEUE_COLUMNS)
    assert manifest["n_rows"] == 3
    assert manifest["policy_hash"] == inputs["policy_hash"]
    assert manifest["input_sha256s"] == inputs["input_sha256s"]
    assert manifest["schema_version"] == 1
    assert manifest["tool_version"] == "0.1.0"
    assert manifest["gates"] == {"status": "pass", "failed": []}
    assert manifest["totals"] == {
        "expected_captured_loss": inputs["totals"]["expected_captured_loss"],
        "n_review": inputs["totals"]["n_review"],
        "review_spend": inputs["totals"]["review_spend"],
    }


def test_build_manifest_files_use_basenames_and_real_sha256(tmp_path: Path) -> None:
    inputs = _manifest_inputs(Budget(type="count", value=3), tmp_path)
    manifest = build_manifest(
        inputs["review_queue"],
        policy_hash=inputs["policy_hash"],
        input_sha256s=inputs["input_sha256s"],
        totals=inputs["totals"],
        files=inputs["files"],
        gate_status=GATE_STATUS_PASS,
        tool_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
    )
    assert manifest["files"] == {
        "csv": {"path": "q.csv", "sha256": file_sha256(inputs["files"]["csv"])},
        "parquet": {"path": "q.parquet", "sha256": file_sha256(inputs["files"]["parquet"])},
    }
    # Not the full (tmp_path-dependent) path: the manifest must not depend on
    # where the caller happened to write the files.
    assert str(tmp_path) not in str(manifest["files"])


def test_build_manifest_only_requested_formats_appear(tmp_path: Path) -> None:
    inputs = _manifest_inputs(Budget(type="count", value=3), tmp_path)
    manifest = build_manifest(
        inputs["review_queue"],
        policy_hash=inputs["policy_hash"],
        input_sha256s=inputs["input_sha256s"],
        totals=inputs["totals"],
        files={"csv": inputs["files"]["csv"]},
        gate_status=GATE_STATUS_PASS,
        tool_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
    )
    assert set(manifest["files"]) == {"csv"}


def test_build_manifest_gate_failure_is_recorded(tmp_path: Path) -> None:
    inputs = _manifest_inputs(Budget(type="count", value=3), tmp_path)
    manifest = build_manifest(
        inputs["review_queue"],
        policy_hash=inputs["policy_hash"],
        input_sha256s=inputs["input_sha256s"],
        totals=inputs["totals"],
        files=inputs["files"],
        gate_status=GateStatus(status="failed", failed=("max_ece", "max_residual_risk")),
        tool_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
    )
    assert manifest["gates"] == {"status": "failed", "failed": ["max_ece", "max_residual_risk"]}


# --- export(): orchestration ---------------------------------------------------------


def test_export_both_formats_writes_three_files(tmp_path: Path) -> None:
    result = _result(Budget(type="count", value=3))
    manifest = export(
        result.review_queue,
        csv_path=tmp_path / "q.csv",
        parquet_path=tmp_path / "q.parquet",
        manifest_path=tmp_path / "review_queue.json",
        policy_hash=result.policy.policy_hash,
        input_sha256s=result.policy.input_sha256s,
        totals=result.policy.totals,
        gate_status=GATE_STATUS_PASS,
        tool_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
    )
    assert (tmp_path / "q.csv").exists()
    assert (tmp_path / "q.parquet").exists()
    assert (tmp_path / "review_queue.json").exists()
    assert set(manifest["files"]) == {"csv", "parquet"}


def test_export_single_format_omits_the_other(tmp_path: Path) -> None:
    result = _result(Budget(type="count", value=3))
    manifest = export(
        result.review_queue,
        csv_path=tmp_path / "q.csv",
        parquet_path=None,
        manifest_path=tmp_path / "review_queue.json",
        policy_hash=result.policy.policy_hash,
        input_sha256s=result.policy.input_sha256s,
        totals=result.policy.totals,
        gate_status=GATE_STATUS_PASS,
        tool_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
    )
    assert not (tmp_path / "q.parquet").exists()
    assert set(manifest["files"]) == {"csv"}


def test_export_requires_at_least_one_format(tmp_path: Path) -> None:
    result = _result(Budget(type="count", value=3))
    with pytest.raises(ValueError, match="at least one"):
        export(
            result.review_queue,
            csv_path=None,
            parquet_path=None,
            manifest_path=tmp_path / "review_queue.json",
            policy_hash=result.policy.policy_hash,
            input_sha256s=result.policy.input_sha256s,
            totals=result.policy.totals,
            gate_status=GATE_STATUS_PASS,
            tool_version="0.1.0",
            created_at="2025-01-01T00:00:00+00:00",
        )
    assert not (tmp_path / "review_queue.json").exists()


def test_export_manifest_written_matches_returned_value(tmp_path: Path) -> None:
    import json

    result = _result(Budget(type="count", value=3))
    manifest_path = tmp_path / "review_queue.json"
    manifest = export(
        result.review_queue,
        csv_path=tmp_path / "q.csv",
        parquet_path=None,
        manifest_path=manifest_path,
        policy_hash=result.policy.policy_hash,
        input_sha256s=result.policy.input_sha256s,
        totals=result.policy.totals,
        gate_status=GATE_STATUS_PASS,
        tool_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
    )
    assert json.loads(manifest_path.read_text()) == manifest


def test_export_is_deterministic_given_the_same_queue(tmp_path: Path) -> None:
    result = _result(Budget(type="count", value=3))
    a, b = tmp_path / "a", tmp_path / "b"
    kwargs = dict(
        csv_path_name="q.csv",
        parquet_path_name="q.parquet",
        policy_hash=result.policy.policy_hash,
        input_sha256s=result.policy.input_sha256s,
        totals=result.policy.totals,
        gate_status=GATE_STATUS_PASS,
        tool_version="0.1.0",
    )

    def run(out: Path, created_at: str) -> dict[str, Any]:
        out.mkdir()
        return export(
            result.review_queue,
            csv_path=out / kwargs["csv_path_name"],
            parquet_path=out / kwargs["parquet_path_name"],
            manifest_path=out / "review_queue.json",
            policy_hash=kwargs["policy_hash"],
            input_sha256s=kwargs["input_sha256s"],
            totals=kwargs["totals"],
            gate_status=kwargs["gate_status"],
            tool_version=kwargs["tool_version"],
            created_at=created_at,
        )

    manifest_a = run(a, "2025-01-01T00:00:00+00:00")
    manifest_b = run(b, "2030-06-15T12:00:00+00:00")
    assert (a / "q.csv").read_bytes() == (b / "q.csv").read_bytes()
    assert (a / "q.parquet").read_bytes() == (b / "q.parquet").read_bytes()
    manifest_a.pop("created_at")
    manifest_b.pop("created_at")
    assert manifest_a == manifest_b


def test_export_empty_queue_end_to_end(tmp_path: Path) -> None:
    result = _result(Budget(type="fraction", value=0.001))
    manifest = export(
        result.review_queue,
        csv_path=tmp_path / "q.csv",
        parquet_path=tmp_path / "q.parquet",
        manifest_path=tmp_path / "review_queue.json",
        policy_hash=result.policy.policy_hash,
        input_sha256s=result.policy.input_sha256s,
        totals=result.policy.totals,
        gate_status=GATE_STATUS_PASS,
        tool_version="0.1.0",
        created_at="2025-01-01T00:00:00+00:00",
    )
    assert manifest["n_rows"] == 0
    assert manifest["totals"]["n_review"] == 0
    assert pl.read_csv(tmp_path / "q.csv", schema_overrides=dict(REVIEW_QUEUE_SCHEMA)).height == 0
    assert pl.read_parquet(tmp_path / "q.parquet").height == 0

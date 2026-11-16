"""Review queue export: CSV/Parquet of `optimizer.PlanResult.review_queue`,
plus a `review_queue.json` sidecar manifest that makes the export traceable
to its policy and source batch even with zero rows (an empty CSV has no row
to carry a hash in; Parquet footer metadata would only cover Parquet).

Nothing here recomputes the selection: the frame passed in is already the
optimizer's own `review_queue` (see docs/interfaces.md §5, §10). This module
only serializes it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl

from triage.artifacts import content_hash, file_sha256, write_artifact

SCHEMA_VERSION = 1

# The sidecar's file name; it always sits next to the export(s) it describes.
MANIFEST_NAME = "review_queue.json"


@dataclass(frozen=True)
class GateStatus:
    """The outcome of the plan-time gates (`max_ece`, `max_residual_risk`),
    evaluated WITHOUT raising, so it can be recorded in the manifest even when
    a gate fails and the CLI later raises for real (docs/interfaces.md §10.3).

    `failed` names the gates that did not pass (a subset of
    `{"max_ece", "max_residual_risk"}`); empty when every declared gate
    passed (a gate that is not declared -- `None` in the cost model -- never
    appears here, declared or not is not the same as pass or fail).
    """

    status: str  # "pass" | "failed"
    failed: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.status not in ("pass", "failed"):
            raise ValueError(f"GateStatus.status must be 'pass' or 'failed', got {self.status!r}")
        if self.status == "pass" and self.failed:
            raise ValueError("GateStatus.status='pass' but `failed` is non-empty")
        if self.status == "failed" and not self.failed:
            raise ValueError("GateStatus.status='failed' but `failed` is empty")

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "failed": list(self.failed)}


GATE_STATUS_PASS = GateStatus(status="pass")


def write_csv(review_queue: pl.DataFrame, path: Path) -> None:
    """Write the review queue as CSV. An empty frame still writes its header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    review_queue.write_csv(path)


def write_parquet(review_queue: pl.DataFrame, path: Path) -> None:
    """Write the review queue as Parquet. An empty frame still writes a valid,
    correctly-typed zero-row file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    review_queue.write_parquet(path)


def queue_hash(review_queue: pl.DataFrame) -> str:
    """Content hash of the queue's rows, independent of export format: a CSV
    and a Parquet export of the same `review_queue` always hash identically,
    so the manifest can prove they agree without re-reading both files."""
    return content_hash({"rows": review_queue.to_dicts()})


def build_manifest(
    review_queue: pl.DataFrame,
    *,
    policy_hash: str,
    input_sha256s: dict[str, str],
    totals: dict[str, Any],
    files: dict[str, Path],
    gate_status: GateStatus,
    tool_version: str,
    created_at: str,
) -> dict[str, Any]:
    """Build the review_queue.json content. `files` maps format name ("csv",
    "parquet") to the path it was ALREADY written to (sha256 is read off
    disk, not guessed); only file NAMES are recorded (not full paths), so the
    manifest does not depend on the output directory and stays comparable
    across runs the way every other artifact in this repo does."""
    return {
        "columns": list(review_queue.columns),
        "created_at": created_at,
        "files": {
            fmt: {"path": path.name, "sha256": file_sha256(path)} for fmt, path in files.items()
        },
        "gates": gate_status.to_dict(),
        "input_sha256s": dict(input_sha256s),
        "n_rows": review_queue.height,
        "policy_hash": policy_hash,
        "queue_hash": queue_hash(review_queue),
        "schema_version": SCHEMA_VERSION,
        "tool_version": tool_version,
        "totals": {
            "expected_captured_loss": totals["expected_captured_loss"],
            "n_review": totals["n_review"],
            "review_spend": totals["review_spend"],
        },
    }


def export(
    review_queue: pl.DataFrame,
    *,
    csv_path: Path | None,
    parquet_path: Path | None,
    manifest_path: Path,
    policy_hash: str,
    input_sha256s: dict[str, str],
    totals: dict[str, Any],
    gate_status: GateStatus,
    tool_version: str,
    created_at: str,
) -> dict[str, Any]:
    """Write the requested queue export(s) and the manifest that ties them to
    the policy. At least one of `csv_path`/`parquet_path` is required -- the
    CLI only calls this when at least one was requested; calling it with
    neither is a programming error, not a user-facing one. Returns the
    written manifest dict."""
    if csv_path is None and parquet_path is None:
        raise ValueError("review_queue.export needs at least one of csv_path or parquet_path")

    written: dict[str, Path] = {}
    if csv_path is not None:
        write_csv(review_queue, csv_path)
        written["csv"] = csv_path
    if parquet_path is not None:
        write_parquet(review_queue, parquet_path)
        written["parquet"] = parquet_path

    manifest = build_manifest(
        review_queue,
        policy_hash=policy_hash,
        input_sha256s=input_sha256s,
        totals=totals,
        files=written,
        gate_status=gate_status,
        tool_version=tool_version,
        created_at=created_at,
    )
    write_artifact(manifest, manifest_path)
    return manifest

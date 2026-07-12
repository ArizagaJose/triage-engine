"""Artifact serialization and verification helpers."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_dumps(obj: dict[str, Any]) -> str:
    """Serialize to canonical JSON form: sorted keys, 2-space indent, no NaN, trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n"


def content_hash(obj: dict[str, Any]) -> str:
    """SHA256 hash of canonical JSON, ignoring 'created_at' and 'policy_hash' at top level."""
    obj_copy = obj.copy()
    obj_copy.pop("created_at", None)
    obj_copy.pop("policy_hash", None)
    canonical = canonical_dumps(obj_copy)
    return hashlib.sha256(canonical.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA256 hash of a file's contents."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_artifact(obj: dict[str, Any], path: Path) -> None:
    """Write artifact to path in canonical JSON form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(canonical_dumps(obj))


def utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format, seconds precision, timezone-aware."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

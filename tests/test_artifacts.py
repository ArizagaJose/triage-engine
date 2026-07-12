"""Unit tests for artifacts module."""

import json
import tempfile
from pathlib import Path

from triage.artifacts import canonical_dumps, content_hash, file_sha256, write_artifact


def test_canonical_dumps_sorts_keys() -> None:
    """Test that canonical_dumps sorts keys alphabetically."""
    obj = {"z": 1, "a": 2, "m": 3}
    result = canonical_dumps(obj)
    assert result.startswith('{\n  "a": 2,')
    assert result.endswith("}\n")


def test_canonical_dumps_ends_with_newline() -> None:
    """Test that canonical_dumps ends with newline."""
    obj = {"key": "value"}
    result = canonical_dumps(obj)
    assert result.endswith("\n")


def test_canonical_dumps_indent() -> None:
    """Test that canonical_dumps uses 2-space indent."""
    obj = {"key": {"nested": "value"}}
    result = canonical_dumps(obj)
    assert "  " in result
    lines = result.split("\n")
    assert any(line.startswith("  ") for line in lines)


def test_content_hash_ignores_created_at() -> None:
    """Test that content_hash ignores created_at field."""
    obj1 = {"key": "value", "created_at": "2025-01-01T00:00:00+00:00"}
    obj2 = {"key": "value", "created_at": "2025-12-31T23:59:59+00:00"}
    assert content_hash(obj1) == content_hash(obj2)


def test_content_hash_ignores_policy_hash() -> None:
    """Test that content_hash ignores policy_hash field."""
    obj1 = {"key": "value", "policy_hash": "abc123"}
    obj2 = {"key": "value", "policy_hash": "xyz789"}
    assert content_hash(obj1) == content_hash(obj2)


def test_content_hash_stable() -> None:
    """Test that content_hash is deterministic."""
    obj = {"key": "value", "number": 42}
    hash1 = content_hash(obj)
    hash2 = content_hash(obj)
    assert hash1 == hash2


def test_file_sha256() -> None:
    """Test that file_sha256 correctly computes file hash."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content")
        temp_path = Path(f.name)

    try:
        result = file_sha256(temp_path)
        # Verify it's a valid hex string of correct length
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)
    finally:
        temp_path.unlink()


def test_write_artifact_creates_parent_dirs() -> None:
    """Test that write_artifact creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir1" / "subdir2" / "artifact.json"
        obj = {"key": "value"}
        write_artifact(obj, path)
        assert path.exists()
        assert path.parent.exists()


def test_write_artifact_canonical_format() -> None:
    """Test that write_artifact writes in canonical format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "artifact.json"
        obj = {"z": 1, "a": 2}
        write_artifact(obj, path)

        content = path.read_text()
        # Should be sorted keys and end with newline
        assert content.startswith('{\n  "a": 2,')
        assert content.endswith("}\n")


def test_write_artifact_roundtrip() -> None:
    """Test that write_artifact can be read back as valid JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "artifact.json"
        obj = {"key": "value", "number": 42, "list": [1, 2, 3]}
        write_artifact(obj, path)

        # Read back and verify
        loaded = json.loads(path.read_text())
        assert loaded == obj

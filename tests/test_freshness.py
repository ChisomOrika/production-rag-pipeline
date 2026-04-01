"""Tests for the freshness tracker (unit tests that don't require DB)."""

import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

import xxhash

from src.freshness.tracker import check_file_changed


def test_check_file_changed_no_change():
    """Test detecting no change in a file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("test content")
        f.flush()
        expected_hash = xxhash.xxh64(b"test content").hexdigest()
        changed, current_hash = check_file_changed(f.name, expected_hash)
        assert not changed
        assert current_hash == expected_hash


def test_check_file_changed_with_change():
    """Test detecting a changed file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("new content")
        f.flush()
        old_hash = xxhash.xxh64(b"old content").hexdigest()
        changed, current_hash = check_file_changed(f.name, old_hash)
        assert changed
        assert current_hash != old_hash

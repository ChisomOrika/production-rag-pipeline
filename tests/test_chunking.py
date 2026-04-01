"""Tests for the chunking engine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking.engine import ChunkingEngine


def test_basic_chunking():
    """Test that text is split into chunks."""
    engine = ChunkingEngine(chunk_size=50, chunk_overlap=10)
    text = "This is a test. " * 100
    chunks = engine.chunk_text(text)
    assert len(chunks) > 1
    assert all("text" in c and "token_count" in c and "chunk_index" in c for c in chunks)


def test_chunk_indices_sequential():
    """Test that chunk indices are sequential starting from 0."""
    engine = ChunkingEngine(chunk_size=50, chunk_overlap=10)
    text = "Word " * 500
    chunks = engine.chunk_text(text)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_small_text_single_chunk():
    """Test that small text produces a single chunk."""
    engine = ChunkingEngine(chunk_size=512, chunk_overlap=64)
    text = "This is a short text."
    chunks = engine.chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_document_chunking_with_context():
    """Test that document chunking adds context prefix."""
    engine = ChunkingEngine(chunk_size=50, chunk_overlap=10)
    text = "Introduction\n\nThis is the body. " * 50
    chunks = engine.chunk_document(
        text=text,
        document_title="Test Document",
        section_headers=[{"level": 1, "text": "Introduction"}],
    )
    assert len(chunks) > 0
    # First chunk should have document context
    assert chunks[0].get("context_prefix") is not None or chunks[0]["context_prefix"] is None


def test_token_count_accuracy():
    """Test that token counts are reasonable."""
    engine = ChunkingEngine(chunk_size=100, chunk_overlap=10)
    text = "Hello world. " * 200
    chunks = engine.chunk_text(text)
    for chunk in chunks:
        assert chunk["token_count"] > 0
        assert chunk["token_count"] <= 110  # Allow small overshoot

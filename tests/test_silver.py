"""Tests for the silver layer parsing."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.silver import parse_html, parse_text


def test_parse_html_basic():
    """Test HTML parsing extracts clean text and title."""
    html = """
    <html>
    <head><title>Test Document</title></head>
    <body>
        <h1>Section One</h1>
        <p>This is paragraph one.</p>
        <h2>Subsection</h2>
        <p>This is paragraph two.</p>
        <script>alert('removed')</script>
    </body>
    </html>
    """
    result = parse_html(html)
    assert result["title"] == "Test Document"
    assert "Section One" in result["clean_text"]
    assert "paragraph one" in result["clean_text"]
    assert "alert" not in result["clean_text"]
    assert len(result["headers"]) == 2
    assert result["headers"][0]["text"] == "Section One"
    assert result["headers"][0]["level"] == 1


def test_parse_text_basic():
    """Test plain text parsing."""
    text = "DOCUMENT TITLE\n\nSome content here.\n\nMORE CONTENT\n\nAnother paragraph."
    result = parse_text(text)
    assert result["title"] == "DOCUMENT TITLE"
    assert "Some content here" in result["clean_text"]
    assert len(result["headers"]) >= 1


def test_parse_html_removes_nav():
    """Test that nav/footer/header elements are removed."""
    html = """
    <html><body>
        <nav>Navigation menu</nav>
        <p>Main content</p>
        <footer>Footer stuff</footer>
    </body></html>
    """
    result = parse_html(html)
    assert "Navigation menu" not in result["clean_text"]
    assert "Footer stuff" not in result["clean_text"]
    assert "Main content" in result["clean_text"]


def test_parse_text_markdown_headers():
    """Test markdown-style header detection."""
    text = "# Top Header\n\nContent\n\n## Second Header\n\nMore content"
    result = parse_text(text)
    assert any(h["text"] == "Top Header" for h in result["headers"])

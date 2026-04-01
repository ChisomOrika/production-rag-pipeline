"""Silver layer: document parsing, cleaning, and structural extraction."""

import json
import re
from pathlib import Path

import xxhash
from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from src.db.models import BronzeDocument, DeadLetterDocument, SilverDocument


def parse_html(content: str) -> dict:
    """Parse HTML content into clean text with structural metadata."""
    soup = BeautifulSoup(content, "lxml")

    # Remove scripts, styles, navigation
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract headers for section structure
    headers = []
    for h in soup.find_all(re.compile(r"^h[1-6]$")):
        headers.append({"level": int(h.name[1]), "text": h.get_text(strip=True)})

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else None

    # Clean text extraction
    text = soup.get_text(separator="\n")
    # Collapse excessive whitespace but preserve paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    return {"title": title, "clean_text": text, "headers": headers}


def parse_text(content: str) -> dict:
    """Parse plain text content."""
    lines = content.strip().split("\n")
    title = lines[0].strip() if lines else None

    # Detect headers (lines that are ALL CAPS or short lines followed by content)
    headers = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and stripped.isupper() and len(stripped) < 200:
            headers.append({"level": 1, "text": stripped})
        elif stripped and stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            headers.append({"level": level, "text": stripped.lstrip("# ").strip()})

    return {"title": title, "clean_text": content.strip(), "headers": headers}


def parse_pdf(file_path: str) -> dict:
    """Parse PDF content into clean text."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)

        full_text = "\n\n".join(pages)
        title = pages[0].split("\n")[0].strip() if pages else None

        return {"title": title, "clean_text": full_text, "headers": []}
    except Exception as e:
        raise ValueError(f"PDF parsing failed: {e}")


def process_to_silver(bronze_doc: BronzeDocument, db: Session) -> SilverDocument | None:
    """Parse a bronze document into a clean silver record.

    Args:
        bronze_doc: The BronzeDocument to process.
        db: SQLAlchemy session.

    Returns:
        SilverDocument record, or None if parsing failed.
    """
    try:
        file_path = Path(bronze_doc.file_path)
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            parsed = parse_pdf(str(file_path))
        elif suffix in (".html", ".htm"):
            content = file_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_html(content)
        else:
            content = file_path.read_text(encoding="utf-8", errors="replace")
            parsed = parse_text(content)

        if not parsed["clean_text"] or len(parsed["clean_text"].strip()) < 10:
            raise ValueError("Parsed text is empty or too short")

        content_hash = xxhash.xxh64(parsed["clean_text"].encode()).hexdigest()

        # Check for duplicate
        existing = db.query(SilverDocument).filter(
            SilverDocument.content_hash == content_hash
        ).first()
        if existing:
            return existing

        # Detect document type from content/metadata
        doc_type = _detect_document_type(bronze_doc, parsed)

        silver = SilverDocument(
            bronze_id=bronze_doc.id,
            title=parsed["title"],
            document_type=doc_type,
            clean_text=parsed["clean_text"],
            section_headers=json.dumps(parsed["headers"]),
            content_hash=content_hash,
        )
        db.add(silver)
        db.commit()
        db.refresh(silver)
        return silver

    except Exception as e:
        db.rollback()
        dead = DeadLetterDocument(
            source_url=bronze_doc.source_url,
            file_path=bronze_doc.file_path,
            stage="silver",
            error_message=str(e),
        )
        db.add(dead)
        db.commit()
        return None


def _detect_document_type(bronze: BronzeDocument, parsed: dict) -> str:
    """Heuristic document type detection from content and metadata."""
    text_lower = parsed["clean_text"][:2000].lower()

    if bronze.source_type == "sec_filing":
        if "annual report" in text_lower or "10-k" in text_lower:
            return "10-K"
        elif "quarterly report" in text_lower or "10-q" in text_lower:
            return "10-Q"
        elif "current report" in text_lower or "8-k" in text_lower:
            return "8-K"
        return "sec_filing"

    if any(kw in text_lower for kw in ["regulation", "compliance", "regulatory", "guidance"]):
        return "regulatory_guidance"
    if any(kw in text_lower for kw in ["policy", "procedure", "protocol"]):
        return "policy_document"

    return "general"

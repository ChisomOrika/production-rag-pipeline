"""Bronze layer: raw document ingestion with full provenance tracking."""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from config.settings import settings
from src.db.models import BronzeDocument, DeadLetterDocument


def ingest_to_bronze(file_metadata: dict, db: Session) -> BronzeDocument | None:
    """Move a document from landing zone to bronze layer with full metadata.

    Args:
        file_metadata: dict with keys: source_url, file_path, file_hash,
                       file_size_bytes, download_timestamp, source_type,
                       and optional document_id_external, metadata_json.
        db: SQLAlchemy session.

    Returns:
        BronzeDocument record, or None if it failed (sent to dead letter).
    """
    try:
        # Check for duplicate by file hash — skip if already ingested
        existing = db.query(BronzeDocument).filter(
            BronzeDocument.file_hash == file_metadata["file_hash"]
        ).first()
        if existing:
            return existing

        # Copy from landing to bronze directory
        bronze_dir = Path(settings.bronze_dir)
        bronze_dir.mkdir(parents=True, exist_ok=True)

        src_path = Path(file_metadata["file_path"])
        dest_path = bronze_dir / src_path.name
        shutil.copy2(str(src_path), str(dest_path))

        # Create bronze record
        record = BronzeDocument(
            source_url=file_metadata["source_url"],
            source_type=file_metadata["source_type"],
            file_path=str(dest_path),
            file_hash=file_metadata["file_hash"],
            file_size_bytes=file_metadata.get("file_size_bytes"),
            download_timestamp=datetime.fromisoformat(file_metadata["download_timestamp"]),
            document_id_external=file_metadata.get("document_id_external"),
            metadata_json=json.dumps(file_metadata.get("extra_metadata", {})),
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        return record

    except Exception as e:
        db.rollback()
        dead = DeadLetterDocument(
            source_url=file_metadata.get("source_url"),
            file_path=file_metadata.get("file_path"),
            stage="bronze",
            error_message=str(e),
        )
        db.add(dead)
        db.commit()
        return None

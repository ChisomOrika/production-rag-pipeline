"""Freshness & versioning layer: detects stale embeddings and triggers re-embedding.

The core problem: even when an organization updates documentation, old versions
persist in the vector store. The LLM has no awareness of staleness — it will
confidently synthesize answers from outdated information. This module builds
version tracking into every embedding and ensures stale content is invalidated.

Key interview story: "The pipeline was working perfectly. Row counts matched. No
errors. But answers were wrong because an older version of a policy document was
still in the vector store alongside the updated one, and retrieval kept pulling
the old version because it had more chunks."
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import xxhash
from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from config.settings import settings
from src.db.models import (
    BronzeDocument,
    DocumentVersion,
    GoldChunk,
    PipelineRun,
    SilverDocument,
)

logger = logging.getLogger(__name__)


class FreshnessTracker:
    """Tracks document versions and detects stale embeddings."""

    def detect_source_changes(self, source_url: str, new_file_hash: str,
                              db: Session) -> dict:
        """Check if a source document has changed since last ingestion.

        Returns:
            dict with keys: changed (bool), previous_hash, previous_version,
                           new_version
        """
        latest_version = db.query(DocumentVersion).filter(
            and_(
                DocumentVersion.source_url == source_url,
                DocumentVersion.is_latest == True,
            )
        ).first()

        if latest_version is None:
            # First time seeing this document
            return {
                "changed": True,
                "previous_hash": None,
                "previous_version": 0,
                "new_version": 1,
                "is_new": True,
            }

        if latest_version.file_hash == new_file_hash:
            return {
                "changed": False,
                "previous_hash": latest_version.file_hash,
                "previous_version": latest_version.version,
                "new_version": latest_version.version,
                "is_new": False,
            }

        return {
            "changed": True,
            "previous_hash": latest_version.file_hash,
            "previous_version": latest_version.version,
            "new_version": latest_version.version + 1,
            "is_new": False,
        }

    def register_version(self, source_url: str, file_hash: str,
                         version: int, bronze_id: int,
                         change_summary: str | None, db: Session) -> DocumentVersion:
        """Register a new document version and mark previous versions as non-latest."""
        # Mark all previous versions as not latest
        db.query(DocumentVersion).filter(
            and_(
                DocumentVersion.source_url == source_url,
                DocumentVersion.is_latest == True,
            )
        ).update({"is_latest": False})

        new_version = DocumentVersion(
            source_url=source_url,
            file_hash=file_hash,
            version=version,
            is_latest=True,
            bronze_id=bronze_id,
            change_summary=change_summary,
        )
        db.add(new_version)
        db.commit()
        db.refresh(new_version)
        return new_version

    def invalidate_stale_chunks(self, source_url: str, old_file_hash: str,
                                db: Session) -> int:
        """Mark all chunks from an old document version as stale.

        Returns:
            Number of chunks invalidated.
        """
        now = datetime.now(timezone.utc)

        # Find bronze docs for this source URL with the old hash
        bronze_docs = db.query(BronzeDocument).filter(
            and_(
                BronzeDocument.source_url == source_url,
                BronzeDocument.file_hash == old_file_hash,
            )
        ).all()

        total_invalidated = 0
        for bronze in bronze_docs:
            silver_docs = db.query(SilverDocument).filter(
                SilverDocument.bronze_id == bronze.id
            ).all()

            for silver in silver_docs:
                count = db.query(GoldChunk).filter(
                    and_(
                        GoldChunk.silver_id == silver.id,
                        GoldChunk.is_current == True,
                    )
                ).update({
                    "is_current": False,
                    "invalidated_at": now,
                })
                total_invalidated += count

        db.commit()
        logger.info(f"Invalidated {total_invalidated} stale chunks for {source_url}")
        return total_invalidated

    def reconcile_vector_store(self, db: Session) -> dict:
        """Compare what's in the vector store against source truth.

        Flags drift: chunks that exist in the store but whose source
        document has been superseded, or source documents that have no
        corresponding chunks.

        Returns:
            dict with reconciliation stats and issues found.
        """
        issues = []

        # Find chunks marked current but whose source hash doesn't match latest version
        latest_versions = db.query(DocumentVersion).filter(
            DocumentVersion.is_latest == True
        ).all()

        latest_hashes = {v.source_url: v.file_hash for v in latest_versions}

        # Check all current gold chunks
        stale_chunks = 0
        orphaned_chunks = 0

        current_chunks = db.query(GoldChunk).filter(GoldChunk.is_current == True).all()
        for chunk in current_chunks:
            # Get the source URL via the silver -> bronze chain
            silver = db.query(SilverDocument).get(chunk.silver_id)
            if silver is None:
                orphaned_chunks += 1
                issues.append(f"Orphaned chunk {chunk.id}: no silver document found")
                continue

            bronze = db.query(BronzeDocument).get(silver.bronze_id)
            if bronze is None:
                orphaned_chunks += 1
                issues.append(f"Orphaned chunk {chunk.id}: no bronze document found")
                continue

            expected_hash = latest_hashes.get(bronze.source_url)
            if expected_hash and chunk.source_document_hash != expected_hash:
                stale_chunks += 1
                issues.append(
                    f"Stale chunk {chunk.id}: hash {chunk.source_document_hash} "
                    f"!= latest {expected_hash} for {bronze.source_url}"
                )

        # Find latest versions with no current chunks (missing embeddings)
        missing_embeddings = 0
        for version in latest_versions:
            bronze = db.query(BronzeDocument).filter(
                and_(
                    BronzeDocument.source_url == version.source_url,
                    BronzeDocument.file_hash == version.file_hash,
                )
            ).first()

            if bronze:
                silver = db.query(SilverDocument).filter(
                    SilverDocument.bronze_id == bronze.id
                ).first()
                if silver:
                    chunk_count = db.query(GoldChunk).filter(
                        and_(
                            GoldChunk.silver_id == silver.id,
                            GoldChunk.is_current == True,
                        )
                    ).count()
                    if chunk_count == 0:
                        missing_embeddings += 1
                        issues.append(
                            f"Missing embeddings for latest version of {version.source_url}"
                        )

        return {
            "total_current_chunks": len(current_chunks),
            "stale_chunks_found": stale_chunks,
            "orphaned_chunks": orphaned_chunks,
            "missing_embeddings": missing_embeddings,
            "issues": issues,
            "is_healthy": stale_chunks == 0 and orphaned_chunks == 0 and missing_embeddings == 0,
        }

    def get_freshness_stats(self, db: Session) -> dict:
        """Get freshness statistics for the monitoring dashboard."""
        total_chunks = db.query(GoldChunk).count()
        current_chunks = db.query(GoldChunk).filter(GoldChunk.is_current == True).count()
        stale_chunks = db.query(GoldChunk).filter(GoldChunk.is_current == False).count()

        # Average re-embedding lag
        latest_runs = db.query(PipelineRun).filter(
            PipelineRun.run_type == "freshness_check",
            PipelineRun.status == "completed",
        ).order_by(PipelineRun.completed_at.desc()).limit(10).all()

        avg_lag_seconds = 0
        if latest_runs:
            lags = [(r.completed_at - r.started_at).total_seconds() for r in latest_runs
                    if r.completed_at and r.started_at]
            avg_lag_seconds = sum(lags) / len(lags) if lags else 0

        # Documents by version count
        version_counts = db.query(
            DocumentVersion.source_url,
            func.count(DocumentVersion.id).label("version_count"),
        ).group_by(DocumentVersion.source_url).all()

        multi_version_docs = [v for v in version_counts if v.version_count > 1]

        return {
            "total_chunks": total_chunks,
            "current_chunks": current_chunks,
            "stale_chunks": stale_chunks,
            "freshness_ratio": current_chunks / total_chunks if total_chunks > 0 else 1.0,
            "avg_reembedding_lag_seconds": avg_lag_seconds,
            "multi_version_documents": len(multi_version_docs),
            "total_tracked_documents": len(version_counts),
        }


def check_file_changed(file_path: str, expected_hash: str) -> tuple[bool, str]:
    """Check if a file on disk has changed from its expected hash.

    Returns:
        (changed: bool, current_hash: str)
    """
    content = Path(file_path).read_bytes()
    current_hash = xxhash.xxh64(content).hexdigest()
    return current_hash != expected_hash, current_hash

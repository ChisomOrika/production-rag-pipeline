"""End-to-end ingestion pipeline: landing → bronze → silver → gold.

Orchestrates the full medallion pipeline with freshness tracking.
Decouples each stage so failures are recoverable — if gold embedding fails,
bronze and silver data are intact and the pipeline can replay from that point.
"""

import json
import logging
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from config.settings import settings
from src.chunking.engine import ChunkingEngine, get_chunking_engine
from src.db.models import BronzeDocument, DeadLetterDocument, PipelineRun
from src.embedding.embedder import EmbeddingService, get_embedding_service, process_to_gold
from src.freshness.tracker import FreshnessTracker
from src.ingestion.bronze import ingest_to_bronze
from src.ingestion.silver import process_to_silver
from src.ingestion.sources import LocalFileSource, SECEdgarSource, GenericURLSource

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrates the full ingestion pipeline with freshness guarantees."""

    def __init__(self, db: Session,
                 embedder: EmbeddingService | None = None,
                 chunker: ChunkingEngine | None = None):
        self.db = db
        self.embedder = embedder or get_embedding_service()
        self.chunker = chunker or get_chunking_engine()
        self.freshness = FreshnessTracker()

    def run_full_pipeline(self, file_metadatas: list[dict],
                          run_type: str = "full") -> PipelineRun:
        """Run the complete ingestion pipeline for a batch of documents.

        Args:
            file_metadatas: List of dicts from source fetchers (source_url, file_path, etc.)
            run_type: "full" or "incremental"

        Returns:
            PipelineRun record with stats.
        """
        run = PipelineRun(run_type=run_type)
        self.db.add(run)
        self.db.commit()

        processed = 0
        failed = 0
        stale_detected = 0
        re_embedded = 0

        for file_meta in file_metadatas:
            try:
                result = self._process_single_document(file_meta)
                processed += 1
                if result.get("stale_invalidated", 0) > 0:
                    stale_detected += result["stale_invalidated"]
                if result.get("re_embedded", False):
                    re_embedded += 1
            except Exception as e:
                failed += 1
                logger.error(f"Pipeline failed for {file_meta.get('source_url')}: {e}")

        run.completed_at = datetime.now(timezone.utc)
        run.status = "completed" if failed == 0 else "completed_with_errors"
        run.documents_processed = processed
        run.documents_failed = failed
        run.stale_detected = stale_detected
        run.re_embedded = re_embedded
        self.db.commit()

        logger.info(
            f"Pipeline run {run.id}: {processed} processed, {failed} failed, "
            f"{stale_detected} stale detected, {re_embedded} re-embedded"
        )
        return run

    def _process_single_document(self, file_meta: dict) -> dict:
        """Process one document through the full pipeline."""
        result = {"stale_invalidated": 0, "re_embedded": False}

        # Step 1: Check freshness / detect changes
        change = self.freshness.detect_source_changes(
            source_url=file_meta["source_url"],
            new_file_hash=file_meta["file_hash"],
            db=self.db,
        )

        if not change["changed"]:
            logger.info(f"No changes for {file_meta['source_url']}, skipping")
            return result

        # Step 2: If source changed, invalidate old chunks
        if not change["is_new"] and change["previous_hash"]:
            invalidated = self.freshness.invalidate_stale_chunks(
                source_url=file_meta["source_url"],
                old_file_hash=change["previous_hash"],
                db=self.db,
            )
            result["stale_invalidated"] = invalidated
            logger.info(f"Invalidated {invalidated} stale chunks for {file_meta['source_url']}")

        # Step 3: Bronze — ingest raw document
        bronze = ingest_to_bronze(file_meta, self.db)
        if bronze is None:
            raise RuntimeError(f"Bronze ingestion failed for {file_meta['source_url']}")

        # Step 4: Register new version
        self.freshness.register_version(
            source_url=file_meta["source_url"],
            file_hash=file_meta["file_hash"],
            version=change["new_version"],
            bronze_id=bronze.id,
            change_summary=f"Version {change['new_version']} detected",
            db=self.db,
        )

        # Step 5: Silver — parse and clean
        silver = process_to_silver(bronze, self.db)
        if silver is None:
            raise RuntimeError(f"Silver processing failed for bronze_id={bronze.id}")

        # Step 6: Chunk
        headers = json.loads(silver.section_headers) if silver.section_headers else []
        chunks = self.chunker.chunk_document(
            text=silver.clean_text,
            document_title=silver.title,
            section_headers=headers,
        )

        # Step 7: Gold — embed and store
        gold_chunks = process_to_gold(
            silver_doc=silver,
            chunks=chunks,
            source_file_hash=file_meta["file_hash"],
            document_version=change["new_version"],
            embedder=self.embedder,
            db=self.db,
        )

        if not gold_chunks:
            raise RuntimeError(f"Gold embedding failed for silver_id={silver.id}")

        result["re_embedded"] = True
        logger.info(
            f"Processed {file_meta['source_url']}: "
            f"v{change['new_version']}, {len(gold_chunks)} chunks"
        )
        return result

    def run_freshness_check(self) -> PipelineRun:
        """Run a freshness check without re-ingesting: reconcile vector store."""
        run = PipelineRun(run_type="freshness_check")
        self.db.add(run)
        self.db.commit()

        reconciliation = self.freshness.reconcile_vector_store(self.db)

        run.completed_at = datetime.now(timezone.utc)
        run.status = "completed"
        run.stale_detected = reconciliation["stale_chunks_found"]
        run.error_message = json.dumps(reconciliation["issues"]) if reconciliation["issues"] else None
        self.db.commit()

        return run

    def ingest_from_directory(self, directory: str,
                              source_type: str = "local") -> PipelineRun:
        """Convenience: ingest all supported files from a local directory."""
        source = LocalFileSource()
        file_metadatas = source.ingest_directory(directory, source_type=source_type)
        return self.run_full_pipeline(file_metadatas, run_type="full")

    def ingest_from_urls(self, urls: list[str],
                         source_type: str = "generic") -> PipelineRun:
        """Convenience: ingest documents from URLs."""
        source = GenericURLSource()
        file_metadatas = []
        for url in urls:
            try:
                meta = source.download(url, source_type=source_type)
                file_metadatas.append(meta)
            except Exception as e:
                logger.error(f"Failed to download {url}: {e}")
                dead = DeadLetterDocument(
                    source_url=url, stage="download", error_message=str(e)
                )
                self.db.add(dead)
                self.db.commit()

        return self.run_full_pipeline(file_metadatas, run_type="full")

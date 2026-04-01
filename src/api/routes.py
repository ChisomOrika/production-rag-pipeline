"""API route definitions."""

import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from src.db.models import (
    BronzeDocument, DeadLetterDocument, GoldChunk, PipelineRun,
    RetrievalLog, SilverDocument,
)
from src.db.session import get_db
from src.embedding.embedder import get_embedding_service
from src.freshness.tracker import FreshnessTracker
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid import get_retriever
from src.retrieval.rag import RAGEngine

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Request/Response schemas
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=50)

class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    retrieval_stats: dict

class IngestDirectoryRequest(BaseModel):
    directory: str
    source_type: str = "local"

class IngestURLsRequest(BaseModel):
    urls: list[str]
    source_type: str = "generic"

class PipelineRunResponse(BaseModel):
    run_id: int
    status: str
    documents_processed: int
    documents_failed: int
    stale_detected: int
    re_embedded: int


# ──────────────────────────────────────────────
# Query routes
# ──────────────────────────────────────────────
query_router = APIRouter()

@query_router.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """Ask a question using the RAG pipeline."""
    try:
        engine = RAGEngine(db=db)
        result = engine.query(request.question, top_k=request.top_k)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@query_router.post("/search")
def search_chunks(request: QueryRequest, db: Session = Depends(get_db)):
    """Search for relevant chunks without LLM generation (retrieval only)."""
    try:
        embedder = get_embedding_service()
        retriever = get_retriever(db, embedder)
        results = retriever.search(request.question, top_k=request.top_k)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
# Pipeline routes
# ──────────────────────────────────────────────
pipeline_router = APIRouter()

@pipeline_router.post("/ingest/directory", response_model=PipelineRunResponse)
def ingest_directory(request: IngestDirectoryRequest, db: Session = Depends(get_db)):
    """Ingest documents from a local directory through the full pipeline."""
    try:
        pipeline = IngestionPipeline(db=db)
        run = pipeline.ingest_from_directory(request.directory, request.source_type)
        return PipelineRunResponse(
            run_id=run.id, status=run.status,
            documents_processed=run.documents_processed,
            documents_failed=run.documents_failed,
            stale_detected=run.stale_detected,
            re_embedded=run.re_embedded,
        )
    except Exception as e:
        logger.error(f"Directory ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.post("/ingest/urls", response_model=PipelineRunResponse)
def ingest_urls(request: IngestURLsRequest, db: Session = Depends(get_db)):
    """Ingest documents from URLs through the full pipeline."""
    try:
        pipeline = IngestionPipeline(db=db)
        run = pipeline.ingest_from_urls(request.urls, request.source_type)
        return PipelineRunResponse(
            run_id=run.id, status=run.status,
            documents_processed=run.documents_processed,
            documents_failed=run.documents_failed,
            stale_detected=run.stale_detected,
            re_embedded=run.re_embedded,
        )
    except Exception as e:
        logger.error(f"URL ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.post("/freshness-check")
def run_freshness_check(db: Session = Depends(get_db)):
    """Run a freshness check and reconciliation."""
    try:
        pipeline = IngestionPipeline(db=db)
        run = pipeline.run_freshness_check()
        tracker = FreshnessTracker()
        stats = tracker.get_freshness_stats(db)
        return {"run_id": run.id, "status": run.status, "freshness_stats": stats}
    except Exception as e:
        logger.error(f"Freshness check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@pipeline_router.get("/runs")
def list_pipeline_runs(limit: int = 20, db: Session = Depends(get_db)):
    """List recent pipeline runs."""
    runs = db.query(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id, "run_type": r.run_type, "status": r.status,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "documents_processed": r.documents_processed,
            "documents_failed": r.documents_failed,
            "stale_detected": r.stale_detected,
            "re_embedded": r.re_embedded,
        }
        for r in runs
    ]

@pipeline_router.get("/dead-letters")
def list_dead_letters(resolved: bool = False, limit: int = 50,
                      db: Session = Depends(get_db)):
    """List documents in the dead letter queue."""
    docs = db.query(DeadLetterDocument).filter(
        DeadLetterDocument.resolved == resolved
    ).order_by(DeadLetterDocument.error_timestamp.desc()).limit(limit).all()
    return [
        {
            "id": d.id, "source_url": d.source_url, "stage": d.stage,
            "error_message": d.error_message,
            "error_timestamp": d.error_timestamp.isoformat() if d.error_timestamp else None,
            "retry_count": d.retry_count,
        }
        for d in docs
    ]


# ──────────────────────────────────────────────
# Monitoring routes
# ──────────────────────────────────────────────
monitoring_router = APIRouter()

@monitoring_router.get("/freshness")
def get_freshness_stats(db: Session = Depends(get_db)):
    """Get document freshness statistics."""
    tracker = FreshnessTracker()
    return tracker.get_freshness_stats(db)

@monitoring_router.get("/reconciliation")
def run_reconciliation(db: Session = Depends(get_db)):
    """Run vector store reconciliation check."""
    tracker = FreshnessTracker()
    return tracker.reconcile_vector_store(db)

@monitoring_router.get("/retrieval-logs")
def get_retrieval_logs(limit: int = 50, db: Session = Depends(get_db)):
    """Get recent retrieval logs."""
    logs = db.query(RetrievalLog).order_by(RetrievalLog.timestamp.desc()).limit(limit).all()
    return [
        {
            "id": l.id, "query": l.query, "retrieval_method": l.retrieval_method,
            "served_stale": l.served_stale, "latency_ms": l.latency_ms,
            "timestamp": l.timestamp.isoformat() if l.timestamp else None,
        }
        for l in logs
    ]

@monitoring_router.get("/stats")
def get_pipeline_stats(db: Session = Depends(get_db)):
    """Get overall pipeline statistics."""
    return {
        "bronze_documents": db.query(BronzeDocument).count(),
        "silver_documents": db.query(SilverDocument).count(),
        "gold_chunks_total": db.query(GoldChunk).count(),
        "gold_chunks_current": db.query(GoldChunk).filter(GoldChunk.is_current == True).count(),
        "gold_chunks_stale": db.query(GoldChunk).filter(GoldChunk.is_current == False).count(),
        "dead_letter_unresolved": db.query(DeadLetterDocument).filter(
            DeadLetterDocument.resolved == False
        ).count(),
        "total_pipeline_runs": db.query(PipelineRun).count(),
        "total_retrieval_logs": db.query(RetrievalLog).count(),
    }

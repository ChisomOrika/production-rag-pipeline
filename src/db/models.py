"""SQLAlchemy models for the RAG pipeline."""

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from config.settings import settings


class Base(DeclarativeBase):
    pass


# ──────────────────────────────────────────────
# Bronze layer: raw documents as received
# ──────────────────────────────────────────────
class BronzeDocument(Base):
    __tablename__ = "bronze_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(String(2048), nullable=False)
    source_type = Column(String(100), nullable=False)  # e.g. "sec_filing", "public_health"
    file_path = Column(String(2048), nullable=False)
    file_hash = Column(String(64), nullable=False)  # xxhash of raw content
    file_size_bytes = Column(Integer)
    download_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    document_id_external = Column(String(512))  # e.g. SEC accession number
    metadata_json = Column(Text)  # additional source metadata as JSON

    silver_documents = relationship("SilverDocument", back_populates="bronze_document")

    __table_args__ = (
        Index("ix_bronze_file_hash", "file_hash"),
        Index("ix_bronze_source_url", "source_url"),
    )


# ──────────────────────────────────────────────
# Silver layer: parsed and cleaned
# ──────────────────────────────────────────────
class SilverDocument(Base):
    __tablename__ = "silver_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    bronze_id = Column(Integer, ForeignKey("bronze_documents.id"), nullable=False)
    title = Column(String(1024))
    document_type = Column(String(100))  # e.g. "10-K", "policy_guidance"
    clean_text = Column(Text, nullable=False)
    section_headers = Column(Text)  # JSON list of headers
    content_hash = Column(String(64), nullable=False)
    parsed_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    parsing_errors = Column(Text)  # any issues during parsing

    bronze_document = relationship("BronzeDocument", back_populates="silver_documents")
    gold_chunks = relationship("GoldChunk", back_populates="silver_document")

    __table_args__ = (
        Index("ix_silver_content_hash", "content_hash"),
        Index("ix_silver_bronze_id", "bronze_id"),
    )


# ──────────────────────────────────────────────
# Gold layer: chunked, embedded, ready for retrieval
# ──────────────────────────────────────────────
class GoldChunk(Base):
    __tablename__ = "gold_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    silver_id = Column(Integer, ForeignKey("silver_documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # position in document
    chunk_text = Column(Text, nullable=False)
    token_count = Column(Integer)
    embedding = Column(Vector(settings.embedding_dimensions))
    embedding_model = Column(String(100))
    embedded_timestamp = Column(DateTime(timezone=True))
    chunk_hash = Column(String(64), nullable=False)

    # Freshness tracking
    source_document_hash = Column(String(64), nullable=False)  # from bronze
    document_version = Column(Integer, default=1)
    is_current = Column(Boolean, default=True)
    invalidated_at = Column(DateTime(timezone=True), nullable=True)

    silver_document = relationship("SilverDocument", back_populates="gold_chunks")

    __table_args__ = (
        Index("ix_gold_silver_id", "silver_id"),
        Index("ix_gold_is_current", "is_current"),
        Index("ix_gold_source_hash", "source_document_hash"),
        UniqueConstraint("silver_id", "chunk_index", "document_version", name="uq_chunk_version"),
    )


# ──────────────────────────────────────────────
# Document version registry
# ──────────────────────────────────────────────
class DocumentVersion(Base):
    __tablename__ = "document_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(String(2048), nullable=False)
    file_hash = Column(String(64), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    detected_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    is_latest = Column(Boolean, default=True)
    bronze_id = Column(Integer, ForeignKey("bronze_documents.id"))
    change_summary = Column(Text)  # what changed from previous version

    __table_args__ = (
        Index("ix_docver_source_url", "source_url"),
        Index("ix_docver_is_latest", "is_latest"),
        UniqueConstraint("source_url", "version", name="uq_source_version"),
    )


# ──────────────────────────────────────────────
# Dead letter queue for failed documents
# ──────────────────────────────────────────────
class DeadLetterDocument(Base):
    __tablename__ = "dead_letter_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_url = Column(String(2048))
    file_path = Column(String(2048))
    stage = Column(String(50), nullable=False)  # "bronze", "silver", "gold"
    error_message = Column(Text, nullable=False)
    error_timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    retry_count = Column(Integer, default=0)
    resolved = Column(Boolean, default=False)


# ──────────────────────────────────────────────
# Pipeline run log
# ──────────────────────────────────────────────
class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_type = Column(String(50), nullable=False)  # "full", "incremental", "freshness_check"
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True))
    status = Column(String(20), default="running")  # "running", "completed", "failed"
    documents_processed = Column(Integer, default=0)
    documents_failed = Column(Integer, default=0)
    stale_detected = Column(Integer, default=0)
    re_embedded = Column(Integer, default=0)
    error_message = Column(Text)


# ──────────────────────────────────────────────
# Retrieval log (for evaluation)
# ──────────────────────────────────────────────
class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query = Column(Text, nullable=False)
    retrieved_chunk_ids = Column(Text)  # JSON list of chunk IDs
    retrieval_method = Column(String(50))  # "hybrid", "vector", "bm25"
    vector_scores = Column(Text)  # JSON
    bm25_scores = Column(Text)  # JSON
    fused_scores = Column(Text)  # JSON
    served_stale = Column(Boolean, default=False)  # did we serve stale content?
    response_text = Column(Text)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    latency_ms = Column(Float)


# ──────────────────────────────────────────────
# Evaluation results
# ──────────────────────────────────────────────
class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eval_run_id = Column(String(100), nullable=False)
    question = Column(Text, nullable=False)
    expected_answer = Column(Text)
    actual_answer = Column(Text)
    retrieved_chunks = Column(Text)  # JSON
    retrieval_precision = Column(Float)
    answer_faithfulness = Column(Float)
    freshness_accuracy = Column(Float)
    timestamp = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

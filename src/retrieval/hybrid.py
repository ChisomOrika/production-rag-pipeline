"""Hybrid retrieval: vector search + BM25 + reciprocal rank fusion.

Dense embeddings capture semantic similarity but miss exact identifiers.
BM25 captures keyword/term matches (regulation numbers, policy names).
Combining both with reciprocal rank fusion yields 20-40% higher recall
compared to dense search alone in enterprise deployments.

Interview story: "Pure vector search kept missing documents when users asked
about specific regulation numbers. The embedding captured the topic but not
the exact identifier. Adding BM25 fixed that immediately."
"""

import json
import logging
import time
from datetime import datetime, timezone

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import and_, text
from sqlalchemy.orm import Session

from config.settings import settings
from src.db.models import GoldChunk, RetrievalLog
from src.embedding.embedder import EmbeddingService

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines pgvector similarity search with BM25 keyword matching."""

    def __init__(self, db: Session, embedder: EmbeddingService):
        self.db = db
        self.embedder = embedder
        self.bm25_index: BM25Okapi | None = None
        self.bm25_chunk_ids: list[int] = []
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Build/rebuild the BM25 index from current gold chunks."""
        chunks = self.db.query(GoldChunk.id, GoldChunk.chunk_text).filter(
            GoldChunk.is_current == True
        ).all()

        if not chunks:
            logger.warning("No current chunks found for BM25 index")
            return

        self.bm25_chunk_ids = [c.id for c in chunks]
        tokenized_corpus = [c.chunk_text.lower().split() for c in chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(chunks)} chunks")

    def rebuild_index(self):
        """Force rebuild of BM25 index (call after re-embedding)."""
        self._build_bm25_index()

    def vector_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Perform pgvector cosine similarity search."""
        # Use pgvector's <=> operator for cosine distance
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        results = self.db.execute(
            text("""
                SELECT id, chunk_text, chunk_index, silver_id,
                       source_document_hash, document_version, is_current,
                       1 - (embedding <=> cast(:embedding as vector)) as similarity
                FROM gold_chunks
                WHERE is_current = true
                ORDER BY embedding <=> cast(:embedding as vector)
                LIMIT :limit
            """),
            {"embedding": embedding_str, "limit": top_k},
        ).fetchall()

        return [
            {
                "chunk_id": r.id,
                "chunk_text": r.chunk_text,
                "chunk_index": r.chunk_index,
                "silver_id": r.silver_id,
                "source_document_hash": r.source_document_hash,
                "document_version": r.document_version,
                "is_current": r.is_current,
                "score": float(r.similarity),
                "method": "vector",
            }
            for r in results
        ]

    def bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Perform BM25 keyword search."""
        if self.bm25_index is None or not self.bm25_chunk_ids:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk_id = self.bm25_chunk_ids[idx]
            chunk = self.db.query(GoldChunk).get(chunk_id)
            if chunk and chunk.is_current:
                results.append({
                    "chunk_id": chunk.id,
                    "chunk_text": chunk.chunk_text,
                    "chunk_index": chunk.chunk_index,
                    "silver_id": chunk.silver_id,
                    "source_document_hash": chunk.source_document_hash,
                    "document_version": chunk.document_version,
                    "is_current": chunk.is_current,
                    "score": float(scores[idx]),
                    "method": "bm25",
                })

        return results

    def reciprocal_rank_fusion(self, result_lists: list[list[dict]],
                               k: int = 60) -> list[dict]:
        """Combine multiple ranked lists using reciprocal rank fusion.

        RRF score = sum(1 / (k + rank_i)) for each list where the item appears.
        k=60 is standard and dampens the effect of high ranks.
        """
        fused_scores: dict[int, float] = {}
        chunk_data: dict[int, dict] = {}

        for results in result_lists:
            for rank, result in enumerate(results):
                chunk_id = result["chunk_id"]
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
                chunk_data[chunk_id] = result

        # Sort by fused score descending
        sorted_ids = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)

        fused_results = []
        for chunk_id in sorted_ids:
            result = chunk_data[chunk_id].copy()
            result["fused_score"] = fused_scores[chunk_id]
            result["method"] = "hybrid"
            fused_results.append(result)

        return fused_results

    def search(self, query: str, top_k: int | None = None,
               vector_weight: float | None = None,
               bm25_weight: float | None = None,
               log_retrieval: bool = True) -> list[dict]:
        """Perform hybrid search combining vector and BM25 results.

        Args:
            query: The search query.
            top_k: Number of results to return.
            vector_weight: Weight for vector results (unused in RRF, kept for logging).
            bm25_weight: Weight for BM25 results (unused in RRF, kept for logging).
            log_retrieval: Whether to log the retrieval for evaluation.

        Returns:
            List of result dicts sorted by fused relevance score.
        """
        start_time = time.time()
        top_k = top_k or settings.top_k

        # Generate query embedding
        query_embedding = self.embedder.embed_single(query)

        # Get results from both methods (fetch more than top_k for better fusion)
        vector_results = self.vector_search(query_embedding, top_k=top_k * 2)
        bm25_results = self.bm25_search(query, top_k=top_k * 2)

        # Fuse results
        fused = self.reciprocal_rank_fusion([vector_results, bm25_results])
        final_results = fused[:top_k]

        # Check if any results are stale (shouldn't happen if is_current filter works)
        served_stale = any(not r.get("is_current", True) for r in final_results)

        latency_ms = (time.time() - start_time) * 1000

        # Log retrieval
        if log_retrieval:
            log = RetrievalLog(
                query=query,
                retrieved_chunk_ids=json.dumps([r["chunk_id"] for r in final_results]),
                retrieval_method="hybrid",
                vector_scores=json.dumps([{"id": r["chunk_id"], "score": r["score"]}
                                          for r in vector_results[:top_k]]),
                bm25_scores=json.dumps([{"id": r["chunk_id"], "score": r["score"]}
                                        for r in bm25_results[:top_k]]),
                fused_scores=json.dumps([{"id": r["chunk_id"], "score": r["fused_score"]}
                                         for r in final_results]),
                served_stale=served_stale,
                latency_ms=latency_ms,
            )
            self.db.add(log)
            self.db.commit()

        if served_stale:
            logger.warning(f"Served stale content for query: {query[:100]}...")

        return final_results


def get_retriever(db: Session, embedder: EmbeddingService | None = None) -> HybridRetriever:
    """Factory function for hybrid retriever."""
    if embedder is None:
        embedder = EmbeddingService()
    return HybridRetriever(db=db, embedder=embedder)

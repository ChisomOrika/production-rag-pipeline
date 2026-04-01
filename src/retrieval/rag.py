"""RAG query engine: retrieves context and generates answers with the LLM."""

import json
import logging
import time

from openai import OpenAI
from sqlalchemy.orm import Session

from config.settings import settings
from src.db.models import RetrievalLog, BronzeDocument, SilverDocument, GoldChunk
from src.embedding.embedder import EmbeddingService
from src.retrieval.hybrid import HybridRetriever, get_retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise, factual assistant that answers questions based ONLY on the provided context documents.

Rules:
1. Only use information from the provided context to answer questions.
2. If the context doesn't contain enough information to answer, say so explicitly.
3. When citing information, reference the document title and section when available.
4. Never make up or infer information not present in the context.
5. If multiple documents provide conflicting information, note the conflict and cite both sources.
6. Pay attention to document versions — prefer information from the most recent version."""


class RAGEngine:
    """Retrieval-Augmented Generation engine."""

    def __init__(self, db: Session,
                 retriever: HybridRetriever | None = None,
                 embedder: EmbeddingService | None = None):
        self.db = db
        self.embedder = embedder or EmbeddingService()
        self.retriever = retriever or get_retriever(db, self.embedder)
        self.llm_client = OpenAI(api_key=settings.openai_api_key)

    def query(self, question: str, top_k: int | None = None) -> dict:
        """Answer a question using RAG.

        Returns:
            dict with keys: answer, sources, retrieval_stats
        """
        start_time = time.time()
        top_k = top_k or settings.top_k

        # Retrieve relevant chunks
        results = self.retriever.search(question, top_k=top_k)

        if not results:
            return {
                "answer": "I couldn't find any relevant documents to answer this question.",
                "sources": [],
                "retrieval_stats": {"chunks_retrieved": 0},
            }

        # Build context from retrieved chunks with source info
        context_parts = []
        sources = []
        for i, result in enumerate(results):
            # Get document metadata
            chunk = self.db.query(GoldChunk).get(result["chunk_id"])
            silver = self.db.query(SilverDocument).get(result["silver_id"]) if chunk else None
            bronze = self.db.query(BronzeDocument).get(silver.bronze_id) if silver else None

            doc_info = {
                "chunk_id": result["chunk_id"],
                "document_title": silver.title if silver else "Unknown",
                "document_type": silver.document_type if silver else "Unknown",
                "source_url": bronze.source_url if bronze else "Unknown",
                "version": result.get("document_version", 1),
                "relevance_score": result.get("fused_score", result.get("score", 0)),
            }
            sources.append(doc_info)

            header = f"[Document {i + 1}: {doc_info['document_title']} (v{doc_info['version']})]"
            context_parts.append(f"{header}\n{result['chunk_text']}")

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer
        response = self.llm_client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            temperature=0.1,
            max_tokens=1024,
        )

        answer = response.choices[0].message.content
        latency_ms = (time.time() - start_time) * 1000

        # Update retrieval log with the response
        latest_log = self.db.query(RetrievalLog).order_by(
            RetrievalLog.id.desc()
        ).first()
        if latest_log and latest_log.query == question:
            latest_log.response_text = answer
            latest_log.latency_ms = latency_ms
            self.db.commit()

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_stats": {
                "chunks_retrieved": len(results),
                "latency_ms": latency_ms,
                "top_score": max(r.get("fused_score", r.get("score", 0)) for r in results),
            },
        }

"""Embedding layer: generates and stores vector embeddings for gold chunks.

Design decision: using OpenAI text-embedding-3-small for cost efficiency.
Changing embedding models requires re-embedding the entire corpus — this is
a cascading decision that should be made deliberately, not incrementally.
"""

import json
from datetime import datetime, timezone

import xxhash
from openai import OpenAI
from sqlalchemy.orm import Session

from config.settings import settings
from src.db.models import DeadLetterDocument, GoldChunk, SilverDocument


class EmbeddingService:
    """Generates embeddings via OpenAI API and stores them in pgvector."""

    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        OpenAI supports up to 2048 texts per batch. We batch in groups of 100
        to balance throughput and error recovery granularity.
        """
        all_embeddings = []
        batch_size = 100

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
                dimensions=self.dimensions,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        return self.embed_texts([text])[0]


def process_to_gold(silver_doc: SilverDocument, chunks: list[dict],
                    source_file_hash: str, document_version: int,
                    embedder: EmbeddingService, db: Session) -> list[GoldChunk]:
    """Embed chunks and store as gold records with freshness metadata.

    Args:
        silver_doc: The source SilverDocument.
        chunks: Output from ChunkingEngine.chunk_document().
        source_file_hash: Bronze document file hash (for freshness tracking).
        document_version: Current version number of this document.
        embedder: EmbeddingService instance.
        db: SQLAlchemy session.

    Returns:
        List of created GoldChunk records.
    """
    gold_chunks = []

    try:
        # Prepare texts for batch embedding
        texts_to_embed = []
        for chunk in chunks:
            # Prepend context prefix for embedding if available
            embed_text = chunk["text"]
            if chunk.get("context_prefix"):
                embed_text = f"{chunk['context_prefix']}\n\n{chunk['text']}"
            texts_to_embed.append(embed_text)

        # Batch embed
        embeddings = embedder.embed_texts(texts_to_embed)

        now = datetime.now(timezone.utc)

        for chunk, embedding in zip(chunks, embeddings):
            chunk_hash = xxhash.xxh64(chunk["text"].encode()).hexdigest()

            gold = GoldChunk(
                silver_id=silver_doc.id,
                chunk_index=chunk["chunk_index"],
                chunk_text=chunk["text"],
                token_count=chunk["token_count"],
                embedding=embedding,
                embedding_model=settings.embedding_model,
                embedded_timestamp=now,
                chunk_hash=chunk_hash,
                source_document_hash=source_file_hash,
                document_version=document_version,
                is_current=True,
            )
            db.add(gold)
            gold_chunks.append(gold)

        db.commit()
        for g in gold_chunks:
            db.refresh(g)

        return gold_chunks

    except Exception as e:
        db.rollback()
        dead = DeadLetterDocument(
            source_url=f"silver_id:{silver_doc.id}",
            stage="gold",
            error_message=str(e),
        )
        db.add(dead)
        db.commit()
        return []


def get_embedding_service() -> EmbeddingService:
    """Factory function for embedding service."""
    return EmbeddingService()

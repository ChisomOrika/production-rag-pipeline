"""Chunking engine: recursive character splitting at 512 tokens with overlap.

Design decision: after experimentation, recursive character splitting at 512 tokens
consistently outperforms semantic chunking on retrieval accuracy. Smaller chunks (128)
split mid-concept and cause hallucination spikes. Larger chunks (1024+) dilute relevance.

512 tokens with 10-20% overlap (64 tokens) is the sweet spot for regulatory documents.
"""

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings


class ChunkingEngine:
    """Produces text chunks optimized for RAG retrieval."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None,
                 model_name: str = "cl100k_base"):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.encoding = tiktoken.get_encoding(model_name)

        # LangChain splitter uses character count by default.
        # We use a token-aware length function so "512" means 512 tokens.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def _token_length(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chunk_text(self, text: str) -> list[dict]:
        """Split text into chunks with metadata.

        Returns:
            List of dicts with keys: text, token_count, chunk_index
        """
        raw_chunks = self.splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(raw_chunks):
            token_count = self._token_length(chunk_text)
            chunks.append({
                "text": chunk_text,
                "token_count": token_count,
                "chunk_index": i,
            })

        return chunks

    def chunk_document(self, text: str, document_title: str | None = None,
                       section_headers: list[dict] | None = None) -> list[dict]:
        """Chunk a full document, enriching each chunk with positional context.

        This adds a context prefix to help the LLM understand where the chunk
        sits within the document — critical for regulatory docs where section
        context changes interpretation.
        """
        chunks = self.chunk_text(text)

        # Enrich with document-level context
        for chunk in chunks:
            context_parts = []
            if document_title:
                context_parts.append(f"Document: {document_title}")

            # Find the most recent section header before this chunk's position
            if section_headers:
                # Approximate: find headers that appear in text before this chunk
                chunk_start = text.find(chunk["text"][:50])
                if chunk_start >= 0:
                    preceding_text = text[:chunk_start]
                    for header in reversed(section_headers):
                        if header["text"] in preceding_text:
                            context_parts.append(f"Section: {header['text']}")
                            break

            chunk["context_prefix"] = " | ".join(context_parts) if context_parts else None

        return chunks


def get_chunking_engine() -> ChunkingEngine:
    """Factory function for the default chunking engine."""
    return ChunkingEngine()

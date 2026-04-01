"""Evaluation framework: measures retrieval precision, answer faithfulness, freshness accuracy.

A bad RAG system is worse than no RAG at all. Once users decide a system can't be
trusted, they don't keep checking back. This module provides the measurement
infrastructure to catch quality degradation before users do.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from openai import OpenAI
from sqlalchemy.orm import Session

from config.settings import settings
from src.db.models import EvaluationResult, GoldChunk, DocumentVersion
from src.embedding.embedder import EmbeddingService
from src.retrieval.hybrid import get_retriever
from src.retrieval.rag import RAGEngine

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluates RAG pipeline quality across three dimensions:
    1. Retrieval precision — did we pull the right chunks?
    2. Answer faithfulness — did the LLM stick to what was retrieved?
    3. Freshness accuracy — did we pull from the current version?
    """

    def __init__(self, db: Session):
        self.db = db
        self.llm = OpenAI(api_key=settings.openai_api_key)

    def evaluate_test_set(self, test_set: list[dict]) -> dict:
        """Run evaluation against a test set.

        Args:
            test_set: List of dicts with keys:
                - question: str
                - expected_answer: str
                - expected_source_urls: list[str] (optional)
                - expected_version: int (optional)

        Returns:
            Aggregate eval stats and per-question results.
        """
        eval_run_id = str(uuid.uuid4())[:8]
        results = []

        rag = RAGEngine(db=self.db)

        for item in test_set:
            try:
                result = self._evaluate_single(item, rag, eval_run_id)
                results.append(result)
            except Exception as e:
                logger.error(f"Eval failed for question: {item['question'][:80]}: {e}")
                results.append({
                    "question": item["question"],
                    "error": str(e),
                    "retrieval_precision": 0.0,
                    "answer_faithfulness": 0.0,
                    "freshness_accuracy": 0.0,
                })

        # Aggregate
        n = len(results)
        valid = [r for r in results if "error" not in r]
        avg_precision = sum(r["retrieval_precision"] for r in valid) / len(valid) if valid else 0
        avg_faithfulness = sum(r["answer_faithfulness"] for r in valid) / len(valid) if valid else 0
        avg_freshness = sum(r["freshness_accuracy"] for r in valid) / len(valid) if valid else 0

        return {
            "eval_run_id": eval_run_id,
            "total_questions": n,
            "successful_evaluations": len(valid),
            "failed_evaluations": n - len(valid),
            "avg_retrieval_precision": round(avg_precision, 4),
            "avg_answer_faithfulness": round(avg_faithfulness, 4),
            "avg_freshness_accuracy": round(avg_freshness, 4),
            "results": results,
        }

    def _evaluate_single(self, item: dict, rag: RAGEngine, eval_run_id: str) -> dict:
        """Evaluate a single question."""
        question = item["question"]
        expected_answer = item.get("expected_answer", "")

        # Get RAG response
        rag_result = rag.query(question)
        actual_answer = rag_result["answer"]
        sources = rag_result["sources"]

        # 1. Retrieval precision: check if expected sources are in retrieved sources
        retrieval_precision = self._score_retrieval_precision(
            sources, item.get("expected_source_urls", [])
        )

        # 2. Answer faithfulness: LLM-as-judge
        answer_faithfulness = self._score_faithfulness(
            question, actual_answer, [s.get("chunk_text", "") for s in sources] if sources else []
        )

        # 3. Freshness accuracy: check if all retrieved chunks are current version
        freshness_accuracy = self._score_freshness(sources)

        # Store result
        eval_record = EvaluationResult(
            eval_run_id=eval_run_id,
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            retrieved_chunks=json.dumps([s.get("chunk_id") for s in sources]),
            retrieval_precision=retrieval_precision,
            answer_faithfulness=answer_faithfulness,
            freshness_accuracy=freshness_accuracy,
        )
        self.db.add(eval_record)
        self.db.commit()

        return {
            "question": question,
            "actual_answer": actual_answer,
            "retrieval_precision": retrieval_precision,
            "answer_faithfulness": answer_faithfulness,
            "freshness_accuracy": freshness_accuracy,
            "sources_count": len(sources),
        }

    def _score_retrieval_precision(self, sources: list[dict],
                                   expected_urls: list[str]) -> float:
        """Score whether retrieved sources include expected documents."""
        if not expected_urls:
            return 1.0 if sources else 0.0

        retrieved_urls = {s.get("source_url", "") for s in sources}
        hits = sum(1 for url in expected_urls if url in retrieved_urls)
        return hits / len(expected_urls) if expected_urls else 0.0

    def _score_faithfulness(self, question: str, answer: str,
                            context_texts: list[str]) -> float:
        """Use LLM-as-judge to score whether the answer is faithful to retrieved context."""
        if not context_texts:
            return 0.0

        context = "\n---\n".join(context_texts[:5])  # Limit context for cost

        response = self.llm.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": (
                    "You are an evaluation judge. Score whether the answer is "
                    "FAITHFUL to the provided context — meaning the answer only "
                    "contains information that can be found in or directly inferred "
                    "from the context. Score from 0.0 to 1.0.\n\n"
                    "Respond with ONLY a number between 0.0 and 1.0."
                )},
                {"role": "user", "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}\n\n"
                    f"Answer: {answer}\n\n"
                    f"Faithfulness score (0.0-1.0):"
                )},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            logger.warning("Failed to parse faithfulness score, defaulting to 0.5")
            return 0.5

    def _score_freshness(self, sources: list[dict]) -> float:
        """Score whether all retrieved chunks are from the latest document version."""
        if not sources:
            return 1.0

        fresh_count = 0
        for source in sources:
            chunk_id = source.get("chunk_id")
            if chunk_id:
                chunk = self.db.query(GoldChunk).get(chunk_id)
                if chunk and chunk.is_current:
                    fresh_count += 1

        return fresh_count / len(sources)


def load_test_set(file_path: str) -> list[dict]:
    """Load evaluation test set from a JSON file."""
    with open(file_path) as f:
        return json.load(f)

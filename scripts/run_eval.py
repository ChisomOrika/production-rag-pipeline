#!/usr/bin/env python3
"""CLI to run the evaluation framework."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import SessionLocal, init_db
from src.evaluation.evaluator import RAGEvaluator, load_test_set


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--test-set", default="eval_data/test_set.json",
                        help="Path to test set JSON")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    init_db()
    db = SessionLocal()

    try:
        test_set = load_test_set(args.test_set)
        print(f"Loaded {len(test_set)} test questions")

        evaluator = RAGEvaluator(db=db)
        results = evaluator.evaluate_test_set(test_set)

        print(f"\n{'=' * 60}")
        print(f"Evaluation Run: {results['eval_run_id']}")
        print(f"Questions: {results['total_questions']}")
        print(f"Successful: {results['successful_evaluations']}")
        print(f"Failed: {results['failed_evaluations']}")
        print(f"{'=' * 60}")
        print(f"Avg Retrieval Precision:  {results['avg_retrieval_precision']:.2%}")
        print(f"Avg Answer Faithfulness:  {results['avg_answer_faithfulness']:.2%}")
        print(f"Avg Freshness Accuracy:   {results['avg_freshness_accuracy']:.2%}")
        print(f"{'=' * 60}")

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")

    finally:
        db.close()


if __name__ == "__main__":
    main()

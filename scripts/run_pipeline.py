#!/usr/bin/env python3
"""CLI to run the ingestion pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import SessionLocal, init_db
from src.ingestion.pipeline import IngestionPipeline


def main():
    parser = argparse.ArgumentParser(description="Run the RAG ingestion pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest from directory
    dir_parser = subparsers.add_parser("directory", help="Ingest from local directory")
    dir_parser.add_argument("path", help="Directory path")
    dir_parser.add_argument("--source-type", default="local")

    # Ingest from URLs
    url_parser = subparsers.add_parser("urls", help="Ingest from URLs")
    url_parser.add_argument("urls", nargs="+", help="URLs to ingest")
    url_parser.add_argument("--source-type", default="generic")

    # Freshness check
    subparsers.add_parser("freshness-check", help="Run freshness check")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    init_db()
    db = SessionLocal()

    try:
        pipeline = IngestionPipeline(db=db)

        if args.command == "directory":
            run = pipeline.ingest_from_directory(args.path, args.source_type)
            print(f"Pipeline run {run.id}: {run.documents_processed} processed, "
                  f"{run.documents_failed} failed, {run.stale_detected} stale detected")

        elif args.command == "urls":
            run = pipeline.ingest_from_urls(args.urls, args.source_type)
            print(f"Pipeline run {run.id}: {run.documents_processed} processed, "
                  f"{run.documents_failed} failed, {run.stale_detected} stale detected")

        elif args.command == "freshness-check":
            run = pipeline.run_freshness_check()
            print(f"Freshness check {run.id}: {run.stale_detected} stale detected")

    finally:
        db.close()


if __name__ == "__main__":
    main()

# Production RAG Pipeline with Data Freshness Guarantees

Most RAG systems break silently in production — not from bad retrieval, but from stale data. Source documents get updated, old versions stay in the vector store, and the LLM confidently generates answers from outdated information with zero indication anything is wrong.

This pipeline treats RAG as a **data engineering problem**. Every embedding tracks which version of which document it came from. When sources change, stale embeddings are invalidated automatically — before users ever see bad answers. Built for SEC filings and regulatory documents where staleness has real consequences.

## Architecture

```
  Sources (SEC EDGAR / URLs / Local)
         │
         ▼
  ┌─────────────────────────────────────┐
  │  Medallion Pipeline                  │
  │  Landing → Bronze → Silver → Gold    │
  │  (raw)    (tracked) (parsed) (embed) │
  │                                      │
  │  Failed docs → Dead Letter Queue     │
  └──────────────────┬──────────────────┘
                     │
          ┌──────────▼──────────┐
          │  pgvector (Postgres) │ ← embeddings + metadata
          │  single database     │   in one place, no sync
          └───┬─────────────┬───┘
              ▼             ▼
        Vector Search   BM25 Search
        (semantic)      (exact terms)
              └──────┬──────┘
                     ▼
          Reciprocal Rank Fusion
                     ▼
             LLM Generation
          (grounded in context)

  ┌─────────────────────────────────────┐
  │  FRESHNESS LAYER (continuous)        │
  │  Hash-based change detection         │
  │  Automatic stale invalidation        │
  │  Vector store ↔ source reconciliation│
  └─────────────────────────────────────┘
```

## What I Learned Building This

**The stale document problem.** Pipeline was working. Row counts matched. No errors. But answers were wrong — an older version of a policy document was still in the vector store alongside the updated one, and retrieval kept pulling the old version because it had more chunks. I had to build document version *invalidation*, not just document addition.

**The chunking trap.** Started with semantic chunking because it sounded smarter. Retrieval quality dropped. Switched to recursive splitting at 512 tokens with ~12% overlap — precision improved. 128-token chunks split mid-concept and spiked hallucinations. Boring and predictable is what you want in a production data pipeline.

**The keyword gap.** Pure vector search missed documents when users asked for specific regulation numbers. The embedding captured the *topic* but not the *exact identifier*. Adding BM25 and fusing with reciprocal rank fusion fixed it immediately — 20-40% higher recall.

## Key Tradeoffs

| Decision | Reasoning |
|----------|-----------|
| **pgvector** over dedicated vector DB | One database for embeddings + metadata = no sync drift. At this scale, operational simplicity wins. Would reconsider at millions of vectors. |
| **512-token recursive chunks** over semantic | Higher retrieval accuracy in testing. Semantic chunking produced inconsistent sizes and lost context. |
| **Hybrid retrieval** (vector + BM25) | Dense search misses exact identifiers. BM25 catches them. Worth maintaining two indices for regulatory data. |
| **Medallion pipeline** | If embedding fails, bronze/silver survive. Replay from any stage. Dead letter queue catches failures. |
| **Version tracking per embedding** | Without invalidation, old versions accumulate and silently degrade answers. |

## What I'd Do Differently

- **Embedding model swaps** require re-embedding the entire corpus. Would build a migration abstraction for this.
- **BM25 index** is rebuilt from scratch on init. Would persist and update incrementally at scale.
- **Chunk boundaries** occasionally break inside tables. Format-aware splitting would help.
- **Streaming re-embedding** (CDC-triggered) would shrink the stale content window vs. current batch approach.
- At **millions of documents**, pgvector's scan performance becomes a bottleneck — would migrate to a dedicated vector store.

## Quick Start

```bash
cp .env.example .env        # Add your OPENAI_API_KEY
docker compose up -d         # API at :8000, Dashboard at :8501
```

Or locally:
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=. python scripts/init_db.py
PYTHONPATH=. python scripts/run_pipeline.py directory ./docs
PYTHONPATH=. uvicorn src.api.app:app --reload
PYTHONPATH=. streamlit run src/dashboard/app.py
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Full RAG query |
| `POST` | `/api/v1/search` | Retrieval only (no LLM) |
| `POST` | `/api/v1/pipeline/ingest/directory` | Ingest local files |
| `POST` | `/api/v1/pipeline/ingest/urls` | Ingest from URLs |
| `POST` | `/api/v1/pipeline/freshness-check` | Run freshness reconciliation |
| `GET` | `/api/v1/monitoring/stats` | Pipeline statistics |
| `GET` | `/api/v1/monitoring/freshness` | Freshness metrics |
| `GET` | `/api/v1/monitoring/reconciliation` | Vector store drift check |

## Evaluation

Measures three dimensions: **retrieval precision** (right chunks?), **answer faithfulness** (LLM stayed grounded?), **freshness accuracy** (current version served?).

```bash
PYTHONPATH=. python scripts/run_eval.py --test-set eval_data/test_set.json
```

## Tech Stack

Python, FastAPI, PostgreSQL + pgvector, OpenAI (text-embedding-3-small + GPT-4o-mini), LangChain text splitters, rank-bm25, APScheduler, Streamlit + Plotly, xxhash.

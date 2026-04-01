# Production RAG Pipeline with Data Freshness Guarantees

## The Problem

Most RAG tutorials end where production problems begin. They show you how to embed documents and query a vector database. What they don't show you is what happens six weeks later when the source documents get updated, the old versions are still sitting in your vector store, and your system is confidently generating answers from outdated regulatory guidance — with zero indication that anything is wrong.

**RAG has a data engineering problem most AI engineers don't see coming.**

I built this pipeline because I recognized that the dominant failure modes in production RAG aren't AI problems — they're the same problems that have plagued data pipelines for decades: stale data, broken ingestion, schema drift, and the absence of contracts between data producers and consumers. The vector database is just a new place for old problems to hide.

This system ingests SEC filings and regulatory documents — sources that get updated, superseded, and sometimes contradict older versions. It treats document freshness as a first-class concern, not an afterthought. Every embedding knows which version of which document it came from, when that document was last verified, and whether it's still the current version.

---

## Architecture

```
                         ┌──────────────────────────┐
                         │   Document Sources        │
                         │   SEC EDGAR / URLs / Local │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │     Landing Zone          │
                         │  (decouples ingestion     │
                         │   from processing —       │
                         │   if embedding fails,     │
                         │   raw data survives)      │
                         └────────────┬─────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
     ┌────────────────┐    ┌──────────────────┐    ┌──────────────────┐
     │    BRONZE       │    │     SILVER        │    │      GOLD        │
     │  Raw documents  │───►│  Parsed, cleaned  │───►│  Chunked &       │
     │  + provenance   │    │  + structural     │    │  embedded        │
     │  (source URL,   │    │  metadata         │    │  (512-token      │
     │   file hash,    │    │  (headers,        │    │   recursive      │
     │   timestamp,    │    │   doc type,       │    │   splits +       │
     │   version)      │    │   sections)       │    │   version tags)  │
     └────────────────┘    └──────────────────┘    └────────┬─────────┘
              │                                              │
              │         ┌──────────────────┐                 │
              └────────►│  Dead Letter      │◄───────────────┘
                        │  Queue            │  (failed docs at any stage
                        │                   │   are captured, not lost)
                        └──────────────────┘
                                                    │
                    ┌───────────────────────────────┘
                    ▼
     ┌──────────────────────────────────────────────────────┐
     │                    pgvector                           │
     │  Embeddings + metadata + versions in ONE database     │
     │  (no sync issues between separate systems)            │
     └───────────────┬──────────────────────┬───────────────┘
                     │                      │
              ┌──────▼───────┐      ┌───────▼──────┐
              │ Vector Search │      │  BM25 Search  │
              │ (semantic     │      │  (exact terms, │
              │  similarity)  │      │   regulation   │
              │              │      │   numbers)     │
              └──────┬───────┘      └───────┬──────┘
                     └──────────┬───────────┘
                                ▼
                  ┌──────────────────────────┐
                  │  Reciprocal Rank Fusion   │
                  │  (combined ranking)       │
                  └────────────┬─────────────┘
                               ▼
                  ┌──────────────────────────┐
                  │  LLM Generation          │
                  │  (grounded in retrieved   │
                  │   context only)           │
                  └──────────────────────────┘

    ╔══════════════════════════════════════════════════════════╗
    ║  FRESHNESS LAYER (runs continuously)                     ║
    ║  • Detects source changes via file hash comparison        ║
    ║  • Invalidates stale embeddings automatically             ║
    ║  • Reconciliation check: vector store vs. source truth    ║
    ║  • Logs every time the system would have served stale     ║
    ╚══════════════════════════════════════════════════════════╝

    ╔══════════════════════════════════════════════════════════╗
    ║  MONITORING DASHBOARD (Streamlit)                        ║
    ║  • Documents by freshness status (current/stale/flagged)  ║
    ║  • Retrieval quality scores over time                     ║
    ║  • Re-embedding lag (source change → index update)        ║
    ║  • Stale content serving log                              ║
    ╚══════════════════════════════════════════════════════════╝
```

---

## The Story Behind This Build

### RAG has a data engineering problem

When I started building this, I expected the hard part to be retrieval tuning or prompt engineering. It wasn't. The hard part was everything that happens *before* a query ever hits the vector database.

**The stale document problem.** The pipeline was working perfectly. Row counts matched. No errors. Retrieval latency was good. But the answers were wrong. An older version of a policy document was still in the vector store alongside the updated one, and the retrieval kept pulling the old version because it had more chunks indexed. The LLM had no idea it was synthesizing from outdated information — it answered with full confidence.

This is the same pattern that breaks traditional data pipelines: a renamed field, a schema change, a source update that doesn't propagate. The vector database is just a new surface for the same class of failure. I had to build document version invalidation — not just document addition. When a source changes, every embedding tied to the old version gets flagged as stale and excluded from retrieval before the new version is even embedded.

**The chunking trap.** I started with semantic chunking because it sounded like the right approach — chunk by meaning, not by arbitrary character boundaries. It wasn't. My retrieval quality actually dropped. The semantic chunker was producing chunks of wildly inconsistent sizes, and the small ones were losing context that the embedding model needed to place them correctly. I switched to recursive character splitting at 512 tokens with ~12% overlap, and retrieval precision improved measurably.

The counterintuitive part: 128-token chunks seemed like they'd give better granularity. They didn't. They split mid-concept. The LLM started receiving sentence fragments without the surrounding context that tied them together, and the hallucination rate spiked. I couldn't figure out why until I audited the actual chunk boundaries.

**The keyword gap.** Pure vector search kept missing documents when users asked about specific regulation numbers or filing identifiers. The embedding captured the *topic* of the regulation but not the *exact identifier*. A user asking about "Rule 10b-5" would get results about insider trading generally, but not the specific rule they needed. Adding BM25 keyword search and fusing results with reciprocal rank fusion fixed this immediately. The tradeoff is maintaining two indices, but for regulatory data where precision matters, it's worth it.

---

## Challenges and Tradeoffs

### Why pgvector instead of a dedicated vector database

I chose pgvector because at this scale, the operational simplicity of one database outweighed the performance benefits of a dedicated vector store. Embeddings, document versions, freshness timestamps, lineage records, and retrieval logs all live in PostgreSQL. There's no sync layer between two systems that can drift apart.

The tradeoff: pgvector can't match Pinecone or Weaviate on sub-50ms latency at millions of vectors. If this system needed to serve 10M+ embeddings at low latency, I'd reconsider. But for a corpus of regulatory documents in the tens-of-thousands range, pgvector handles it comfortably, and the operational cost of running one database instead of two is significant.

### Why recursive splitting over semantic chunking

A 2026 comparison of chunking approaches found that recursive character splitting at 512 tokens achieves the highest answer accuracy, consistently outperforming semantic chunking by a meaningful margin. My own testing confirmed this on regulatory documents.

Semantic chunking sounds smarter. It produces "meaningful" segments. But in practice, it creates wildly variable chunk sizes, and the small chunks lose the context that embeddings need. Recursive splitting with overlap is boring and predictable — which is exactly what you want in a production data pipeline.

### Why hybrid retrieval matters for this domain

Dense embeddings are great at capturing "this is about financial regulation." They're bad at capturing "this is specifically about Rule 10b-5." Regulatory documents are full of specific identifiers, section numbers, and policy names that users search for by exact term. BM25 catches those. Combining both with reciprocal rank fusion yielded 20-40% higher retrieval recall compared to dense search alone.

### The freshness-first design

Every embedding in the system carries its source document hash and version number. When a source document changes:

1. The change is detected via file hash comparison
2. All embeddings from the old version are invalidated (marked `is_current=false`)
3. The document is re-processed through the full pipeline
4. New embeddings are created with the new version tag
5. A reconciliation check verifies the vector store matches source truth

This is the part most RAG systems skip entirely. They add documents but never invalidate them. Over time, the vector store accumulates contradictory versions and the system silently degrades.

### The dead letter pattern

Documents that fail at any stage (download, parsing, embedding) land in a dead letter queue instead of disappearing. This is borrowed from message queue systems (SQS, Kafka DLQ) and applied to document processing. Failed documents are tracked with error details, retry counts, and the stage where they failed. The monitoring dashboard surfaces these prominently.

---

## What I'd Do Differently

- **Embedding model lock-in.** Changing embedding models requires re-embedding the entire corpus. I'd build an abstraction layer that makes model swaps a configuration change with an automated re-embedding migration, rather than a manual process.

- **Chunk boundary intelligence.** The recursive splitter occasionally breaks inside tables or structured data. I'd invest in format-aware splitting that detects tables, lists, and code blocks and keeps them intact.

- **Caching the BM25 index.** Currently, the BM25 index is rebuilt from the database on retriever initialization. At scale, I'd persist the index and update it incrementally when chunks change, rather than rebuilding from scratch.

- **Streaming re-embedding.** The current approach re-embeds documents in batch after detecting changes. A streaming architecture (change data capture from the source, triggering immediate re-embedding) would reduce the window where stale content can be served.

- **Multi-tenant isolation.** The current architecture assumes a single document corpus. For a multi-tenant SaaS deployment, I'd partition the vector store and freshness tracking by tenant, with per-tenant staleness guarantees.

- **Dedicated vector store at scale.** If the corpus grew to millions of documents, pgvector's linear scan performance would become a bottleneck. I'd migrate to a dedicated vector database (Qdrant or Weaviate) with ANN indexing, accepting the operational complexity of keeping metadata and embeddings in sync across two systems.

---

## Project Structure

```
production-rag-pipeline/
├── config/
│   └── settings.py                 # Central Pydantic configuration
├── src/
│   ├── ingestion/
│   │   ├── sources.py              # SEC EDGAR, URL, and local file fetchers
│   │   ├── bronze.py               # Raw document ingestion with provenance
│   │   ├── silver.py               # HTML/PDF/text parsing and cleaning
│   │   └── pipeline.py             # Full pipeline orchestrator
│   ├── chunking/
│   │   └── engine.py               # Recursive 512-token splitting
│   ├── embedding/
│   │   └── embedder.py             # OpenAI batch embedding + gold storage
│   ├── freshness/
│   │   └── tracker.py              # Version tracking, stale detection, reconciliation
│   ├── retrieval/
│   │   ├── hybrid.py               # Vector + BM25 + reciprocal rank fusion
│   │   └── rag.py                  # Full RAG query engine
│   ├── evaluation/
│   │   └── evaluator.py            # 3-dimension quality evaluation
│   ├── dashboard/
│   │   └── app.py                  # Streamlit monitoring dashboard
│   ├── orchestration/
│   │   └── scheduler.py            # APScheduler pipeline scheduling
│   └── api/
│       ├── app.py                  # FastAPI application
│       └── routes.py               # REST endpoints
├── tests/                          # Unit tests
├── scripts/                        # CLI tools
├── eval_data/                      # Evaluation test sets
├── docker-compose.yml              # Full stack deployment
└── Dockerfile
```

---

## Quick Start

### Docker Compose (recommended)

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env

docker compose up -d

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
# API docs:  http://localhost:8000/docs
```

### Local Development

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Configure DATABASE_URL and OPENAI_API_KEY

PYTHONPATH=. python scripts/init_db.py          # Initialize database
PYTHONPATH=. python scripts/run_pipeline.py directory ./docs  # Ingest documents
PYTHONPATH=. uvicorn src.api.app:app --reload    # Start API
PYTHONPATH=. streamlit run src/dashboard/app.py  # Start dashboard
PYTHONPATH=. pytest tests/ -v                    # Run tests
```

---

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | Ask a question (full RAG pipeline) |
| `POST` | `/api/v1/search` | Retrieval only (no LLM generation) |
| `POST` | `/api/v1/pipeline/ingest/directory` | Ingest from local directory |
| `POST` | `/api/v1/pipeline/ingest/urls` | Ingest from URLs |
| `POST` | `/api/v1/pipeline/freshness-check` | Run freshness reconciliation |
| `GET`  | `/api/v1/pipeline/runs` | List pipeline run history |
| `GET`  | `/api/v1/pipeline/dead-letters` | View failed document queue |
| `GET`  | `/api/v1/monitoring/freshness` | Freshness statistics |
| `GET`  | `/api/v1/monitoring/reconciliation` | Vector store vs. source drift check |
| `GET`  | `/api/v1/monitoring/retrieval-logs` | Retrieval quality logs |
| `GET`  | `/api/v1/monitoring/stats` | Pipeline statistics |

```bash
# Example query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What triggers an 8-K filing requirement?", "top_k": 5}'
```

---

## Evaluation

The evaluation framework measures three dimensions that matter in production:

- **Retrieval precision** — did we pull the right chunks for the question?
- **Answer faithfulness** — did the LLM stick to what was retrieved, or did it hallucinate?
- **Freshness accuracy** — did we serve content from the current document version?

```bash
PYTHONPATH=. python scripts/run_eval.py --test-set eval_data/test_set.json --output results.json
```

Results are tracked over time in the monitoring dashboard, so you can see quality trends as the document corpus changes.

---

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| API | FastAPI | Async-ready, auto-generated OpenAPI docs |
| Database | PostgreSQL + pgvector | Single database for embeddings and metadata |
| Embeddings | OpenAI text-embedding-3-small | Cost-efficient, good quality for this domain |
| LLM | GPT-4o-mini | Fast, cheap, sufficient for grounded generation |
| Text splitting | LangChain RecursiveCharacterTextSplitter | Token-aware recursive splitting with overlap |
| Keyword search | rank-bm25 | Lightweight BM25 for exact term matching |
| Scheduling | APScheduler | Lightweight cron without Airflow overhead |
| Dashboard | Streamlit + Plotly | Fast to build, interactive monitoring |
| Hashing | xxhash | Fast content hashing for change detection |

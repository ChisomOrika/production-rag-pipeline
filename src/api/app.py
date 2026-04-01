"""FastAPI application: REST API for the RAG pipeline."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from src.db.session import init_db

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    init_db()
    logger.info("RAG Pipeline API started")
    yield
    logger.info("RAG Pipeline API shutting down")


app = FastAPI(
    title="Production RAG Pipeline",
    description="RAG Pipeline with Data Freshness Guarantees",
    version="1.0.0",
    lifespan=lifespan,
)

# Import and include routers
from src.api.routes import query_router, pipeline_router, monitoring_router

app.include_router(query_router, prefix="/api/v1", tags=["Query"])
app.include_router(pipeline_router, prefix="/api/v1/pipeline", tags=["Pipeline"])
app.include_router(monitoring_router, prefix="/api/v1/monitoring", tags=["Monitoring"])


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "rag-pipeline"}

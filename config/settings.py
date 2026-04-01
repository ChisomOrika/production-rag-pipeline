"""Central configuration for the RAG pipeline."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://raguser:ragpass@localhost:5432/rag_pipeline"

    # OpenAI
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    llm_model: str = "gpt-4o-mini"

    # Data directories
    data_dir: str = "./data"
    landing_dir: str = "./data/landing"
    bronze_dir: str = "./data/bronze"
    silver_dir: str = "./data/silver"
    gold_dir: str = "./data/gold"

    # SEC EDGAR
    sec_edgar_user_agent: str = "RAGPipeline admin@example.com"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k: int = 10
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # Freshness
    staleness_check_interval_hours: int = 6
    max_embedding_lag_minutes: int = 30

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

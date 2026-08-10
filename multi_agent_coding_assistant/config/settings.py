"""Application configuration.

Uses python-dotenv and pydantic-settings for validation.
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache(maxsize=1)
def _load_env() -> None:
    load_dotenv()


class Settings(BaseSettings):
    """Strongly-typed configuration."""

    openai_api_key: str | None = None

    fastapi_host: str = "0.0.0.0"
    fastapi_port: int = 8000

    # Maximum self-correction iterations for the workflow orchestrator.
    # Single source of truth; never hard-coded elsewhere.
    max_iterations: int = 3

    # --- Repository RAG configuration --------------------------------------
    # Embedding model (local sentence-transformers). This is deliberately
    # separate from the Gemini generation model (LLMClient).
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # ChromaDB persistence directory.
    vector_store_path: str = ".rag_chroma"

    # Chunking defaults.
    rag_chunk_size: int = 500
    rag_chunk_overlap: int = 50

    # Retrieval default top-k.
    rag_top_k: int = 5

    # Allowed repository roots. A user-provided path must resolve inside one
    # of these or indexing is rejected. Empty by default (indexing disabled
    # until explicitly configured) for safety.
    allowed_repository_roots: list[str] = []

    model_config = SettingsConfigDict(env_prefix="")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env()
    return Settings()

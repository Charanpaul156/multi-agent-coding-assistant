"""Application configuration.

Uses python-dotenv and pydantic-settings for validation.
"""

from functools import lru_cache

from pathlib import Path

from typing import Any
import json
import re

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@lru_cache(maxsize=1)
def _load_env() -> None:
    load_dotenv()
    backend_env = Path("backend/.env")
    if backend_env.exists():
        load_dotenv(dotenv_path=str(backend_env), override=False)


class Settings(BaseSettings):
    """Strongly-typed configuration."""

    # --- LLM Provider configuration ---------------------------------------
    llm_provider: str = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    groq_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"

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
    allowed_repository_roots: list[str] | str = []

    @field_validator("allowed_repository_roots", mode="after")
    @classmethod
    def _validate_allowed_repository_roots(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("[") and v.endswith("]"):
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, list):
                        return [str(p).strip() for p in parsed if str(p).strip()]
                except Exception:
                    pass
            return [p.strip() for p in re.split(r"[,;\n]", v) if p.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(p).strip() for p in v if str(p).strip()]
        return []

    model_config = SettingsConfigDict(env_prefix="")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env()
    return Settings()

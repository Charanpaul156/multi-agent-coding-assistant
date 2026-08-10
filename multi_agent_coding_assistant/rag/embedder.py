"""Embedder abstraction.

The application depends only on this `Embedder` protocol. The concrete
implementation (local sentence-transformers, or a deterministic fake for
tests) is injected via DI. This keeps the rest of the RAG pipeline decoupled
from the embedding technology.

The embedding model is deliberately SEPARATE from the Gemini generation model
(`backend.infrastructure.llm_client.LLMClient`). We never use the LLM client
for embeddings and never modify it.
"""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod


class Embedder(ABC):
    """Convert text chunks into dense embedding vectors.

    Implementations:
        - SentenceTransformerEmbedder: local, offline, deterministic.
        - DeterministicEmbedder: hash-based fake for tests (no model load).
    """

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents. Returns one vector per input text."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        raise NotImplementedError


class DeterministicEmbedder(Embedder):
    """A deterministic, model-free embedder for tests and offline use.

    Produces small fixed-dimension vectors based on token hashing. It is
    deterministic and does NOT require downloading any model. Useful for unit
    tests and for exercising the pipeline without an external model.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    @staticmethod
    def _stable_hash(token: str) -> int:
        """Stable, process-independent hash for deterministic embeddings.

        Python's built-in ``hash()`` is randomized per process
        (PYTHONHASHSEED), which would make results non-deterministic across
        runs. We use ``zlib.crc32`` instead for reproducibility.
        """
        return zlib.crc32(token.encode("utf-8"))

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        # Simple hashing bag-of-words into a fixed-dim vector.
        for token in text.split():
            idx = self._stable_hash(token) % self._dim
            vector[idx] += 1.0
        # Normalize to unit length for cosine-similarity semantics.
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

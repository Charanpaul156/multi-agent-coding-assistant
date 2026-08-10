"""SentenceTransformer embedder implementation.

Loads a local `sentence-transformers` model (e.g. all-MiniLM-L6-v2) and
produces dense embeddings. This is the preferred local embedding strategy for
the portfolio project: offline, free, deterministic, no API key.

The model is loaded lazily on first use so importing the module does not
download anything. This implementation is intentionally isolated so the rest
of the pipeline depends only on the `Embedder` abstraction.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from rag.embedder import Embedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a local sentence-transformers model."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        # Lazy import so pytest collection does not require the dependency.
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading sentence-transformers model: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            # Probe the dimension from a single token embedding.
            model = self._get_model()
            probe = model.encode(["probe"], convert_to_numpy=True)
            self._dimension = int(probe.shape[1])
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        encoded = model.encode(texts, convert_to_numpy=True)
        return [row.tolist() for row in encoded]

    def embed_query(self, text: str) -> list[float]:
        model = self._get_model()
        encoded = model.encode([text], convert_to_numpy=True)
        return encoded[0].tolist()


@lru_cache(maxsize=1)
def get_sentence_transformer_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """Return a cached SentenceTransformerEmbedder for a model name."""
    return SentenceTransformerEmbedder(model_name)

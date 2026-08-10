"""Shared test fixtures.

Provides an in-memory `VectorStore` fake so pipeline/retriever/vector-store
tests do not require a real ChromaDB or an embedding model.
"""

from __future__ import annotations

import math

import pytest

from rag.vector_store import StoredDocument, VectorStore


class InMemoryVectorStore(VectorStore):
    """A deterministic in-memory VectorStore for tests.

    Uses cosine similarity over the supplied embeddings. This lets us test the
    retriever/pipeline without a real vector database.
    """

    def __init__(self) -> None:
        self._docs: dict[str, StoredDocument] = {}
        self._vectors: dict[str, list[float]] = {}

    def add_documents(
        self,
        documents: list[StoredDocument],
        embeddings: list[list[float]],
    ) -> None:
        assert len(documents) == len(embeddings)
        for doc, emb in zip(documents, embeddings):
            self._docs[doc.id] = doc
            self._vectors[doc.id] = emb

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[StoredDocument]:
        scored: list[tuple[float, StoredDocument]] = []
        for doc_id, vec in self._vectors.items():
            doc = self._docs[doc_id]
            if metadata_filter:
                if not all(doc.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue
            sim = self._cosine(query_embedding, vec)
            scored.append((1.0 - sim, doc))
        scored.sort(key=lambda x: x[0])
        return [
            StoredDocument(
                id=doc.id,
                content=doc.content,
                metadata=dict(doc.metadata),
                distance=dist,
            )
            for dist, doc in scored[:top_k]
        ]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def delete_repository(self, repository: str) -> None:
        to_delete = [
            doc_id
            for doc_id, doc in self._docs.items()
            if doc.metadata.get("repository") == repository
        ]
        for doc_id in to_delete:
            self._docs.pop(doc_id, None)
            self._vectors.pop(doc_id, None)

    def clear(self) -> None:
        self._docs.clear()
        self._vectors.clear()

    def count(self, repository: str | None = None) -> int:
        if not repository:
            return len(self._docs)
        return sum(
            1
            for doc in self._docs.values()
            if doc.metadata.get("repository") == repository
        )


@pytest.fixture
def memory_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


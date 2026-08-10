"""Vector store abstraction.

The application depends only on this `VectorStore` protocol. The concrete
implementation (ChromaDB) is injected via DI. This keeps the rest of the
application decoupled from ChromaDB's API.

Operations:
    - add_documents: index a batch of documents with embeddings + metadata.
    - similarity_search: return the most relevant documents for a query vector.
    - delete_repository: remove all documents belonging to a repository.
    - clear: wipe the store.
    - count: number of indexed documents (optionally per repository).

Documents are represented by `StoredDocument`, which carries the content and
metadata but never raw vector-database objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StoredDocument:
    """A document stored in the vector store.

    Attributes:
        id: Unique document id.
        content: The chunk text.
        metadata: Arbitrary key/value metadata (repository, file_path, lines,
            language, etc.).
        distance: Optional similarity/distance score (populated on retrieval).
    """

    id: str
    content: str
    metadata: dict
    distance: float | None = None


class VectorStore(ABC):
    """Abstract vector store for code retrieval."""

    @abstractmethod
    def add_documents(
        self,
        documents: list[StoredDocument],
        embeddings: list[list[float]],
    ) -> None:
        """Index documents with their precomputed embeddings."""
        raise NotImplementedError

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[StoredDocument]:
        """Return the top-k most similar documents for a query embedding."""
        raise NotImplementedError

    @abstractmethod
    def delete_repository(self, repository: str) -> None:
        """Delete all documents belonging to a repository identifier."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the store."""
        raise NotImplementedError

    @abstractmethod
    def count(self, repository: str | None = None) -> int:
        """Return the number of indexed documents.

        If ``repository`` is provided, count only documents for that repo.
        """
        raise NotImplementedError

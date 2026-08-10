"""Repository retriever.

Converts a natural-language query into relevant repository chunks. This is the
single entry point that agents / application layer use to obtain repository
context. It never returns raw vector-database objects; it returns `RetrievedChunk`
objects with clean, agent-safe fields.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag.embedder import Embedder
from rag.vector_store import StoredDocument, VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A repository chunk returned by the retriever.

    Attributes:
        content: The chunk text (safe for agents).
        file_path: Repository-relative POSIX path.
        start_line: 1-based inclusive start line.
        end_line: 1-based inclusive end line.
        language: Language label.
        repository: Repository identifier.
        chunk_index: Zero-based chunk index within the file.
        distance: Similarity/distance score (lower is more similar).
        metadata: Full metadata dict (as returned by the vector store).
    """

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    repository: str
    chunk_index: int
    distance: float | None = None
    metadata: dict = None  # type: ignore

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class RepositoryRetriever:
    """Retrieve relevant repository chunks for a natural-language query."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._top_k = max(1, int(top_k))

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        repository: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k relevant chunks for ``query``.

        Args:
            query: Natural-language query.
            top_k: Overrides the default top-k if provided.
            repository: If set, restrict results to a given repository.

        Returns:
            A list of `RetrievedChunk` results, ordered by relevance.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = self._top_k if top_k is None else max(1, int(top_k))
        query_embedding = self._embedder.embed_query(query)

        metadata_filter = {"repository": repository} if repository else None
        results = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=k,
            metadata_filter=metadata_filter,
        )

        return [self._to_retrieved_chunk(doc) for doc in results]

    def _to_retrieved_chunk(self, doc: StoredDocument) -> RetrievedChunk:
        metadata = doc.metadata or {}
        return RetrievedChunk(
            content=doc.content,
            file_path=str(metadata.get("file_path", "")),
            start_line=int(metadata.get("start_line") or 0),
            end_line=int(metadata.get("end_line") or 0),
            language=str(metadata.get("language", "")),
            repository=str(metadata.get("repository", "")),
            chunk_index=int(metadata.get("chunk_index") or 0),
            distance=doc.distance,
            metadata=metadata,
        )

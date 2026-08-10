"""ChromaDB vector store implementation.

This is the concrete `VectorStore` implementation backed by ChromaDB. The
application never depends on ChromaDB directly; it only uses the `VectorStore`
abstraction. This module maps the generic `StoredDocument`/metadata model onto
ChromaDB's collection API.

The collection name is derived from the embedding dimension so that different
embedding sizes do not collide. Metadata is stored as Chroma document metadata
(flat scalar values only).
"""

from __future__ import annotations

import logging
from functools import lru_cache

from rag.vector_store import StoredDocument, VectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store."""

    def __init__(
        self,
        persist_directory: str,
        collection_name: str,
        embedding_dimension: int,
    ) -> None:
        self._persist_directory = persist_directory
        self._collection_name = collection_name
        self._embedding_dimension = embedding_dimension
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy ChromaDB client / collection
    # ------------------------------------------------------------------

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=self._persist_directory)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _prepare_metadata(self, metadata: dict) -> dict:
        """Return only scalar metadata values that ChromaDB accepts."""
        prepared: dict = {}
        for key, value in metadata.items():
            if isinstance(value, bool):
                prepared[key] = str(value)
            elif isinstance(value, (str, int, float)):
                prepared[key] = value
        return prepared

    # ------------------------------------------------------------------
    # VectorStore API
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[StoredDocument],
        embeddings: list[list[float]],
    ) -> None:
        if not documents:
            return
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have equal length")

        collection = self._get_collection()
        ids = [d.id for d in documents]
        contents = [d.content for d in documents]
        metadatas = [self._prepare_metadata(d.metadata) for d in documents]

        collection.add(
            ids=ids,
            documents=contents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("ChromaVectorStore: indexed %d documents", len(documents))

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[StoredDocument]:
        collection = self._get_collection()
        where = None
        if metadata_filter:
            parts = []
            for key, value in metadata_filter.items():
                if isinstance(value, bool):
                    parts.append({key: {"$eq": str(value)}})
                elif isinstance(value, (str, int, float)):
                    parts.append({key: {"$eq": value}})
            if parts:
                where = {"$and": parts} if len(parts) > 1 else parts[0]

        try:
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
            )
        except Exception as exc:
            logger.warning("ChromaVectorStore: query error: %s", exc)
            return []

        documents = []
        ids = result.get("ids", [[]])[0] or []
        contents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []

        for i, doc_id in enumerate(ids):
            documents.append(
                StoredDocument(
                    id=doc_id,
                    content=contents[i] if i < len(contents) else "",
                    metadata=metadatas[i] if i < len(metadatas) else {},
                    distance=distances[i] if i < len(distances) else None,
                )
            )
        return documents

    def delete_repository(self, repository: str) -> None:
        collection = self._get_collection()
        try:
            collection.delete(where={"repository": {"$eq": repository}})
            logger.info("ChromaVectorStore: deleted repository %s", repository)
        except Exception as exc:
            logger.warning("ChromaVectorStore: delete failed: %s", exc)

    def clear(self) -> None:
        collection = self._get_collection()
        try:
            existing = collection.get()
            ids = existing.get("ids", [])
            if ids:
                collection.delete(ids=ids)
            logger.info("ChromaVectorStore: cleared %d documents", len(ids))
        except Exception as exc:
            logger.warning("ChromaVectorStore: clear failed: %s", exc)

    def count(self, repository: str | None = None) -> int:
        collection = self._get_collection()
        try:
            where = {"repository": {"$eq": repository}} if repository else None
            return int(collection.count(where=where) or 0)
        except Exception as exc:
            logger.warning("ChromaVectorStore: count failed: %s", exc)
            return 0


@lru_cache(maxsize=1)
def get_chroma_vector_store(
    persist_directory: str,
    embedding_dimension: int,
) -> ChromaVectorStore:
    """Return a cached ChromaVectorStore for the given dimension."""
    return ChromaVectorStore(
        persist_directory=persist_directory,
        collection_name=f"repo_rag_{embedding_dimension}d",
        embedding_dimension=embedding_dimension,
    )

"""RAG pipeline orchestrator.

Coordinates the full ingestion and retrieval flow:

    Repository
        -> RepositoryLoader
        -> CodeChunker
        -> Embedder
        -> VectorStore
        -> RepositoryRetriever

The pipeline exposes two high-level operations:
    - index_repository(path): load, chunk, embed, and store a repository.
    - search(query): retrieve relevant chunks for a query.

It also exposes status helpers (indexed files/chunks) for the UI and API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.chunker import CodeChunker
from rag.config import RagConfig
from rag.embedder import Embedder
from rag.file_loader import RepositoryLoader
from rag.retriever import RepositoryRetriever, RetrievedChunk
from rag.vector_store import StoredDocument, VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexResult:
    """Result of indexing a repository."""

    repository: str
    file_count: int
    chunk_count: int
    success: bool = True
    error: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """Result of searching the repository."""

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)


class RepositoryRagPipeline:
    """High-level RAG pipeline for repository understanding."""

    def __init__(
        self,
        config: RagConfig,
        loader: RepositoryLoader,
        chunker: CodeChunker,
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._config = config
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._retriever = RepositoryRetriever(
            vector_store=vector_store,
            embedder=embedder,
            top_k=config.top_k,
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_repository(
        self,
        path: str | Path,
        *,
        replace_existing: bool = True,
    ) -> IndexResult:
        """Index a repository directory.

        Args:
            path: Absolute path to the repository root.
            replace_existing: If True (default), delete any previously indexed
                data for the same repository before re-indexing.

        Returns:
            An `IndexResult` describing the outcome.
        """
        try:
            files = self._loader.load_repository(path)
        except Exception as exc:
            logger.exception("Indexing failed")
            return IndexResult(
                repository=str(path),
                file_count=0,
                chunk_count=0,
                success=False,
                error=str(exc),
            )

        # Determine the repository id from the root name even when empty.
        repository = ""
        try:
            root = self._loader._validate_root(path)
            repository = root.name
        except Exception:
            pass
        if files:
            repository = files[0].repository

        if not files:
            return IndexResult(
                repository=repository,
                file_count=0,
                chunk_count=0,
                success=True,
            )

        # Collect all chunks across files.
        all_chunks = []
        for source in files:
            all_chunks.extend(self._chunker.chunk_file(source))

        # Embed all chunk contents in one batch.
        contents = [chunk.content for chunk in all_chunks]
        embeddings = self._embedder.embed_documents(contents)

        documents = [
            StoredDocument(
                id=chunk.id,
                content=chunk.content,
                metadata=chunk.metadata_dict(),
            )
            for chunk in all_chunks
        ]

        if replace_existing:
            self._vector_store.delete_repository(repository)

        self._vector_store.add_documents(documents, embeddings)

        logger.info(
            "RepositoryRagPipeline: indexed repo=%s files=%d chunks=%d",
            repository,
            len(files),
            len(all_chunks),
        )
        return IndexResult(
            repository=repository,
            file_count=len(files),
            chunk_count=len(all_chunks),
            success=True,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        repository: str | None = None,
    ) -> SearchResult:
        """Search the indexed repository and return relevant chunks."""
        chunks = self._retriever.retrieve(
            query=query,
            top_k=top_k,
            repository=repository,
        )
        return SearchResult(query=query, chunks=chunks)

    # ------------------------------------------------------------------
    # Status / management
    # ------------------------------------------------------------------

    def count(self, repository: str | None = None) -> int:
        """Return the number of indexed chunks (optionally per repository)."""
        return self._vector_store.count(repository=repository)

    @property
    def config(self) -> RagConfig:
        return self._config

    @property
    def retriever(self) -> RepositoryRetriever:
        return self._retriever

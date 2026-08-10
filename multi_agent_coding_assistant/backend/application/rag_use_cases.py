"""Application use-cases for Repository RAG.

Framework-agnostic DTOs and use-cases. The application layer depends only on
the `RepositoryRagPipeline` abstraction; it never touches ChromaDB, the
embedding model, or repository filesystem traversal directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from rag.pipeline import RepositoryRagPipeline
from rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexRepositoryRequest:
    """Request DTO for indexing a repository."""

    repository_path: str


@dataclass(frozen=True)
class IndexRepositoryResult:
    """Result DTO for indexing a repository."""

    success: bool
    repository: str
    file_count: int
    chunk_count: int
    error: str | None = None


@dataclass(frozen=True)
class SearchRepositoryRequest:
    """Request DTO for searching the repository."""

    query: str
    top_k: int | None = None
    repository: str | None = None


@dataclass(frozen=True)
class SearchRepositoryResult:
    """Result DTO for a repository search."""

    success: bool
    query: str
    results: list[RetrievedChunk] = field(default_factory=list)
    error: str | None = None


@dataclass(frozen=True)
class GetRagStatusResult:
    """Status of the RAG index."""

    chunk_count: int
    repositories: list[str] = field(default_factory=list)


class IndexRepositoryUseCase:
    """Use-case: index a repository into the RAG store."""

    def __init__(self, pipeline: RepositoryRagPipeline) -> None:
        self._pipeline = pipeline

    def execute(self, request: IndexRepositoryRequest) -> IndexRepositoryResult:
        if not isinstance(request.repository_path, str):
            raise TypeError("repository_path must be a string")
        if not request.repository_path.strip():
            raise ValueError("repository_path must be non-empty")

        logger.info("IndexRepositoryUseCase: start path=%r", request.repository_path)
        result = self._pipeline.index_repository(request.repository_path)
        logger.info("IndexRepositoryUseCase: finished")
        return IndexRepositoryResult(
            success=result.success,
            repository=result.repository,
            file_count=result.file_count,
            chunk_count=result.chunk_count,
            error=result.error,
        )


class SearchRepositoryUseCase:
    """Use-case: search the indexed repository."""

    def __init__(self, pipeline: RepositoryRagPipeline) -> None:
        self._pipeline = pipeline

    def execute(self, request: SearchRepositoryRequest) -> SearchRepositoryResult:
        if not isinstance(request.query, str):
            raise TypeError("query must be a string")
        if not request.query.strip():
            raise ValueError("query must be non-empty")

        logger.info("SearchRepositoryUseCase: start query=%r", request.query)
        result = self._pipeline.search(
            request.query,
            top_k=request.top_k,
            repository=request.repository,
        )
        logger.info("SearchRepositoryUseCase: returned %d chunks", len(result.chunks))
        return SearchRepositoryResult(
            success=True,
            query=request.query,
            results=result.chunks,
        )


class GetRagStatusUseCase:
    """Use-case: return the current RAG index status."""

    def __init__(self, pipeline: RepositoryRagPipeline) -> None:
        self._pipeline = pipeline

    def execute(self) -> GetRagStatusResult:
        chunk_count = self._pipeline.count()
        return GetRagStatusResult(
            chunk_count=chunk_count,
            repositories=[],
        )

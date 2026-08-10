"""Tests for RepositoryRagPipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunker import CodeChunker
from rag.config import RagConfig
from rag.embedder import DeterministicEmbedder
from rag.file_loader import RepositoryLoader
from rag.pipeline import IndexResult, RepositoryRagPipeline, SearchResult
from tests.conftest import InMemoryVectorStore


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        "def login(): pass\n", encoding="utf-8"
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "models.py").write_text(
        "class User: pass\n", encoding="utf-8"
    )
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
    return tmp_path


def _config(repo_dir: Path) -> RagConfig:
    return RagConfig(allowed_repository_roots=(str(repo_dir),))


def _pipeline(repo_dir: Path, store: InMemoryVectorStore) -> RepositoryRagPipeline:
    conf = _config(repo_dir)
    return RepositoryRagPipeline(
        config=conf,
        loader=RepositoryLoader(conf),
        chunker=CodeChunker(chunk_size=50, chunk_overlap=5),
        embedder=DeterministicEmbedder(dimension=64),
        vector_store=store,
    )


def test_index_repository_indexes_supported_files(
    repo_dir: Path, memory_store: InMemoryVectorStore
) -> None:
    pipeline = _pipeline(repo_dir, memory_store)
    result = pipeline.index_repository(repo_dir)
    assert isinstance(result, IndexResult)
    assert result.success is True
    assert result.file_count == 3  # app.py, models.py, notes.md
    assert result.chunk_count >= 3
    assert result.repository == repo_dir.name
    assert memory_store.count() == result.chunk_count


def test_index_then_search_repo(repo_dir: Path, memory_store: InMemoryVectorStore) -> None:
    pipeline = _pipeline(repo_dir, memory_store)
    pipeline.index_repository(repo_dir)
    search = pipeline.search("login function")
    assert isinstance(search, SearchResult)
    assert search.query == "login function"
    assert len(search.chunks) > 0
    assert any("login" in c.content for c in search.chunks)


def test_search_empty_store_return_empty(repo_dir: Path, memory_store: InMemoryVectorStore) -> None:
    pipeline = _pipeline(repo_dir, memory_store)
    assert pipeline.search("anything").chunks == []


def test_index_missing_repository_returns_failure(
    tmp_path: Path, memory_store: InMemoryVectorStore
) -> None:
    conf = RagConfig(allowed_repository_roots=(str(tmp_path),))
    pipeline = RepositoryRagPipeline(
        config=conf,
        loader=RepositoryLoader(conf),
        chunker=CodeChunker(chunk_size=50, chunk_overlap=5),
        embedder=DeterministicEmbedder(dimension=64),
        vector_store=memory_store,
    )
    missing = tmp_path / "does_not_exist"
    result = pipeline.index_repository(missing)
    assert result.success is False
    assert result.error is not None


def test_index_empty_repository(tmp_path: Path, memory_store: InMemoryVectorStore) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    conf = RagConfig(allowed_repository_roots=(str(tmp_path),))
    pipeline = RepositoryRagPipeline(
        config=conf,
        loader=RepositoryLoader(conf),
        chunker=CodeChunker(chunk_size=50, chunk_overlap=5),
        embedder=DeterministicEmbedder(dimension=64),
        vector_store=memory_store,
    )
    result = pipeline.index_repository(empty)
    assert result.success is True
    assert result.file_count == 0
    assert result.chunk_count == 0


def test_count_per_repository(repo_dir: Path, memory_store: InMemoryVectorStore) -> None:
    pipeline = _pipeline(repo_dir, memory_store)
    pipeline.index_repository(repo_dir)
    assert memory_store.count(repository=repo_dir.name) == memory_store.count()

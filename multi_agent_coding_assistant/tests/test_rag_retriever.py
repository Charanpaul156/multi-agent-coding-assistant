"""Tests for RepositoryRetriever."""

from __future__ import annotations

import pytest

from rag.embedder import DeterministicEmbedder
from rag.retriever import RepositoryRetriever, RetrievedChunk
from rag.vector_store import StoredDocument
from tests.conftest import InMemoryVectorStore


def _add_doc(
    store: InMemoryVectorStore,
    content: str,
    *,
    repo: str = "repo",
    file_path: str = "app.py",
    start_line: int = 1,
    end_line: int = 1,
    chunk_index: int = 0,
) -> None:
    doc = StoredDocument(
        id=f"{repo}::{file_path}::{chunk_index}",
        content=content,
        metadata={
            "repository": repo,
            "file_path": file_path,
            "language": "python",
            "chunk_index": chunk_index,
            "start_line": start_line,
            "end_line": end_line,
            "file_hash": "hash",
        },
    )
    emb = DeterministicEmbedder(dimension=64)
    store.add_documents([doc], [emb.embed_documents([content])[0]])


def test_retrieve_returns_chunks_with_metadata(
    memory_store: InMemoryVectorStore,
) -> None:
    _add_doc(
        memory_store,
        "def login(): pass",
        file_path="auth/service.py",
        start_line=10,
        end_line=12,
    )
    retriever = RepositoryRetriever(
        vector_store=memory_store,
        embedder=DeterministicEmbedder(dimension=64),
        top_k=5,
    )
    results = retriever.retrieve("login function")
    assert len(results) == 1
    chunk = results[0]
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.file_path == "auth/service.py"
    assert chunk.start_line == 10
    assert chunk.end_line == 12
    assert chunk.repository == "repo"
    assert chunk.language == "python"
    assert chunk.content == "def login(): pass"


def test_retrieve_empty_query_raises(memory_store: InMemoryVectorStore) -> None:
    retriever = RepositoryRetriever(
        vector_store=memory_store,
        embedder=DeterministicEmbedder(dimension=64),
    )
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_retrieve_top_k(memory_store: InMemoryVectorStore) -> None:
    for i in range(5):
        _add_doc(memory_store, f"function {i} login", chunk_index=i)
    retriever = RepositoryRetriever(
        vector_store=memory_store,
        embedder=DeterministicEmbedder(dimension=64),
        top_k=10,
    )
    results = retriever.retrieve("login", top_k=2)
    assert len(results) == 2


def test_retrieve_repository_filter(memory_store: InMemoryVectorStore) -> None:
    _add_doc(memory_store, "payment login", repo="app", file_path="a.py")
    _add_doc(memory_store, "different content here", repo="other", file_path="b.py")
    retriever = RepositoryRetriever(
        vector_store=memory_store,
        embedder=DeterministicEmbedder(dimension=64),
    )
    results = retriever.retrieve("login", repository="app")
    assert all(r.repository == "app" for r in results)


def test_retrieve_empty_store(memory_store: InMemoryVectorStore) -> None:
    retriever = RepositoryRetriever(
        vector_store=memory_store,
        embedder=DeterministicEmbedder(dimension=64),
    )
    assert retriever.retrieve("anything") == []

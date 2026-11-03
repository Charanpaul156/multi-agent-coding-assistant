"""Tests for RepositoryRetriever."""

from __future__ import annotations

from types import SimpleNamespace

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


def test_retrieve_reranks_by_path_and_content_without_extra_embedding(
    memory_store: InMemoryVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        StoredDocument(
            id="docs",
            content="Authentication overview and login flow",
            metadata={"file_path": "docs/auth.md", "start_line": 1, "end_line": 4},
            distance=0.30,
        ),
        StoredDocument(
            id="implementation",
            content="def login_handler():\n    return authenticate_user()",
            metadata={"file_path": "src/auth/login.py", "start_line": 8, "end_line": 9},
            distance=0.35,
        ),
    ]
    requested: list[int] = []
    monkeypatch.setattr(
        memory_store,
        "similarity_search",
        lambda query_embedding, top_k, metadata_filter=None: requested.append(top_k) or documents,
    )
    embedder = DeterministicEmbedder(dimension=64)
    embedding_calls = 0
    original_embed_query = embedder.embed_query

    def count_embedding(query: str) -> list[float]:
        nonlocal embedding_calls
        embedding_calls += 1
        return original_embed_query(query)

    monkeypatch.setattr(embedder, "embed_query", count_embedding)
    retriever = RepositoryRetriever(memory_store, embedder, top_k=2)

    results = retriever.retrieve("authentication login handler", top_k=1)

    assert results[0].file_path == "src/auth/login.py"
    assert results[0].ranking_score > 0
    assert any("path" in reason for reason in results[0].ranking_reasons)
    assert requested == [3]
    assert embedding_calls == 1


def test_retrieve_deduplicates_exact_content_but_preserves_distinct_chunks(
    memory_store: InMemoryVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        StoredDocument("first", "def login(): pass", {"file_path": "auth.py", "start_line": 1}, 0.1),
        StoredDocument("duplicate", " def login():   pass ", {"file_path": "auth.py", "start_line": 1}, 0.2),
        StoredDocument("other", "def logout(): pass", {"file_path": "auth.py", "start_line": 1}, 0.3),
    ]
    monkeypatch.setattr(
        memory_store,
        "similarity_search",
        lambda query_embedding, top_k, metadata_filter=None: documents,
    )
    retriever = RepositoryRetriever(memory_store, DeterministicEmbedder(dimension=64))

    results = retriever.retrieve("login", top_k=5)

    assert len(results) == 2
    assert {chunk.content.strip() for chunk in results} == {"def login(): pass", "def logout(): pass"}


def test_retrieve_skips_malformed_documents_and_sanitizes_metadata(
    memory_store: InMemoryVectorStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = [
        SimpleNamespace(content=None, metadata={}, distance=0.1),
        StoredDocument("valid", "def login(): pass", {"file_path": "auth.py", "start_line": "bad"}, 0.1),
    ]
    monkeypatch.setattr(
        memory_store,
        "similarity_search",
        lambda query_embedding, top_k, metadata_filter=None: documents,
    )
    retriever = RepositoryRetriever(memory_store, DeterministicEmbedder(dimension=64))

    results = retriever.retrieve("login")

    assert len(results) == 1
    assert results[0].start_line == 0

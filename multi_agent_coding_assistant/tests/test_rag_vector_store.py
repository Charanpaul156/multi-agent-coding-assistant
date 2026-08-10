"""Tests for the VectorStore abstraction using the in-memory fake.

These tests verify the abstract contract (add_documents, similarity_search,
delete_repository, clear, count) without requiring a real ChromaDB.
"""

from __future__ import annotations

from rag.embedder import DeterministicEmbedder
from rag.vector_store import StoredDocument
from tests.conftest import InMemoryVectorStore


def _doc(
    doc_id: str,
    content: str = "content",
    repo: str = "repo",
    file_path: str = "app.py",
) -> StoredDocument:
    return StoredDocument(
        id=doc_id,
        content=content,
        metadata={
            "repository": repo,
            "file_path": file_path,
            "language": "python",
            "chunk_index": 0,
            "start_line": 1,
            "end_line": 1,
            "file_hash": "hash",
        },
    )


def test_add_and_count(memory_store: InMemoryVectorStore) -> None:
    emb = DeterministicEmbedder(dimension=64)
    memory_store.add_documents(
        [_doc("a")],
        [emb.embed_documents(["content a"])[0]],
    )
    assert memory_store.count() == 1
    assert memory_store.count(repository="repo") == 1
    assert memory_store.count(repository="other") == 0


def test_add_requires_equal_lengths(memory_store: InMemoryVectorStore) -> None:
    emb = DeterministicEmbedder(dimension=64)
    try:
        memory_store.add_documents([_doc("a")], [])
        assert False, "expected ValueError/AssertionError"
    except Exception:
        pass


def test_similarity_search_returns_scored(memory_store: InMemoryVectorStore) -> None:
    emb = DeterministicEmbedder(dimension=64)
    memory_store.add_documents(
        [_doc("a", "authentication login flow"), _doc("b", "payment processing")],
        emb.embed_documents(["authentication login flow", "payment processing"]),
    )
    results = memory_store.similarity_search(emb.embed_query("login"), top_k=2)
    assert len(results) == 2
    # a mentions login so it should rank first.
    assert results[0].id == "a"
    assert results[0].distance is not None


def test_similarity_search_metadata_filter(memory_store: InMemoryVectorStore) -> None:
    emb = DeterministicEmbedder(dimension=64)
    memory_store.add_documents(
        [_doc("a", "login", repo="app"), _doc("b", "login", repo="other")],
        emb.embed_documents(["login", "login"]),
    )
    results = memory_store.similarity_search(
        emb.embed_query("login"),
        metadata_filter={"repository": "app"},
    )
    assert [r.id for r in results] == ["a"]


def test_delete_repository(memory_store: InMemoryVectorStore) -> None:
    emb = DeterministicEmbedder(dimension=64)
    memory_store.add_documents(
        [_doc("a", "login", repo="app"), _doc("b", "x", repo="other")],
        emb.embed_documents(["login", "x"]),
    )
    memory_store.delete_repository("app")
    assert memory_store.count(repository="app") == 0
    assert memory_store.count(repository="other") == 1


def test_clear(memory_store: InMemoryVectorStore) -> None:
    emb = DeterministicEmbedder(dimension=64)
    memory_store.add_documents(
        [_doc("a"), _doc("b")],
        emb.embed_documents(["a", "b"]),
    )
    memory_store.clear()
    assert memory_store.count() == 0

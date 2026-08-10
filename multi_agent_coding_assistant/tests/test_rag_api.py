"""Tests for the RAG API endpoints (index/search/status).

Uses the FastAPI dependency_overrides mechanism to inject an in-memory
pipeline so no real ChromaDB model/sentence-transformers are loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.application.rag_use_cases import (
    GetRagStatusResult,
    GetRagStatusUseCase,
    IndexRepositoryResult,
    IndexRepositoryUseCase,
    SearchRepositoryResult,
    SearchRepositoryUseCase,
)
from rag.retriever import RetrievedChunk


class FakeIndexUseCase(IndexRepositoryUseCase):
    def __init__(self, pipeline=None) -> None:
        self.last_path: str | None = None

    def execute(self, request) -> Any:
        self.last_path = request.repository_path
        if "outside" in request.repository_path.lower():
            raise ValueError("Repository path is outside the allowed roots")
        return IndexRepositoryResult(
            repository="repo",
            file_count=3,
            chunk_count=7,
            success=True,
        )


class FakeSearchUseCase(SearchRepositoryUseCase):
    def __init__(self, pipeline=None) -> None:
        pass

    def execute(self, request) -> Any:
        return SearchRepositoryResult(
            success=True,
            query=request.query,
            results=[
                RetrievedChunk(
                    content="def login(): pass",
                    file_path="auth/service.py",
                    start_line=10,
                    end_line=12,
                    language="python",
                    repository="repo",
                    chunk_index=0,
                    distance=0.1,
                )
            ],
        )


class FakeStatusUseCase(GetRagStatusUseCase):
    def __init__(self, pipeline=None) -> None:
        pass

    def execute(self) -> Any:
        return GetRagStatusResult(chunk_count=7, repositories=["repo"])


@pytest.fixture(autouse=True)
def _override_deps():
    """Override the RAG deps so tests never touch real Chroma/embeddings."""
    import backend.api.deps as deps
    from backend.main import app

    app.dependency_overrides[deps.get_index_repository_use_case] = lambda: FakeIndexUseCase()
    app.dependency_overrides[deps.get_search_repository_use_case] = lambda: FakeSearchUseCase()
    app.dependency_overrides[deps.get_rag_status_use_case] = lambda: FakeStatusUseCase()
    yield
    app.dependency_overrides.clear()


def test_index_repository_endpoint_success() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    resp = client.post("/index-repository", json={"repository_path": "/repo/src"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["repository"] == "repo"
    assert data["file_count"] == 3
    assert data["chunk_count"] == 7


def test_index_repository_empty_path() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    resp = client.post("/index-repository", json={"repository_path": "  "})
    assert resp.status_code == 400


def test_index_repository_outside_root() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    resp = client.post(
        "/index-repository", json={"repository_path": "/something/outside"}
    )
    assert resp.status_code == 400


def test_search_repository_endpoint() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    resp = client.post(
        "/search-repository", json={"query": "authentication login"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["file_path"] == "auth/service.py"
    assert result["start_line"] == 10
    assert result["end_line"] == 12
    assert "login" in result["content"]


def test_search_repository_empty_query() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    resp = client.post("/search-repository", json={"query": "  "})
    assert resp.status_code == 400


def test_rag_status_endpoint() -> None:
    from fastapi.testclient import TestClient

    from backend.main import app

    client = TestClient(app)
    resp = client.get("/rag/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["chunk_count"] == 7
    assert data["repositories"] == ["repo"]

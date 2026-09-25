"""API tests for POST /generate-documentation.

Overrides the DI dependency so no real LLM/network calls are made.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from agents.documentation_agent import (
    DocumentationAgent,
    DocumentationAgentError,
    DocumentationReport,
)
from backend.api.deps import (
    get_documentation_agent,
    get_generate_documentation_use_case,
)
from backend.application.documentation_use_cases import (
    GenerateDocumentationRequest,
    GenerateDocumentationResult,
    GenerateDocumentationUseCase,
)
from backend.main import app


def _make_report() -> DocumentationReport:
    return DocumentationReport(
        summary="Math operations module.",
        module_description="Module providing basic arithmetic helper functions.",
        function_docs=[
            {
                "name": "add",
                "signature": "def add(a: int, b: int) -> int",
                "description": "Add two numbers together.",
                "parameters": ["a: First number.", "b: Second number."],
                "returns": "Sum of two numbers.",
            }
        ],
        class_docs=[],
        usage_examples=["result = add(1, 2)"],
        markdown_documentation="# Math Module\n\n## add(a, b)\nAdds two numbers.",
    )


class _FakeDocumentationUseCase(GenerateDocumentationUseCase):
    def __init__(
        self,
        report: Optional[DocumentationReport] = None,
        exc: Optional[Exception] = None,
    ) -> None:
        self._report = report
        self._exc = exc
        self.last_request: Optional[GenerateDocumentationRequest] = None
        self.calls = 0

    def execute(
        self, request: GenerateDocumentationRequest
    ) -> GenerateDocumentationResult:
        self.calls += 1
        self.last_request = request
        if self._exc is not None:
            raise self._exc
        if self._report is None:
            raise AssertionError("No DocumentationReport configured on fake use case")
        return GenerateDocumentationResult(report=self._report)


@pytest.fixture(autouse=True)
def _cleanup_dependency_overrides():
    yield
    app.dependency_overrides.clear()


def _client_with(use_case: _FakeDocumentationUseCase) -> TestClient:
    app.dependency_overrides[get_generate_documentation_use_case] = lambda: use_case
    return TestClient(app)


# 1. Successful documentation generation
def test_generate_documentation_success() -> None:
    expected_report = _make_report()
    use_case = _FakeDocumentationUseCase(report=expected_report)
    client = _client_with(use_case)

    resp = client.post(
        "/generate-documentation",
        json={"code": "def add(a, b):\n    return a + b\n"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["report"]["summary"] == "Math operations module."
    assert body["report"]["module_description"] == (
        "Module providing basic arithmetic helper functions."
    )
    assert len(body["report"]["function_docs"]) == 1
    assert body["report"]["function_docs"][0]["name"] == "add"
    assert body["report"]["class_docs"] == []
    assert body["report"]["usage_examples"] == ["result = add(1, 2)"]
    assert "# Math Module" in body["report"]["markdown_documentation"]
    assert use_case.calls == 1
    assert use_case.last_request is not None
    assert use_case.last_request.code == "def add(a, b):\n    return a + b"
    assert use_case.last_request.retrieved_context is None


# 2. Request with retrieved_context
def test_generate_documentation_with_retrieved_context() -> None:
    use_case = _FakeDocumentationUseCase(report=_make_report())
    client = _client_with(use_case)

    context = "Repository context: this function is part of the core math engine."
    code = "def add(a, b):\n    return a + b\n"
    resp = client.post(
        "/generate-documentation",
        json={"code": code, "retrieved_context": context},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert use_case.calls == 1
    assert use_case.last_request is not None
    assert use_case.last_request.code == code.strip()
    assert use_case.last_request.retrieved_context == context


# 3. Invalid/empty code
def test_generate_documentation_empty_code_returns_400() -> None:
    use_case = _FakeDocumentationUseCase(report=_make_report())
    client = _client_with(use_case)

    resp = client.post("/generate-documentation", json={"code": "   "})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()
    assert use_case.calls == 0


def test_generate_documentation_missing_code_returns_422() -> None:
    use_case = _FakeDocumentationUseCase(report=_make_report())
    client = _client_with(use_case)

    resp = client.post("/generate-documentation", json={})
    assert resp.status_code == 422
    assert use_case.calls == 0


# 4. Agent/use-case failure handling according to API conventions
def test_generate_documentation_agent_failure_returns_500() -> None:
    use_case = _FakeDocumentationUseCase(
        exc=DocumentationAgentError("Agent failed to parse LLM output")
    )
    client = _client_with(use_case)

    resp = client.post(
        "/generate-documentation",
        json={"code": "def foo(): pass"},
    )
    assert resp.status_code == 500
    assert "documentation generation failed" in resp.json()["detail"].lower()
    assert use_case.calls == 1


def test_generate_documentation_runtime_error_returns_500() -> None:
    use_case = _FakeDocumentationUseCase(exc=RuntimeError("LLM API timeout"))
    client = _client_with(use_case)

    resp = client.post(
        "/generate-documentation",
        json={"code": "def foo(): pass"},
    )
    assert resp.status_code == 500
    assert use_case.calls == 1


# 5. Dependency injection is actually used
def test_generate_documentation_di_wiring_provider() -> None:
    # Verify the DI provider in deps.py correctly constructs the use case
    use_case = get_generate_documentation_use_case()
    assert isinstance(use_case, GenerateDocumentationUseCase)
    assert isinstance(use_case.documentation_agent, DocumentationAgent)


def test_generate_documentation_di_override_invoked() -> None:
    # Verify that injecting a custom use-case into FastAPI dependency overrides
    # is actually called when hitting the route
    called = False

    class _SpyUseCase(GenerateDocumentationUseCase):
        def __init__(self):
            pass

        def execute(self, request: GenerateDocumentationRequest) -> GenerateDocumentationResult:
            nonlocal called
            called = True
            return GenerateDocumentationResult(report=_make_report())

    app.dependency_overrides[get_generate_documentation_use_case] = _SpyUseCase
    client = TestClient(app)

    resp = client.post(
        "/generate-documentation",
        json={"code": "def foo(): pass"},
    )
    assert resp.status_code == 200
    assert called is True

"""API tests for POST /generate-tests.

Overrides the DI dependency so no real LLM/network is required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agents.test_generator_agent import TestCaseInfo, TestGenerationReport
from backend.api.deps import get_generate_tests_use_case
from backend.application.test_generation_use_cases import (
    GenerateTestsRequest,
    GenerateTestsResult,
    GenerateTestsUseCase,
)
from backend.main import app


def _make_report() -> TestGenerationReport:
    return TestGenerationReport(
        test_overview="Overview",
        test_framework="pytest",
        test_cases=[
            TestCaseInfo("test_add", "adds", "add(1,2)", "returns 3"),
        ],
        edge_cases=[],
        negative_cases=[],
        coverage_suggestions=["more"],
        generated_test_code="def test_add():\n    assert add(1, 2) == 3\n",
        final_summary="Done",
    )


class _FakeUseCase(GenerateTestsUseCase):
    def __init__(self, report=None, exc=None):
        # Do not call super().__init__ (no agent needed for the fake).
        self._report = report
        self._exc = exc

    def execute(self, request: GenerateTestsRequest):
        if self._exc is not None:
            raise self._exc
        return GenerateTestsResult(report=self._report)


def _client_with(use_case: _FakeUseCase) -> TestClient:
    app.dependency_overrides[get_generate_tests_use_case] = lambda: use_case
    client = TestClient(app)
    client.__testclient_use_case = use_case  # type: ignore[attr-defined]
    return client


def test_generate_tests_success() -> None:
    use_case = _FakeUseCase(report=_make_report())
    client = _client_with(use_case)

    resp = client.post("/generate-tests", json={"generated_code": "def add(a,b):..."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["report"]["test_framework"] == "pytest"
    assert body["report"]["test_cases"][0]["name"] == "test_add"
    assert "def test_add" in body["report"]["generated_test_code"]

    app.dependency_overrides.clear()


def test_generate_tests_empty_code_returns_400() -> None:
    use_case = _FakeUseCase(report=_make_report())
    client = _client_with(use_case)

    resp = client.post("/generate-tests", json={"generated_code": "   "})
    assert resp.status_code == 400

    app.dependency_overrides.clear()


def test_generate_tests_missing_code_returns_422() -> None:
    use_case = _FakeUseCase(report=_make_report())
    client = _client_with(use_case)

    resp = client.post("/generate-tests", json={})
    assert resp.status_code == 422

    app.dependency_overrides.clear()


def test_generate_tests_agent_failure_returns_500() -> None:
    use_case = _FakeUseCase(exc=RuntimeError("boom"))
    client = _client_with(use_case)

    resp = client.post("/generate-tests", json={"generated_code": "def add():..."})
    assert resp.status_code == 500

    app.dependency_overrides.clear()

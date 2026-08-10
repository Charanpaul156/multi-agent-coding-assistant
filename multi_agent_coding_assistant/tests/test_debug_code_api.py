"""API tests for POST /debug-code.

Overrides the DI dependency so no real LLM is required.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agents.debugger_agent import DebugReport
from backend.api.deps import get_debug_code_use_case
from backend.application.debugging_use_cases import (
    DebugCodeRequest,
    DebugCodeResult,
)
from backend.main import app


def _make_report() -> DebugReport:
    return DebugReport(
        issue_detected=True,
        error_type="ZeroDivisionError",
        root_cause="Division by zero",
        affected_component="divide",
        explanation="No guard for zero divisor.",
        suggested_changes=["Add a guard."],
        corrected_code="def divide(a, b):\n    if b == 0: return None\n    return a / b\n",
        confidence=0.9,
        final_summary="Fixed.",
    )


class _FakeDebugUseCase:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def execute(self, request: DebugCodeRequest):
        if self._exc is not None:
            raise self._exc
        return self._result


def test_debug_code_success() -> None:
    use_case = _FakeDebugUseCase(result=DebugCodeResult(report=_make_report()))
    app.dependency_overrides[get_debug_code_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post(
        "/debug-code",
        json={
            "generated_code": "def divide(a, b):\n    return a / b\n",
            "execution_stderr": "ZeroDivisionError",
            "execution_exit_code": 1,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    report = body["debug_report"]
    assert report["issue_detected"] is True
    assert report["error_type"] == "ZeroDivisionError"
    assert report["corrected_code"]
    assert report["confidence"] == 0.9

    app.dependency_overrides.clear()


def test_debug_code_empty_code_returns_400() -> None:
    use_case = _FakeDebugUseCase(result=DebugCodeResult(report=_make_report()))
    app.dependency_overrides[get_debug_code_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post("/debug-code", json={"generated_code": "   "})
    assert resp.status_code == 400

    app.dependency_overrides.clear()


def test_debug_code_missing_code_returns_422() -> None:
    use_case = _FakeDebugUseCase(result=DebugCodeResult(report=_make_report()))
    app.dependency_overrides[get_debug_code_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post("/debug-code", json={})
    assert resp.status_code == 422

    app.dependency_overrides.clear()


def test_debug_code_agent_failure_returns_500() -> None:
    use_case = _FakeDebugUseCase(exc=RuntimeError("debugger down"))
    app.dependency_overrides[get_debug_code_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post("/debug-code", json={"generated_code": "def f(): pass\n"})
    assert resp.status_code == 500

    app.dependency_overrides.clear()

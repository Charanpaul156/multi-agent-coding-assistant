"""API tests for POST /run-workflow.

Overrides the DI dependency so no real LLM is required.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from agents.planner_agent import ImplementationPlan
from agents.code_reviewer_agent import ReviewReport
from backend.api.deps import get_run_workflow_use_case
from backend.application.workflow_use_cases import (
    RunWorkflowUseCase,
    WorkflowRequest,
    WorkflowResult,
    WorkflowStatus,
)
from backend.main import app
from backend.tools.python_executor import ExecutionResponse
from backend.tools.test_executor import TestExecutionResponse


def _make_plan() -> ImplementationPlan:
    return ImplementationPlan(
        problem_summary="Building a calculator",
        project_type="console",
        requirements=["add"],
        modules=["calculator"],
        functions=["add"],
        classes=[],
        external_libraries=[],
        database_needed=False,
        api_needed=[],
        algorithm="n/a",
        edge_cases=[],
        estimated_complexity="low",
        future_improvements=[],
    )


def _make_review() -> ReviewReport:
    return ReviewReport(
        overall_score=85,
        strengths=["good"],
        weaknesses=["none"],
        pep8_issues=[],
        performance_suggestions=[],
        security_concerns=[],
        logic_issues=[],
        maintainability=[],
        error_handling=[],
        recommendations=["ok"],
        final_summary="looks good",
    )


class _FakeWorkflowUseCase(RunWorkflowUseCase):
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def execute(self, request: WorkflowRequest):
        if self._exc is not None:
            raise self._exc
        return self._result


def _full_result() -> WorkflowResult:
    return WorkflowResult(
        success=True,
        workflow_status=WorkflowStatus.COMPLETED,
        planning=_make_plan(),
        generated_code="def add(a,b): return a+b",
        generated_tests="def test_add():",
        execution=ExecutionResponse(True, "1", "", 1.0, 0),
        test_execution=TestExecutionResponse(True, "1 passed", "", 1.0, 0, 1, 0),
        review=_make_review(),
    )


def test_run_workflow_full_response() -> None:
    use_case = _FakeWorkflowUseCase(result=_full_result())
    app.dependency_overrides[get_run_workflow_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post("/run-workflow", json={"prompt": "build calculator"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["success"] is True
    wf = body["workflow"]
    assert wf["workflow_status"] == "completed"
    assert wf["generated_code"]
    assert wf["generated_tests"]
    assert wf["execution"]["exit_code"] == 0
    assert wf["test_execution"]["passed"] == 1
    assert wf["planning"]["problem_summary"]
    assert wf["error"] is None
    assert wf["test_error"] is None

    app.dependency_overrides.clear()


def test_run_workflow_empty_prompt_returns_400() -> None:
    use_case = _FakeWorkflowUseCase(result=_full_result())
    app.dependency_overrides[get_run_workflow_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post("/run-workflow", json={"prompt": "   "})
    assert resp.status_code == 400

    app.dependency_overrides.clear()


def test_run_workflow_missing_prompt_returns_422() -> None:
    use_case = _FakeWorkflowUseCase(result=_full_result())
    app.dependency_overrides[get_run_workflow_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post("/run-workflow", json={})
    assert resp.status_code == 422

    app.dependency_overrides.clear()


def test_run_workflow_planner_failure() -> None:
    use_case = _FakeWorkflowUseCase(
        result=WorkflowResult(
            success=False,
            workflow_status=WorkflowStatus.PLANNING_FAILED,
            error="planning failed",
        )
    )
    app.dependency_overrides[get_run_workflow_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post("/run-workflow", json={"prompt": "build calculator"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["workflow"]["workflow_status"] == "planning_failed"
    assert body["workflow"]["generated_code"] is None

    app.dependency_overrides.clear()


def test_run_workflow_test_generation_warning() -> None:
    use_case = _FakeWorkflowUseCase(
        result=WorkflowResult(
            success=True,
            workflow_status=WorkflowStatus.COMPLETED_WITH_WARNINGS,
            planning=_make_plan(),
            generated_code="def add(a,b): return a+b",
            generated_tests=None,
            execution=ExecutionResponse(True, "1", "", 1.0, 0),
            test_execution=None,
            review=_make_review(),
            test_error="test generation failed",
        )
    )
    app.dependency_overrides[get_run_workflow_use_case] = lambda: use_case
    client = TestClient(app)

    resp = client.post("/run-workflow", json={"prompt": "build calculator"})
    assert resp.status_code == 200
    wf = resp.json()["workflow"]
    assert wf["workflow_status"] == "completed_with_warnings"
    assert wf["generated_tests"] is None
    assert wf["test_execution"] is None
    assert wf["test_error"] == "test generation failed"

    app.dependency_overrides.clear()


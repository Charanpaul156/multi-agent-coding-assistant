"""API tests for the asynchronous workflow jobs endpoints.

Verifies:
1. POST /workflow/jobs returns 202 Accepted immediately.
2. Response contains stable job_id and status 'queued'.
3. Background job execution completes and stores result in JobManager.
4. GET /workflow/jobs/{job_id} retrieves completed status and serialized result.
5. Failed workflow execution transitions job to 'failed' and stores error description.
6. GET /workflow/jobs/{job_id} returns 404 for unknown job IDs.
7. Empty/whitespace prompt returns 400 Bad Request.
8. Shared JobManager instance is used across creation, execution, and query.
9. Job ID is propagated to events emitted during execution.
10. No secrets, credentials, or private model reasoning leak in GET responses.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from agents.planner_agent import ImplementationPlan
from agents.code_reviewer_agent import ReviewReport
from backend.api.deps import (
    get_event_bus,
    get_job_manager,
    get_job_runner,
    get_run_workflow_use_case,
)
from backend.application.job_manager import JobManager
from backend.application.job_runner import JobRunner
from backend.application.workflow_use_cases import (
    RunWorkflowUseCase,
    WorkflowRequest,
    WorkflowResult,
    WorkflowStatus,
)
from backend.infrastructure.event_bus import EventBus
from backend.main import app
from backend.tools.python_executor import ExecutionResponse
from backend.tools.test_executor import TestExecutionResponse


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

def _make_sample_plan() -> ImplementationPlan:
    return ImplementationPlan(
        problem_summary="Asynchronous task executor",
        project_type="module",
        requirements=["execute async tasks"],
        modules=["tasks"],
        functions=["run"],
        classes=[],
        external_libraries=[],
        database_needed=False,
        api_needed=[],
        algorithm="fifo",
        edge_cases=[],
        estimated_complexity="low",
        future_improvements=[],
    )


def _make_sample_review() -> ReviewReport:
    return ReviewReport(
        overall_score=92,
        strengths=["clean code"],
        weaknesses=[],
        pep8_issues=[],
        performance_suggestions=[],
        security_concerns=[],
        logic_issues=[],
        maintainability=[],
        error_handling=[],
        recommendations=[],
        final_summary="verified",
    )


def _make_sample_workflow_result() -> WorkflowResult:
    return WorkflowResult(
        success=True,
        workflow_status=WorkflowStatus.COMPLETED,
        planning=_make_sample_plan(),
        generated_code="def run(): return 42\n",
        generated_tests="def test_run(): assert run() == 42\n",
        execution=ExecutionResponse(True, "42", "", 5.0, 0),
        test_execution=TestExecutionResponse(True, "1 passed", "", 8.0, 0, 1, 0),
        review=_make_sample_review(),
        execution_time_ms=13.0,
    )


class _MockWorkflowUseCase(RunWorkflowUseCase):
    """Stub use case returning preconfigured result or raising exception."""

    def __init__(self, result: WorkflowResult | None = None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.call_count = 0
        self.last_request: WorkflowRequest | None = None

    def execute(self, request: WorkflowRequest) -> WorkflowResult:
        self.call_count += 1
        self.last_request = request

        # If an event emitter is attached, emit standard progress events
        if request.event_emitter and request.job_id:
            from backend.domain.job_models import WorkflowEvent
            request.event_emitter.emit(
                WorkflowEvent(
                    event_type="workflow_started",
                    job_id=request.job_id,
                    payload={"step": "init"},
                )
            )

        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_di_overrides():
    """Ensure dependency overrides are cleaned after each test."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def test_create_workflow_job_returns_202_and_queued() -> None:
    wf_stub = _MockWorkflowUseCase(result=_make_sample_workflow_result())
    job_mgr = JobManager()
    event_bus = EventBus()
    runner = JobRunner(job_manager=job_mgr, event_bus=event_bus)

    app.dependency_overrides[get_run_workflow_use_case] = lambda: wf_stub
    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus
    app.dependency_overrides[get_job_runner] = lambda: runner

    client = TestClient(app)

    response = client.post("/workflow/jobs", json={"prompt": "Build task executor"})
    assert response.status_code == 202

    body = response.json()
    assert "job_id" in body
    assert body["job_id"]
    # The immediate response indicates status is 'queued'
    assert body["status"] == "queued"


def test_get_workflow_job_completed_result() -> None:
    sample_result = _make_sample_workflow_result()
    wf_stub = _MockWorkflowUseCase(result=sample_result)
    job_mgr = JobManager()
    event_bus = EventBus()
    runner = JobRunner(job_manager=job_mgr, event_bus=event_bus)

    app.dependency_overrides[get_run_workflow_use_case] = lambda: wf_stub
    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus
    app.dependency_overrides[get_job_runner] = lambda: runner

    client = TestClient(app)

    # 1. Post job
    create_resp = client.post("/workflow/jobs", json={"prompt": "Build task executor"})
    assert create_resp.status_code == 202
    job_id = create_resp.json()["job_id"]

    # 2. Get status (in TestClient, BackgroundTasks executed after response was sent)
    status_resp = client.get(f"/workflow/jobs/{job_id}")
    assert status_resp.status_code == 200

    job_data = status_resp.json()
    assert job_data["job_id"] == job_id
    assert job_data["status"] == "completed"
    assert job_data["created_at"]
    assert job_data["started_at"]
    assert job_data["completed_at"]
    assert job_data["error"] is None

    result = job_data["result"]
    assert result is not None
    assert result["success"] is True
    assert result["workflow_status"] == "completed"
    assert result["generated_code"] == "def run(): return 42\n"
    assert result["planning"]["problem_summary"] == "Asynchronous task executor"
    assert result["review"]["overall_score"] == 92


def test_failed_job_reaches_failed_and_stores_error() -> None:
    wf_stub = _MockWorkflowUseCase(exc=RuntimeError("AI model execution quota exceeded"))
    job_mgr = JobManager()
    event_bus = EventBus()
    runner = JobRunner(job_manager=job_mgr, event_bus=event_bus)

    app.dependency_overrides[get_run_workflow_use_case] = lambda: wf_stub
    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus
    app.dependency_overrides[get_job_runner] = lambda: runner

    client = TestClient(app)

    create_resp = client.post("/workflow/jobs", json={"prompt": "Crash workflow"})
    assert create_resp.status_code == 202
    job_id = create_resp.json()["job_id"]

    status_resp = client.get(f"/workflow/jobs/{job_id}")
    assert status_resp.status_code == 200

    job_data = status_resp.json()
    assert job_data["job_id"] == job_id
    assert job_data["status"] == "failed"
    assert "AI model execution quota exceeded" in job_data["error"]
    assert job_data["result"] is None


def test_unknown_job_returns_404() -> None:
    job_mgr = JobManager()
    app.dependency_overrides[get_job_manager] = lambda: job_mgr

    client = TestClient(app)
    resp = client.get("/workflow/jobs/nonexistent-job-uuid-1234")
    assert resp.status_code == 404
    assert "Job not found" in resp.json()["detail"]


def test_empty_prompt_returns_400() -> None:
    client = TestClient(app)

    resp1 = client.post("/workflow/jobs", json={"prompt": ""})
    assert resp1.status_code == 400

    resp2 = client.post("/workflow/jobs", json={"prompt": "    "})
    assert resp2.status_code == 400


def test_same_job_manager_instance_is_used() -> None:
    wf_stub = _MockWorkflowUseCase(result=_make_sample_workflow_result())
    job_mgr = JobManager()
    event_bus = EventBus()
    runner = JobRunner(job_manager=job_mgr, event_bus=event_bus)

    app.dependency_overrides[get_run_workflow_use_case] = lambda: wf_stub
    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus
    app.dependency_overrides[get_job_runner] = lambda: runner

    client = TestClient(app)

    create_resp = client.post("/workflow/jobs", json={"prompt": "Check singleton job manager"})
    job_id = create_resp.json()["job_id"]

    # Direct check on the injected JobManager instance
    job = job_mgr.get_job(job_id)
    assert job is not None
    assert job.job_id == job_id


def test_job_id_propagated_into_emitted_events() -> None:
    wf_stub = _MockWorkflowUseCase(result=_make_sample_workflow_result())
    job_mgr = JobManager()
    event_bus = EventBus()
    runner = JobRunner(job_manager=job_mgr, event_bus=event_bus)

    app.dependency_overrides[get_run_workflow_use_case] = lambda: wf_stub
    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus
    app.dependency_overrides[get_job_runner] = lambda: runner

    client = TestClient(app)

    create_resp = client.post("/workflow/jobs", json={"prompt": "Verify event job_id"})
    job_id = create_resp.json()["job_id"]

    # Ensure the use case received the exact job_id and an event emitter
    assert wf_stub.last_request is not None
    assert wf_stub.last_request.job_id == job_id
    assert wf_stub.last_request.event_emitter is not None


def test_no_secret_or_prompt_leaks_in_api_response() -> None:
    sensitive_prompt = "Build auth with SECRET_KEY=super_private_token_999"
    wf_stub = _MockWorkflowUseCase(result=_make_sample_workflow_result())
    job_mgr = JobManager()
    runner = JobRunner(job_manager=job_mgr)

    app.dependency_overrides[get_run_workflow_use_case] = lambda: wf_stub
    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_job_runner] = lambda: runner

    client = TestClient(app)

    create_resp = client.post("/workflow/jobs", json={"prompt": sensitive_prompt})
    job_id = create_resp.json()["job_id"]

    status_resp = client.get(f"/workflow/jobs/{job_id}")
    raw_response_text = status_resp.text

    assert "super_private_token_999" not in raw_response_text

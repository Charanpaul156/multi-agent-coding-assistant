from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import (
    get_advanced_orchestrator,
    get_event_bus,
    get_job_manager,
    get_job_runner,
)
from backend.api.job_schemas import AdvancedOrchestrationJobPayload
from backend.api.routes import create_advanced_orchestration_job
from backend.application.advanced_orchestration_jobs import (
    run_advanced_orchestration_job,
)
from backend.application.job_manager import JobManager
from backend.application.job_runner import JobRunner
from backend.domain.job_models import JobStatus, WorkflowEvent
from backend.domain.orchestration_models import (
    AgentExecutionRecord,
    AgentExecutionStatus,
    OrchestrationResult,
    WorkflowMode,
)
from backend.infrastructure.event_bus import EventBus
from backend.main import app


class _StubOrchestrator:
    def __init__(self, *, result: OrchestrationResult | None = None, error: Exception | None = None):
        self.result = result or OrchestrationResult(
            mode=WorkflowMode.DOCUMENTATION,
            success=False,
            summary="Documentation generation failed",
            error="No documentation returned",
        )
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def execute(self, prompt: str, **kwargs: Any) -> OrchestrationResult:
        self.calls.append({"prompt": prompt, **kwargs})
        if self.error:
            raise self.error
        emitter = kwargs.get("event_emitter")
        job_id = kwargs.get("job_id")
        if emitter is not None:
            emitter.emit(WorkflowEvent("step_start", job_id, payload={"step": "stub"}))
            emitter.emit(WorkflowEvent("step_complete", job_id, payload={"step": "stub"}))
        return self.result


@pytest.fixture(autouse=True)
def _clear_dependency_overrides():
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


def _completed_result() -> OrchestrationResult:
    return OrchestrationResult(
        mode=WorkflowMode.DOCUMENTATION,
        success=False,
        summary="The run returned with an orchestration failure.",
        agent_records=[
            AgentExecutionRecord(
                agent_name="DocumentationAgent",
                status=AgentExecutionStatus.FAILED,
                started_at="2025-01-01T00:00:00+00:00",
                completed_at="2025-01-01T00:00:01+00:00",
                duration_ms=1000,
                input_summary="prompt",
                output_summary="failed",
                error="agent failed",
            )
        ],
        total_duration_ms=1000,
        output_data={"documentation": None},
        error="agent failed",
    )


def test_submit_advanced_job_returns_202_and_stores_safe_metadata() -> None:
    manager = JobManager()
    bus = EventBus()
    runner = JobRunner(manager, bus, bus)
    orchestrator = _StubOrchestrator(result=_completed_result())
    app.dependency_overrides[get_job_manager] = lambda: manager
    app.dependency_overrides[get_event_bus] = lambda: bus
    app.dependency_overrides[get_job_runner] = lambda: runner
    app.dependency_overrides[get_advanced_orchestrator] = lambda: orchestrator

    response = TestClient(app).post(
        "/advanced-orchestration/jobs",
        json={
            "mode": "documentation",
            "prompt": "Document this private project prompt",
            "code_input": "private source text",
            "repository_path": "C:/private/project",
            "language": "python",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    job = manager.get_job(body["job_id"])
    assert job is not None
    assert job.status == JobStatus.COMPLETED
    assert job.metadata == {
        "job_type": "advanced_orchestration",
        "mode": "documentation",
        "prompt_length": len("Document this private project prompt"),
        "code_input_length": len("private source text"),
        "files_count": 0,
        "files_total_chars": 0,
    }
    assert "private" not in str(job.metadata)
    assert "repository_path" not in job.metadata
    assert job.result.success is False
    assert job.status == JobStatus.COMPLETED


def test_submission_keeps_job_queued_until_background_task_runs() -> None:
    manager = JobManager()
    runner = JobRunner(manager)
    orchestrator = _StubOrchestrator()
    from fastapi import BackgroundTasks

    tasks = BackgroundTasks()
    response = create_advanced_orchestration_job(
        AdvancedOrchestrationJobPayload(
            mode=WorkflowMode.DOCUMENTATION,
            prompt="document",
            code_input="source",
        ),
        tasks,
        runner,
        orchestrator,  # type: ignore[arg-type]
    )

    job = manager.get_job(response.job_id)
    assert job is not None
    assert response.status == "queued"
    assert job.status == JobStatus.QUEUED
    assert len(tasks.tasks) == 1


@pytest.mark.parametrize("target_status", [JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.FAILED])
def test_advanced_status_returns_actual_job_lifecycle(target_status: JobStatus) -> None:
    manager = JobManager()
    job = manager.create_job(
        metadata={"job_type": "advanced_orchestration", "mode": "documentation"}
    )
    if target_status == JobStatus.RUNNING:
        manager.start_job(job.job_id)
    elif target_status == JobStatus.FAILED:
        manager.start_job(job.job_id)
        manager.fail_job(job.job_id, error="unexpected task error")
    app.dependency_overrides[get_job_manager] = lambda: manager

    response = TestClient(app).get(f"/advanced-orchestration/jobs/{job.job_id}")

    assert response.status_code == 200
    assert response.json()["status"] == target_status.value
    if target_status == JobStatus.FAILED:
        assert response.json()["error"] == "unexpected task error"


def test_advanced_status_serializes_result_even_when_orchestration_failed() -> None:
    manager = JobManager()
    job = manager.create_job(
        metadata={"job_type": "advanced_orchestration", "mode": "documentation"}
    )
    manager.start_job(job.job_id)
    manager.complete_job(job.job_id, result=_completed_result())
    app.dependency_overrides[get_job_manager] = lambda: manager

    response = TestClient(app).get(f"/advanced-orchestration/jobs/{job.job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["result"]["success"] is False
    assert body["result"]["mode"] == "documentation"
    assert body["result"]["error"] == "agent failed"
    assert body["result"]["agent_records"][0]["status"] == "failed"


def test_advanced_status_unknown_or_standard_job_is_not_found() -> None:
    manager = JobManager()
    standard_job = manager.create_job(metadata={"prompt_length": 8})
    app.dependency_overrides[get_job_manager] = lambda: manager
    client = TestClient(app)

    assert client.get("/advanced-orchestration/jobs/unknown").status_code == 404
    assert client.get(
        f"/advanced-orchestration/jobs/{standard_job.job_id}"
    ).status_code == 404


@pytest.mark.anyio
async def test_unexpected_task_exception_marks_job_failed_without_exposure(
    caplog,
) -> None:
    sentinel = "ADVANCED_JOB_EXCEPTION_SECRET_SENTINEL_789"
    safe_error = "Advanced orchestration job failed. See server diagnostics for details."
    manager = JobManager()
    bus = EventBus()
    runner = JobRunner(manager, bus, bus)
    job = runner.create_job(metadata={"job_type": "advanced_orchestration"})
    subscription = bus.subscribe(job.job_id)

    run_advanced_orchestration_job(
        runner,
        _StubOrchestrator(error=RuntimeError(sentinel)),  # type: ignore[arg-type]
        job.job_id,
        mode=WorkflowMode.DOCUMENTATION,
        prompt="document",
        code_input="source",
    )

    event = await subscription.get(timeout=1)
    assert event is not None
    assert event.event_type == "job_failed"
    assert event.payload["error"] == safe_error
    assert await subscription.get(timeout=1) is None
    subscription.close()

    failed = manager.get_job(job.job_id)
    assert failed is not None
    assert failed.status == JobStatus.FAILED
    assert failed.error == safe_error
    assert sentinel not in caplog.text
    assert "RuntimeError" in caplog.text

    app.dependency_overrides[get_job_manager] = lambda: manager
    app.dependency_overrides[get_event_bus] = lambda: bus
    client = TestClient(app)

    status_response = client.get(f"/advanced-orchestration/jobs/{job.job_id}")
    assert status_response.status_code == 200
    assert status_response.json()["error"] == safe_error

    stream_response = client.get(
        f"/advanced-orchestration/jobs/{job.job_id}/stream"
    )
    assert stream_response.status_code == 200
    assert "job_failed" in stream_response.text
    assert sentinel not in stream_response.text
    event_data = json.loads(stream_response.text.split("data: ", 1)[1].strip())
    assert event_data["payload"]["error"] == safe_error


@pytest.mark.anyio
async def test_task_uses_scoped_emitter_and_publishes_step_events() -> None:
    manager = JobManager()
    bus = EventBus()
    runner = JobRunner(manager, bus, bus)
    job = runner.create_job(metadata={"job_type": "advanced_orchestration"})
    subscription = bus.subscribe(job.job_id)

    run_advanced_orchestration_job(
        runner,
        _StubOrchestrator(),  # type: ignore[arg-type]
        job.job_id,
        mode=WorkflowMode.DOCUMENTATION,
        prompt="document",
        code_input="source",
    )

    events = []
    while True:
        event = await subscription.get(timeout=1)
        if event is None:
            break
        events.append(event)
    subscription.close()

    assert [event.event_type for event in events] == [
        "step_start",
        "step_complete",
        "job_completed",
    ]
    assert all(event.job_id == job.job_id for event in events)

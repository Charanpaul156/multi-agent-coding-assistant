"""Tests for Server-Sent Events (SSE) real-time workflow streaming.

Verifies:
1. Unknown job returns 404
2. SSE response has text/event-stream and appropriate headers
3. One published event is streamed correctly
4. Multiple events preserve order
5. Event JSON is valid
6. SSE framing uses double newline (\\n\\n)
7. job_completed closes the stream
8. job_failed closes the stream
9. Subscription cleanup occurs (no leaked subscribers/queues)
10. Client disconnect does not affect the job
11. Terminal job does not cause infinite waiting
12. No secrets/approval tokens appear in SSE output
13. Existing job status API still works
14. Existing synchronous /run-workflow still works
15. Job ID in stream matches requested job ID
16. EventBus completion signaling terminates the stream
17. Multiple concurrent SSE subscribers receive events independently
"""

from __future__ import annotations

import json
import threading
import time
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
from backend.domain.job_models import JobStatus, WorkflowEvent, WorkflowJob
from backend.infrastructure.event_bus import EventBus
from backend.main import app
from backend.tools.python_executor import ExecutionResponse
from backend.tools.test_executor import TestExecutionResponse


# ---------------------------------------------------------------------------
# Test Helpers
# ---------------------------------------------------------------------------

def _parse_sse_events(sse_text: str) -> list[dict[str, Any]]:
    """Parse raw SSE stream text into structured event dictionaries."""
    events = []
    # SSE blocks are delimited by double newline
    blocks = sse_text.split("\n\n")
    for block in blocks:
        stripped = block.strip()
        if not stripped or stripped.startswith(":"):
            # Empty or SSE comment (e.g. keepalive)
            continue
        ev_type = None
        data_obj = None
        for line in stripped.split("\n"):
            if line.startswith("event:"):
                ev_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw_data = line[len("data:"):].strip()
                data_obj = json.loads(raw_data)
        if ev_type is not None or data_obj is not None:
            events.append({"event": ev_type, "data": data_obj})
    return events


def _make_sample_workflow_result() -> WorkflowResult:
    return WorkflowResult(
        success=True,
        workflow_status=WorkflowStatus.COMPLETED,
        planning=ImplementationPlan(
            problem_summary="Plan",
            project_type="module",
            requirements=[],
            modules=[],
            functions=[],
            classes=[],
            external_libraries=[],
            database_needed=False,
            api_needed=[],
            algorithm="algo",
            edge_cases=[],
            estimated_complexity="low",
            future_improvements=[],
        ),
        generated_code="def run(): return 42\n",
        generated_tests="def test_run(): assert True\n",
        execution=ExecutionResponse(True, "42", "", 5.0, 0),
        test_execution=TestExecutionResponse(True, "1 passed", "", 8.0, 0, 1, 0),
        review=ReviewReport(
            overall_score=90,
            strengths=[],
            weaknesses=[],
            pep8_issues=[],
            performance_suggestions=[],
            security_concerns=[],
            logic_issues=[],
            maintainability=[],
            error_handling=[],
            recommendations=[],
            final_summary="good",
        ),
        execution_time_ms=13.0,
    )


class _StubWorkflowUseCase(RunWorkflowUseCase):
    def __init__(self, result: WorkflowResult | None = None, exc: Exception | None = None):
        self.result = result
        self.exc = exc
        self.call_count = 0

    def execute(self, request: WorkflowRequest) -> WorkflowResult:
        self.call_count += 1
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result


@pytest.fixture(autouse=True)
def _clean_di_overrides():
    """Ensure clean dependency overrides per test."""
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_unknown_job_returns_404() -> None:
    """Requirement 1: Unknown job returns 404."""
    job_mgr = JobManager()
    event_bus = EventBus()
    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)
    resp = client.get("/workflow/jobs/nonexistent-id-1234/stream")
    assert resp.status_code == 404
    assert "Job not found" in resp.json()["detail"]


def test_sse_response_has_text_event_stream() -> None:
    """Requirement 2: SSE response has text/event-stream and appropriate headers."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_headers_test")
    job_mgr.start_job(job.job_id)
    job_mgr.complete_job(job.job_id, result={"done": True})

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)
    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "no-cache" in resp.headers.get("cache-control", "")
    assert resp.headers.get("x-accel-buffering") == "no"


def test_one_published_event_streamed_correctly() -> None:
    """Requirement 3: One published event is streamed correctly."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_one_ev")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    def publish_worker():
        time.sleep(0.05)
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="step_start",
                job_id=job.job_id,
                payload={"step": "planner", "detail": "analyzing prompt"},
            )
        )
        time.sleep(0.02)
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="job_completed",
                job_id=job.job_id,
                payload={"job_id": job.job_id, "status": "completed"},
            )
        )

    t = threading.Thread(target=publish_worker)
    t.start()

    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    t.join()

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert len(events) == 2

    assert events[0]["event"] == "step_start"
    assert events[0]["data"]["event_type"] == "step_start"
    assert events[0]["data"]["job_id"] == job.job_id
    assert events[0]["data"]["payload"] == {"step": "planner", "detail": "analyzing prompt"}

    assert events[1]["event"] == "job_completed"
    assert events[1]["data"]["event_type"] == "job_completed"


def test_multiple_events_preserve_order() -> None:
    """Requirement 4: Multiple events preserve publication order."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_order_test")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    expected_steps = [
        ("workflow_started", {"repo": None}),
        ("step_start", {"step": "planner"}),
        ("step_complete", {"step": "planner"}),
        ("step_start", {"step": "coder"}),
        ("step_complete", {"step": "coder"}),
        ("step_start", {"step": "test_generator"}),
        ("step_complete", {"step": "test_generator"}),
        ("step_start", {"step": "executor"}),
        ("test_result", {"passed": 3, "failed": 0}),
        ("step_start", {"step": "reviewer"}),
        ("review_result", {"score": 95}),
        ("job_completed", {"job_id": job.job_id, "status": "completed"}),
    ]

    def publish_ordered_events():
        time.sleep(0.05)
        for ev_type, payload in expected_steps:
            event_bus.publish_sync(
                WorkflowEvent(
                    event_type=ev_type,
                    job_id=job.job_id,
                    payload=payload,
                )
            )
            time.sleep(0.01)

    t = threading.Thread(target=publish_ordered_events)
    t.start()

    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    t.join()

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert len(events) == len(expected_steps)

    for i, (expected_type, expected_payload) in enumerate(expected_steps):
        assert events[i]["event"] == expected_type
        assert events[i]["data"]["event_type"] == expected_type
        assert events[i]["data"]["payload"] == expected_payload


def test_event_json_is_valid() -> None:
    """Requirement 5: Event JSON is valid in every SSE message."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_json_valid")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    def publish_worker():
        time.sleep(0.05)
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="iteration_progress",
                job_id=job.job_id,
                payload={"iteration": 1, "nested": {"valid": True, "counts": [1, 2, 3]}},
            )
        )
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="job_completed",
                job_id=job.job_id,
                payload={"job_id": job.job_id, "status": "completed"},
            )
        )

    t = threading.Thread(target=publish_worker)
    t.start()

    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    t.join()

    # Manually check every data: line
    lines = resp.text.split("\n")
    data_lines = [l for l in lines if l.startswith("data: ")]
    assert len(data_lines) == 2
    for line in data_lines:
        json_str = line[len("data: "):]
        parsed = json.loads(json_str)
        assert "event_type" in parsed
        assert "job_id" in parsed
        assert "timestamp" in parsed
        assert "payload" in parsed


def test_sse_framing_uses_double_newline() -> None:
    """Requirement 6: SSE framing conforms to standard event: ... \\n data: ... \\n\\n."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_framing_test")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    def publish_worker():
        time.sleep(0.05)
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="step_start",
                job_id=job.job_id,
                payload={"step": "coder"},
            )
        )
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="job_completed",
                job_id=job.job_id,
                payload={"job_id": job.job_id, "status": "completed"},
            )
        )

    t = threading.Thread(target=publish_worker)
    t.start()

    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    t.join()

    raw_text = resp.text
    # Each message ends with \n\n
    assert raw_text.endswith("\n\n")
    # Splitting by \n\n should yield valid individual messages
    blocks = [b for b in raw_text.split("\n\n") if b]
    assert len(blocks) == 2
    for block in blocks:
        lines = block.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")


def test_job_completed_closes_stream() -> None:
    """Requirement 7: job_completed event terminates the SSE stream cleanly."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_completed_term")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    def publish_worker():
        time.sleep(0.05)
        event_bus.publish_sync(
            WorkflowEvent(event_type="step_start", job_id=job.job_id, payload={})
        )
        time.sleep(0.02)
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="job_completed",
                job_id=job.job_id,
                payload={"status": "completed"},
            )
        )

    t = threading.Thread(target=publish_worker)
    t.start()

    start_time = time.monotonic()
    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    elapsed = time.monotonic() - start_time
    t.join()

    # Stream returned and did not hang indefinitely
    assert elapsed < 3.0
    events = _parse_sse_events(resp.text)
    assert len(events) == 2
    assert events[-1]["event"] == "job_completed"


def test_job_failed_closes_stream() -> None:
    """Requirement 8: job_failed event terminates the SSE stream cleanly."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_failed_term")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    def publish_worker():
        time.sleep(0.05)
        event_bus.publish_sync(
            WorkflowEvent(event_type="step_start", job_id=job.job_id, payload={})
        )
        time.sleep(0.02)
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="job_failed",
                job_id=job.job_id,
                payload={"status": "failed", "error": "Execution timeout"},
            )
        )

    t = threading.Thread(target=publish_worker)
    t.start()

    start_time = time.monotonic()
    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    elapsed = time.monotonic() - start_time
    t.join()

    assert elapsed < 3.0
    events = _parse_sse_events(resp.text)
    assert len(events) == 2
    assert events[-1]["event"] == "job_failed"
    assert events[-1]["data"]["payload"]["error"] == "Execution timeout"


def test_subscription_cleanup_occurs() -> None:
    """Requirement 9: Subscriptions are cleaned up when the stream finishes."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_cleanup_test")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    assert event_bus.subscriber_count(job.job_id) == 0

    client = TestClient(app)

    def publish_worker():
        time.sleep(0.05)
        # Subscriber should be active while connection is open
        assert event_bus.subscriber_count(job.job_id) == 1
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="job_completed",
                job_id=job.job_id,
                payload={"status": "completed"},
            )
        )

    t = threading.Thread(target=publish_worker)
    t.start()

    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    t.join()

    assert resp.status_code == 200
    # After stream finishes, subscriber must be cleanly unsubscribed
    assert event_bus.subscriber_count(job.job_id) == 0


@pytest.mark.anyio
async def test_client_disconnect_does_not_affect_job() -> None:
    """Requirement 10: Client disconnect cleans up subscription and does not cancel the job."""
    from unittest.mock import AsyncMock
    from backend.api.routes import stream_workflow_events

    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_disconnect_test")
    job_mgr.start_job(job.job_id)

    # Mock request simulating client disconnecting on next iteration
    mock_req = AsyncMock()
    mock_req.is_disconnected.side_effect = [False, True]

    # stream_workflow_events registers the EventBus subscription
    resp = await stream_workflow_events(job.job_id, mock_req, job_mgr, event_bus)
    assert event_bus.subscriber_count(job.job_id) == 1

    # Publish an event into the active subscription
    event_bus.publish_sync(
        WorkflowEvent(
            event_type="step_start",
            job_id=job.job_id,
            payload={"step": "active_work"},
        )
    )

    # Consume chunk; iterator cleanly exits when is_disconnected is detected
    chunks = []
    async for chunk in resp.body_iterator:
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "step_start" in chunks[0]

    # 1. Subscription must be cleanly unsubscribed on disconnect
    assert event_bus.subscriber_count(job.job_id) == 0

    # 2. Job in JobManager must remain active (not cancelled or failed)
    current_job = job_mgr.get_job(job.job_id)
    assert current_job is not None
    assert current_job.status == JobStatus.RUNNING


def test_terminal_job_does_not_wait_forever() -> None:
    """Requirement 11: A job already in terminal state sends final event and terminates immediately."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_already_terminal")
    job_mgr.start_job(job.job_id)
    job_mgr.complete_job(job.job_id, result={"result": 123})

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    start = time.monotonic()
    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    elapsed = time.monotonic() - start

    assert resp.status_code == 200
    assert elapsed < 0.5  # Returns immediately without waiting
    events = _parse_sse_events(resp.text)
    assert len(events) == 1
    assert events[0]["event"] == "job_completed"
    assert events[0]["data"]["payload"]["status"] == "completed"

    # Also test an already-failed job
    failed_job = job_mgr.create_job(job_id="job_already_failed")
    job_mgr.start_job(failed_job.job_id)
    job_mgr.fail_job(failed_job.job_id, error="Pre-existing failure")

    resp_failed = client.get(f"/workflow/jobs/{failed_job.job_id}/stream")
    assert resp_failed.status_code == 200
    failed_events = _parse_sse_events(resp_failed.text)
    assert len(failed_events) == 1
    assert failed_events[0]["event"] == "job_failed"
    assert failed_events[0]["data"]["payload"]["status"] == "failed"
    assert failed_events[0]["data"]["payload"]["error"] == "Pre-existing failure"


def test_no_secrets_or_approval_tokens_in_sse_output() -> None:
    """Requirement 12 & 10: Approval tokens, secrets, and credentials never appear in SSE output."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_sec_audit")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    sensitive_token = "secret_approval_token_xyz_998877"
    sensitive_secret = "Bearer super_private_credential"

    def publish_worker():
        time.sleep(0.05)
        # Attempt to publish events containing sensitive fields
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="approval_required",
                job_id=job.job_id,
                payload={
                    "requires_approval": True,
                    "approval_status": "preview",
                    "approval_token": sensitive_token,
                    "ticket_token": sensitive_token,
                    "token": sensitive_token,
                },
            )
        )
        event_bus.publish_sync(
            WorkflowEvent(
                event_type="job_completed",
                job_id=job.job_id,
                payload={"status": "completed"},
            )
        )

    t = threading.Thread(target=publish_worker)
    t.start()

    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    t.join()

    raw_sse_text = resp.text

    # Invariant: Approval token and credentials must NEVER leak into the SSE wire output
    assert sensitive_token not in raw_sse_text
    assert sensitive_secret not in raw_sse_text
    assert "approval_token" not in raw_sse_text
    assert "ticket_token" not in raw_sse_text

    # Verify standard safe payload fields remain
    events = _parse_sse_events(raw_sse_text)
    assert len(events) == 2
    assert events[0]["event"] == "approval_required"
    assert events[0]["data"]["payload"]["requires_approval"] is True


def test_existing_job_status_api_still_works() -> None:
    """Requirement 13: Existing POST /workflow/jobs and GET /workflow/jobs/{job_id} remain intact."""
    wf_stub = _StubWorkflowUseCase(result=_make_sample_workflow_result())
    job_mgr = JobManager()
    event_bus = EventBus()
    runner = JobRunner(job_manager=job_mgr, event_bus=event_bus)

    app.dependency_overrides[get_run_workflow_use_case] = lambda: wf_stub
    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus
    app.dependency_overrides[get_job_runner] = lambda: runner

    client = TestClient(app)

    create_resp = client.post("/workflow/jobs", json={"prompt": "Build task executor"})
    assert create_resp.status_code == 202
    job_id = create_resp.json()["job_id"]

    status_resp = client.get(f"/workflow/jobs/{job_id}")
    assert status_resp.status_code == 200
    body = status_resp.json()
    assert body["job_id"] == job_id
    assert body["status"] == "completed"
    assert body["result"]["success"] is True


def test_existing_synchronous_run_workflow_still_works() -> None:
    """Requirement 14: Existing synchronous POST /run-workflow still works."""
    wf_stub = _StubWorkflowUseCase(result=_make_sample_workflow_result())
    app.dependency_overrides[get_run_workflow_use_case] = lambda: wf_stub

    client = TestClient(app)
    resp = client.post("/run-workflow", json={"prompt": "Run synchronous workflow"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["workflow"]["workflow_status"] == "completed"
    assert body["workflow"]["generated_code"] == "def run(): return 42\n"


def test_job_id_in_stream_matches_requested_job_id() -> None:
    """Requirement 15: The job_id in every streamed event matches the requested job_id."""
    requested_id = "target_workflow_job_42"
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id=requested_id)
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    def publish_worker():
        time.sleep(0.05)
        event_bus.publish_sync(
            WorkflowEvent(event_type="step_start", job_id=requested_id, payload={"step": "init"})
        )
        event_bus.publish_sync(
            WorkflowEvent(event_type="job_completed", job_id=requested_id, payload={"status": "completed"})
        )

    t = threading.Thread(target=publish_worker)
    t.start()

    resp = client.get(f"/workflow/jobs/{requested_id}/stream")
    t.join()

    events = _parse_sse_events(resp.text)
    assert len(events) == 2
    for ev in events:
        assert ev["data"]["job_id"] == requested_id


def test_event_bus_completion_signal_closes_stream() -> None:
    """Requirement 16: EventBus signal_complete_sync closes the consumer stream."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_signal_complete")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    def publish_worker():
        time.sleep(0.05)
        event_bus.publish_sync(
            WorkflowEvent(event_type="step_start", job_id=job.job_id, payload={"step": "custom"})
        )
        time.sleep(0.02)
        # Signal complete directly without a terminal event
        event_bus.signal_complete_sync(job.job_id)

    t = threading.Thread(target=publish_worker)
    t.start()

    resp = client.get(f"/workflow/jobs/{job.job_id}/stream")
    t.join()

    events = _parse_sse_events(resp.text)
    assert len(events) == 1
    assert events[0]["event"] == "step_start"
    assert event_bus.subscriber_count(job.job_id) == 0


def test_multiple_concurrent_subscribers() -> None:
    """Requirement 17: Multiple independent SSE subscribers stream simultaneously."""
    job_mgr = JobManager()
    event_bus = EventBus()
    job = job_mgr.create_job(job_id="job_multi_sub")
    job_mgr.start_job(job.job_id)

    app.dependency_overrides[get_job_manager] = lambda: job_mgr
    app.dependency_overrides[get_event_bus] = lambda: event_bus

    client = TestClient(app)

    results: dict[str, str] = {}

    def subscriber_client(client_id: str):
        c = TestClient(app)
        resp = c.get(f"/workflow/jobs/{job.job_id}/stream")
        results[client_id] = resp.text

    t1 = threading.Thread(target=subscriber_client, args=("sub1",))
    t2 = threading.Thread(target=subscriber_client, args=("sub2",))

    t1.start()
    t2.start()

    # Wait for both subscribers to register
    time.sleep(0.08)
    assert event_bus.subscriber_count(job.job_id) == 2

    # Publish events to both
    event_bus.publish_sync(
        WorkflowEvent(event_type="step_start", job_id=job.job_id, payload={"step": "shared"})
    )
    event_bus.publish_sync(
        WorkflowEvent(event_type="job_completed", job_id=job.job_id, payload={"status": "completed"})
    )

    t1.join()
    t2.join()

    assert event_bus.subscriber_count(job.job_id) == 0
    assert len(results) == 2

    events1 = _parse_sse_events(results["sub1"])
    events2 = _parse_sse_events(results["sub2"])

    assert len(events1) == 2
    assert len(events2) == 2
    assert events1[0]["event"] == "step_start"
    assert events2[0]["event"] == "step_start"

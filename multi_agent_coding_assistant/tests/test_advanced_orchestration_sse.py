from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from backend.api.routes import (
    _format_sse_event,
    stream_advanced_orchestration_events,
)
from backend.application.job_manager import JobManager
from backend.domain.job_models import JobStatus, WorkflowEvent
from backend.infrastructure.event_bus import EventBus


class _RequestStub:
    def __init__(self, *, disconnected: bool = False) -> None:
        self.is_disconnected = AsyncMock(return_value=disconnected)


def _advanced_job(manager: JobManager, job_id: str, *, terminal: str | None = None):
    job = manager.create_job(
        job_id=job_id,
        metadata={"job_type": "advanced_orchestration", "mode": "codebase_chat"},
    )
    if terminal is not None:
        manager.start_job(job_id)
        if terminal == "completed":
            manager.complete_job(job_id, result={"done": True})
        else:
            manager.fail_job(job_id, error="task failed")
    else:
        manager.start_job(job_id)
    return job


async def _read_stream(response: StreamingResponse) -> str:
    chunks: list[str] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, bytes) else chunk)
    return "".join(chunks)


def _decode_sse(raw: str) -> list[dict]:
    output = []
    for block in raw.strip().split("\n\n"):
        if not block or block.startswith(":"):
            continue
        event_name = next(line[7:] for line in block.splitlines() if line.startswith("event: "))
        data = next(line[6:] for line in block.splitlines() if line.startswith("data: "))
        output.append({"event": event_name, "data": json.loads(data)})
    return output


@pytest.mark.anyio
async def test_advanced_stream_delivers_scoped_step_events_and_terminal_event() -> None:
    manager = JobManager()
    bus = EventBus()
    job = _advanced_job(manager, "advanced_stream_steps")
    response = await stream_advanced_orchestration_events(
        job.job_id, _RequestStub(), manager, bus  # type: ignore[arg-type]
    )
    assert bus.subscriber_count(job.job_id) == 1

    for event_type in ("step_start", "step_complete", "job_completed"):
        bus.publish_sync(
            WorkflowEvent(
                event_type,
                job.job_id,
                payload={"step": "ResearcherAgent"},
            )
        )

    raw = await _read_stream(response)
    events = _decode_sse(raw)
    assert [event["event"] for event in events] == [
        "step_start",
        "step_complete",
        "job_completed",
    ]
    assert all(event["data"]["job_id"] == job.job_id for event in events)
    assert raw.endswith("\n\n")
    assert bus.subscriber_count(job.job_id) == 0


@pytest.mark.anyio
async def test_advanced_stream_does_not_receive_another_jobs_events() -> None:
    manager = JobManager()
    bus = EventBus()
    job = _advanced_job(manager, "advanced_target_job")
    _advanced_job(manager, "advanced_other_job")
    response = await stream_advanced_orchestration_events(
        job.job_id, _RequestStub(), manager, bus  # type: ignore[arg-type]
    )

    bus.publish_sync(
        WorkflowEvent("step_start", "advanced_other_job", payload={"step": "other"})
    )
    bus.publish_sync(
        WorkflowEvent("step_start", job.job_id, payload={"step": "target"})
    )
    bus.publish_sync(WorkflowEvent("job_completed", job.job_id))

    events = _decode_sse(await _read_stream(response))
    assert [event["data"]["payload"].get("step") for event in events] == [
        "target",
        None,
    ]


@pytest.mark.anyio
async def test_unknown_or_standard_job_is_not_found() -> None:
    manager = JobManager()
    standard = manager.create_job(job_id="standard_stream_job")
    bus = EventBus()

    with pytest.raises(HTTPException) as missing:
        await stream_advanced_orchestration_events(
            "unknown", _RequestStub(), manager, bus  # type: ignore[arg-type]
        )
    assert missing.value.status_code == 404

    with pytest.raises(HTTPException) as wrong_kind:
        await stream_advanced_orchestration_events(
            standard.job_id, _RequestStub(), manager, bus  # type: ignore[arg-type]
        )
    assert wrong_kind.value.status_code == 404


@pytest.mark.anyio
@pytest.mark.parametrize("terminal", ["completed", "failed"])
async def test_already_terminal_advanced_job_returns_terminal_event(terminal: str) -> None:
    manager = JobManager()
    bus = EventBus()
    job = _advanced_job(manager, f"advanced_terminal_{terminal}", terminal=terminal)

    response = await stream_advanced_orchestration_events(
        job.job_id, _RequestStub(), manager, bus  # type: ignore[arg-type]
    )
    events = _decode_sse(await _read_stream(response))

    assert [event["event"] for event in events] == [
        "job_completed" if terminal == "completed" else "job_failed"
    ]
    assert bus.subscriber_count(job.job_id) == 0


@pytest.mark.anyio
async def test_active_stream_closes_on_terminal_event_and_cleans_subscription() -> None:
    manager = JobManager()
    bus = EventBus()
    job = _advanced_job(manager, "advanced_terminal_live")
    response = await stream_advanced_orchestration_events(
        job.job_id, _RequestStub(), manager, bus  # type: ignore[arg-type]
    )
    bus.publish_sync(WorkflowEvent("step_complete", job.job_id, payload={"step": "done"}))
    bus.publish_sync(WorkflowEvent("job_completed", job.job_id))

    events = _decode_sse(await _read_stream(response))

    assert len(events) == 2
    assert bus.subscriber_count(job.job_id) == 0


@pytest.mark.anyio
async def test_client_disconnect_cleans_subscription_without_cancelling_job() -> None:
    manager = JobManager()
    bus = EventBus()
    job = _advanced_job(manager, "advanced_disconnect")
    response = await stream_advanced_orchestration_events(
        job.job_id,
        _RequestStub(disconnected=True),
        manager,
        bus,  # type: ignore[arg-type]
    )
    assert bus.subscriber_count(job.job_id) == 1

    assert await _read_stream(response) == ""

    assert bus.subscriber_count(job.job_id) == 0
    assert manager.get_job(job.job_id).status == JobStatus.RUNNING


def test_sse_filter_removes_nested_sensitive_fields() -> None:
    event = WorkflowEvent(
        "step_complete",
        "advanced_sensitive",
        payload={
            "step": "ResearcherAgent",
            "nested": {
                "approval_token": "approval-secret",
                "prompt": "private prompt",
                "source_code": "private source",
                "credentials": {"api_key": "private key"},
                "safe": "visible",
            },
        },
    )

    raw = _format_sse_event(event)
    parsed = _decode_sse(raw)[0]

    assert parsed["data"]["payload"] == {
        "step": "ResearcherAgent",
        "nested": {"safe": "visible"},
    }
    for sensitive in ("approval-secret", "private prompt", "private source", "private key"):
        assert sensitive not in raw

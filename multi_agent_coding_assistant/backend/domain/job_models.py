"""Domain models for background workflow jobs and real-time event streaming.

Framework-agnostic immutable dataclasses. No FastAPI, Pydantic, or Streamlit dependencies.
These models represent asynchronous execution jobs, lifecycle states, and structured
observable progress events emitted during agent workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping


def _utc_now() -> datetime:
    """Return the current datetime in UTC timezone."""
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    """Lifecycle states for background workflow execution jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """Return True if this status represents a finished terminal state."""
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)

    @property
    def is_active(self) -> bool:
        """Return True if this status represents an active non-terminal state."""
        return self in (JobStatus.QUEUED, JobStatus.RUNNING)


@dataclass(frozen=True)
class WorkflowJob:
    """An asynchronous workflow execution job entity.

    Attributes:
        job_id: Unique string identifier for the job.
        status: Current lifecycle JobStatus.
        created_at: UTC timestamp when the job was created.
        started_at: Optional UTC timestamp when execution began.
        completed_at: Optional UTC timestamp when the job finished.
        result: Optional JSON-serializable result payload.
        error: Optional error message string if failed or cancelled.
        metadata: Optional dictionary of immutable metadata.
    """

    job_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = field(default_factory=_utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: Any | None = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowEvent:
    """A structured observable progress event emitted during workflow execution.

    Events describe external observable progress (e.g. step started, tests run),
    never internal secrets, credentials, or hidden chain-of-thought tokens.

    Attributes:
        event_type: Observable event name (e.g., 'step_start', 'step_complete',
            'iteration_progress', 'test_result', 'review_result', 'job_completed',
            'job_failed').
        job_id: The job_id this event belongs to.
        timestamp: UTC datetime when the event occurred.
        payload: Optional structured JSON-serializable dictionary with event details.
    """

    event_type: str
    job_id: str
    timestamp: datetime = field(default_factory=_utc_now)
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize event to a JSON-compatible dictionary."""
        return {
            "event_type": self.event_type,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": dict(self.payload),
        }

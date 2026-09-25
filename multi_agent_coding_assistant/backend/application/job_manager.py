"""Application-layer JobManager for background workflow execution lifecycle.

Provides in-memory job state management, strict lifecycle transition validation,
result/error capture, and TTL expiration cleanup.
"""

from __future__ import annotations

import logging
import secrets
import threading
from datetime import datetime, timezone
from typing import Any, Mapping

from backend.domain.job_models import JobStatus, WorkflowJob

logger = logging.getLogger(__name__)

DEFAULT_JOB_TTL_SECONDS = 3600.0  # 1 hour


class JobManagerError(Exception):
    """Base exception for JobManager errors."""


class JobNotFoundError(JobManagerError, KeyError):
    """Raised when a job_id does not exist."""


class InvalidJobTransitionError(JobManagerError, ValueError):
    """Raised when attempting an illegal lifecycle transition."""


# Explicit allowed state transitions map: source_status -> set of allowed destination statuses
_ALLOWED_TRANSITIONS: dict[JobStatus, set[JobStatus]] = {
    JobStatus.QUEUED: {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.RUNNING: {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED},
    JobStatus.COMPLETED: set(),  # Terminal
    JobStatus.FAILED: set(),     # Terminal
    JobStatus.CANCELLED: set(),  # Terminal
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobManager:
    """In-memory coordinator of asynchronous workflow execution jobs."""

    def __init__(self, default_ttl_seconds: float = DEFAULT_JOB_TTL_SECONDS) -> None:
        self._default_ttl_seconds = max(0.0, float(default_ttl_seconds))
        self._jobs: dict[str, WorkflowJob] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Job Creation & Retrieval
    # ------------------------------------------------------------------ #

    def create_job(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        job_id: str | None = None,
    ) -> WorkflowJob:
        """Create a new job in the QUEUED state."""
        gid = job_id or f"job_{secrets.token_urlsafe(16)}"
        job = WorkflowJob(
            job_id=gid,
            status=JobStatus.QUEUED,
            created_at=_utc_now(),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._jobs[gid] = job
        logger.info("JobManager: created job %s", gid)
        return job

    def get_job(self, job_id: str) -> WorkflowJob | None:
        """Retrieve a job by id, or return None if not found."""
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, *, status: JobStatus | None = None) -> list[WorkflowJob]:
        """List all known jobs, optionally filtered by status."""
        with self._lock:
            jobs = list(self._jobs.values())
        if status is not None:
            return [j for j in jobs if j.status == status]
        return jobs

    # ------------------------------------------------------------------ #
    # Lifecycle Transitions
    # ------------------------------------------------------------------ #

    def _validate_transition(self, current: JobStatus, target: JobStatus, job_id: str) -> None:
        allowed = _ALLOWED_TRANSITIONS.get(current, set())
        if target not in allowed:
            raise InvalidJobTransitionError(
                f"Cannot transition job '{job_id}' from {current.value} to {target.value}"
            )

    def start_job(self, job_id: str) -> WorkflowJob:
        """Transition a job from QUEUED to RUNNING."""
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise JobNotFoundError(f"Job not found: {job_id}")

            self._validate_transition(current.status, JobStatus.RUNNING, job_id)
            updated = WorkflowJob(
                job_id=current.job_id,
                status=JobStatus.RUNNING,
                created_at=current.created_at,
                started_at=_utc_now(),
                metadata=current.metadata,
            )
            self._jobs[job_id] = updated
            return updated

    def complete_job(self, job_id: str, *, result: Any) -> WorkflowJob:
        """Transition a job from RUNNING to COMPLETED and attach result."""
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise JobNotFoundError(f"Job not found: {job_id}")

            self._validate_transition(current.status, JobStatus.COMPLETED, job_id)
            now = _utc_now()
            updated = WorkflowJob(
                job_id=current.job_id,
                status=JobStatus.COMPLETED,
                created_at=current.created_at,
                started_at=current.started_at or now,
                completed_at=now,
                result=result,
                metadata=current.metadata,
            )
            self._jobs[job_id] = updated
            return updated

    def fail_job(self, job_id: str, *, error: str) -> WorkflowJob:
        """Transition a job to FAILED and record error message."""
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise JobNotFoundError(f"Job not found: {job_id}")

            self._validate_transition(current.status, JobStatus.FAILED, job_id)
            now = _utc_now()
            updated = WorkflowJob(
                job_id=current.job_id,
                status=JobStatus.FAILED,
                created_at=current.created_at,
                started_at=current.started_at,
                completed_at=now,
                error=error,
                metadata=current.metadata,
            )
            self._jobs[job_id] = updated
            return updated

    def cancel_job(self, job_id: str, *, reason: str = "Job cancelled by user") -> WorkflowJob:
        """Transition a job to CANCELLED."""
        with self._lock:
            current = self._jobs.get(job_id)
            if current is None:
                raise JobNotFoundError(f"Job not found: {job_id}")

            self._validate_transition(current.status, JobStatus.CANCELLED, job_id)
            now = _utc_now()
            updated = WorkflowJob(
                job_id=current.job_id,
                status=JobStatus.CANCELLED,
                created_at=current.created_at,
                started_at=current.started_at,
                completed_at=now,
                error=reason,
                metadata=current.metadata,
            )
            self._jobs[job_id] = updated
            return updated

    # ------------------------------------------------------------------ #
    # Cleanup & Maintenance
    # ------------------------------------------------------------------ #

    def cleanup_expired_jobs(self, max_age_seconds: float | None = None) -> int:
        """Purge finished terminal jobs older than max_age_seconds.

        Actively running or queued jobs are NEVER deleted regardless of age.

        Returns:
            The count of purged jobs.
        """
        ttl = max_age_seconds if max_age_seconds is not None else self._default_ttl_seconds
        now = _utc_now()
        purged = 0

        with self._lock:
            expired_ids = []
            for jid, job in self._jobs.items():
                # Never delete active (QUEUED or RUNNING) jobs
                if not job.status.is_terminal:
                    continue

                reference_time = job.completed_at or job.created_at
                age_seconds = (now - reference_time).total_seconds()
                if age_seconds >= ttl:
                    expired_ids.append(jid)

            for jid in expired_ids:
                del self._jobs[jid]
                purged += 1

        if purged > 0:
            logger.info("JobManager: purged %d expired terminal jobs", purged)
        return purged

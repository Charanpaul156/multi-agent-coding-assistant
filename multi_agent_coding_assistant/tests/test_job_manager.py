"""Unit tests for the application-layer JobManager."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from backend.application.job_manager import (
    InvalidJobTransitionError,
    JobManager,
    JobNotFoundError,
)
from backend.domain.job_models import JobStatus


@pytest.fixture
def manager() -> JobManager:
    return JobManager(default_ttl_seconds=3600.0)


# 1. Create job
def test_create_job(manager: JobManager):
    job = manager.create_job(metadata={"prompt": "Build calculator"})
    assert job.job_id.startswith("job_")
    assert job.metadata == {"prompt": "Build calculator"}


# 2. Initial status = QUEUED
def test_initial_status_queued(manager: JobManager):
    job = manager.create_job()
    assert job.status == JobStatus.QUEUED
    assert job.status.is_active is True
    assert job.status.is_terminal is False
    assert job.created_at is not None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.result is None
    assert job.error is None


# 3. Transition to RUNNING
def test_transition_to_running(manager: JobManager):
    job = manager.create_job()
    running = manager.start_job(job.job_id)

    assert running.status == JobStatus.RUNNING
    assert running.started_at is not None
    assert running.started_at >= job.created_at


# 4. Transition to COMPLETED
def test_transition_to_completed(manager: JobManager):
    job = manager.create_job()
    manager.start_job(job.job_id)
    completed = manager.complete_job(job.job_id, result={"code": "def add(): pass"})

    assert completed.status == JobStatus.COMPLETED
    assert completed.status.is_terminal is True
    assert completed.completed_at is not None
    assert completed.result == {"code": "def add(): pass"}


# 5. Transition to FAILED
def test_transition_to_failed(manager: JobManager):
    job = manager.create_job()
    manager.start_job(job.job_id)
    failed = manager.fail_job(job.job_id, error="syntax error in coder agent")

    assert failed.status == JobStatus.FAILED
    assert failed.status.is_terminal is True
    assert failed.completed_at is not None
    assert failed.error == "syntax error in coder agent"


# 6. Invalid transitions rejected
def test_invalid_transitions_rejected(manager: JobManager):
    job = manager.create_job()

    # QUEUED -> COMPLETED directly is not allowed (must go through RUNNING)
    with pytest.raises(InvalidJobTransitionError):
        manager.complete_job(job.job_id, result={})

    # Start the job
    manager.start_job(job.job_id)

    # RUNNING -> RUNNING is not allowed
    with pytest.raises(InvalidJobTransitionError):
        manager.start_job(job.job_id)

    # Finish the job
    manager.complete_job(job.job_id, result="done")

    # Terminal transitions must be rejected
    with pytest.raises(InvalidJobTransitionError):
        manager.start_job(job.job_id)

    with pytest.raises(InvalidJobTransitionError):
        manager.fail_job(job.job_id, error="late fail")

    with pytest.raises(InvalidJobTransitionError):
        manager.cancel_job(job.job_id)


# 7. Result stored correctly
def test_result_stored(manager: JobManager):
    job = manager.create_job()
    manager.start_job(job.job_id)
    expected_result = {"status": "ok", "tests_passed": 30, "score": 98}
    manager.complete_job(job.job_id, result=expected_result)

    retrieved = manager.get_job(job.job_id)
    assert retrieved is not None
    assert retrieved.result == expected_result


# 8. Error stored correctly
def test_error_stored(manager: JobManager):
    job = manager.create_job()
    manager.fail_job(job.job_id, error="timeout exceeded")

    retrieved = manager.get_job(job.job_id)
    assert retrieved is not None
    assert retrieved.status == JobStatus.FAILED
    assert retrieved.error == "timeout exceeded"


# 9. Missing job behavior
def test_missing_job_behavior(manager: JobManager):
    assert manager.get_job("non_existent") is None

    with pytest.raises(JobNotFoundError):
        manager.start_job("non_existent")

    with pytest.raises(JobNotFoundError):
        manager.complete_job("non_existent", result=None)

    with pytest.raises(JobNotFoundError):
        manager.fail_job("non_existent", error="fail")

    with pytest.raises(JobNotFoundError):
        manager.cancel_job("non_existent")


# 10. TTL cleanup
def test_ttl_cleanup(manager: JobManager):
    # Job 1 completed (should expire with max_age = 0)
    j1 = manager.create_job(job_id="j1")
    manager.start_job("j1")
    manager.complete_job("j1", result="done")

    # Job 2 failed (should expire with max_age = 0)
    j2 = manager.create_job(job_id="j2")
    manager.fail_job("j2", error="failed")

    # Purge jobs older than 0.0 seconds
    purged = manager.cleanup_expired_jobs(max_age_seconds=0.0)
    assert purged == 2
    assert manager.get_job("j1") is None
    assert manager.get_job("j2") is None


# 11. Active jobs are retained during TTL cleanup
def test_active_jobs_retained_during_cleanup(manager: JobManager):
    # Job 1 is QUEUED
    j_queued = manager.create_job(job_id="j_queued")

    # Job 2 is RUNNING
    j_running = manager.create_job(job_id="j_running")
    manager.start_job("j_running")

    # Job 3 is COMPLETED
    j_done = manager.create_job(job_id="j_done")
    manager.start_job("j_done")
    manager.complete_job("j_done", result="done")

    # Purge with 0s TTL
    purged = manager.cleanup_expired_jobs(max_age_seconds=0.0)
    assert purged == 1

    # Active jobs MUST be retained
    assert manager.get_job("j_queued") is not None
    assert manager.get_job("j_running") is not None
    assert manager.get_job("j_done") is None


# 12. Timestamps update correctly
def test_timestamps_update_correctly(manager: JobManager):
    job = manager.create_job()
    t_create = job.created_at
    assert t_create.tzinfo is not None

    running = manager.start_job(job.job_id)
    t_start = running.started_at
    assert t_start is not None
    assert t_start >= t_create

    completed = manager.complete_job(job.job_id, result="done")
    t_complete = completed.completed_at
    assert t_complete is not None
    assert t_complete >= t_start

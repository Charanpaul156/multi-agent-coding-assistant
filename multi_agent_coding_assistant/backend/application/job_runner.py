"""Application JobRunner for executing background workflow jobs.

Coordinates the lifecycle of a WorkflowJob:
QUEUED -> RUNNING -> COMPLETED or FAILED.
Binds a stable job_id, passes an EventEmitter to the callable,
stores final results or errors, and signals stream completion on the EventBus.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Mapping

from backend.application.event_emitter import EventEmitter, ScopedEventEmitter
from backend.application.job_manager import JobManager
from backend.domain.job_models import WorkflowJob
from backend.infrastructure.event_bus import EventBus

logger = logging.getLogger(__name__)


class JobRunner:
    """Coordinates lifecycle, execution, and event signaling for workflow jobs."""

    def __init__(
        self,
        job_manager: JobManager,
        event_bus: EventBus | None = None,
        event_emitter: EventEmitter | None = None,
    ) -> None:
        self._job_manager = job_manager
        self._event_bus = event_bus
        self._event_emitter: EventEmitter | None = event_emitter if event_emitter is not None else event_bus

    @property
    def job_manager(self) -> JobManager:
        return self._job_manager

    @property
    def event_bus(self) -> EventBus | None:
        return self._event_bus

    @property
    def event_emitter(self) -> EventEmitter | None:
        return self._event_emitter

    def create_job(
        self,
        *,
        job_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> WorkflowJob:
        """Create a new job in the QUEUED state."""
        return self._job_manager.create_job(job_id=job_id, metadata=metadata)

    def run_job(
        self,
        task_callable: Callable[..., Any],
        *,
        job_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> WorkflowJob:
        """Execute a task callable within a managed WorkflowJob lifecycle.

        Args:
            task_callable: Synchronous callable accepting (job_id, scoped_emitter),
                (job_id), or () that performs the workflow work.
            job_id: Optional existing or desired job ID. If not present in JobManager,
                a new QUEUED job is created.
            metadata: Optional immutable metadata for the job.
            raise_on_error: If True, re-raises any exception after transitioning the
                job to FAILED and emitting failure events. Defaults to False.

        Returns:
            The finished terminal WorkflowJob (status COMPLETED or FAILED).
        """
        # 1. Ensure job exists in JobManager (QUEUED state)
        if job_id:
            existing = self._job_manager.get_job(job_id)
            if existing is None:
                job = self._job_manager.create_job(job_id=job_id, metadata=metadata)
            else:
                job = existing
        else:
            job = self._job_manager.create_job(metadata=metadata)

        actual_job_id = job.job_id

        # 2. Transition to RUNNING
        self._job_manager.start_job(actual_job_id)

        # 3. Create scoped emitter for this job_id
        scoped_emitter: ScopedEventEmitter | None = None
        if self._event_emitter is not None:
            scoped_emitter = ScopedEventEmitter(self._event_emitter, actual_job_id)

        # 4. Execute callable
        try:
            result = self._invoke_callable(task_callable, actual_job_id, scoped_emitter)

            # 5. Transition to COMPLETED and store result
            completed_job = self._job_manager.complete_job(actual_job_id, result=result)

            # 6. Emit completion event
            if scoped_emitter is not None:
                scoped_emitter.emit_event(
                    "job_completed",
                    {
                        "job_id": actual_job_id,
                        "status": "completed",
                    },
                )

            # 7. Signal event-stream completion through EventBus
            if self._event_bus is not None:
                self._event_bus.signal_complete_sync(actual_job_id)

            return completed_job

        except Exception as exc:
            logger.exception("JobRunner execution failed for job %s: %s", actual_job_id, exc)

            # Transition to FAILED and store error
            failed_job = self._job_manager.fail_job(actual_job_id, error=str(exc))

            # Emit failure event
            if scoped_emitter is not None:
                scoped_emitter.emit_event(
                    "job_failed",
                    {
                        "job_id": actual_job_id,
                        "status": "failed",
                        "error": str(exc),
                    },
                )

            # Signal event-stream completion through EventBus
            if self._event_bus is not None:
                self._event_bus.signal_complete_sync(actual_job_id)

            if raise_on_error:
                raise

            return failed_job

    def _invoke_callable(
        self,
        task_callable: Callable[..., Any],
        job_id: str,
        emitter: EventEmitter | None,
    ) -> Any:
        """Call task_callable matching its accepted parameter signature."""
        try:
            sig = inspect.signature(task_callable)
            params = list(sig.parameters.values())

            # Check for varargs (*args)
            if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
                return task_callable(job_id, emitter)

            non_default = [
                p for p in params
                if p.default == inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            ]

            if len(params) == 0 or len(non_default) == 0:
                return task_callable()
            if len(non_default) == 1:
                return task_callable(job_id)
            return task_callable(job_id, emitter)
        except (ValueError, TypeError):
            # Fallback for builtins or special callables
            try:
                return task_callable(job_id, emitter)
            except TypeError:
                try:
                    return task_callable(job_id)
                except TypeError:
                    return task_callable()

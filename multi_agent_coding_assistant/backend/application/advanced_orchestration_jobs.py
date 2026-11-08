"""Background execution entry point for advanced orchestration jobs."""

from __future__ import annotations

import logging

from backend.application.advanced_orchestrator import AdvancedOrchestrator
from backend.application.event_emitter import EventEmitter
from backend.application.job_runner import JobRunner
from backend.domain.orchestration_models import OrchestrationResult, WorkflowMode

logger = logging.getLogger(__name__)

_SAFE_ADVANCED_JOB_ERROR = "Advanced orchestration job failed. See server diagnostics for details."


def run_advanced_orchestration_job(
    job_runner: JobRunner,
    orchestrator: AdvancedOrchestrator,
    job_id: str,
    *,
    mode: WorkflowMode,
    prompt: str,
    code_input: str | None = None,
    files_input: dict[str, str] | None = None,
    repository_path: str | None = None,
    language: str | None = None,
) -> None:
    """Run an advanced orchestration job under the shared job lifecycle.

    A normally returned result completes the job even when ``result.success``
    is false; that field describes orchestration outcome. Exceptions escaping
    the task are recorded by ``JobRunner`` as a failed job.
    """

    def _task(
        scoped_job_id: str, event_emitter: EventEmitter | None
    ) -> OrchestrationResult:
        try:
            return orchestrator.execute(
                prompt,
                mode=mode,
                code_input=code_input,
                files_input=files_input,
                repository_path=repository_path,
                language=language,
                event_emitter=event_emitter,
                job_id=scoped_job_id,
            )
        except Exception as exc:
            logger.error(
                "Advanced orchestration job %s failed (%s)",
                scoped_job_id,
                type(exc).__name__,
            )
            raise RuntimeError(_SAFE_ADVANCED_JOB_ERROR) from None

    job_runner.run_job(_task, job_id=job_id)

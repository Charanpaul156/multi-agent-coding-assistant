"""Application use-cases for repository-aware code modification.

This is the `ModifyRepositoryUseCase`. It coordinates the full pipeline:

    User Request
        -> RAG context (retriever)
        -> Planner
        -> CoderAgent.generate_changes
        -> (augment original content/hash)
        -> ChangeValidator
        -> DiffGenerator
        -> (dry_run? show diffs : ChangeApplier apply)
        -> generate tests -> execute tests -> review
        -> (if problems) DebuggerAgent.correct_changes -> re-validate -> re-apply
        -> repeat (max_iterations enforced)

Design constraints:
    - Framework-agnostic: dataclasses only; no FastAPI/Pydantic imports.
    - Agents never write files; only the ChangeApplier touches the filesystem.
    - No shell execution. No Git. No new framework.
    - Backward compatible: existing use-cases/agents are reused or injected.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

from backend.application.event_emitter import EventEmitter
from backend.domain.job_models import WorkflowEvent
from backend.domain.change_models import (
    ApplicationResult,
    ApprovalStatus,
    ChangeOperation,
    ChangeSet,
    DiffEntry,
    FileChange,
    ValidationReport,
)
from backend.infrastructure.approval_store import ApprovalStore
from backend.infrastructure.change_applier import ChangeApplier
from backend.infrastructure.change_validation import ChangeValidator
from backend.infrastructure.diff_generator import diff_for_change
from backend.infrastructure.llm_client import _sanitize_error
from rag.config import RagConfig

from agents.coder_agent import CoderAgent
from agents.debugger_agent import DebuggerAgent
from agents.planner_agent import ImplementationPlan, PlannerAgent
from backend.application.planning_use_cases import (
    GeneratePlanRequest,
    GeneratePlanResult,
    GeneratePlanUseCase,
)
from backend.application.review_use_cases import (
    ReviewCodeRequest,
    ReviewCodeResult,
    ReviewCodeUseCase,
)
from backend.application.test_generation_use_cases import (
    GenerateTestsRequest,
    GenerateTestsResult,
    GenerateTestsUseCase,
)
from backend.tools.test_executor import (
    ExecuteTestsUseCase,
    TestExecutionRequest,
    TestExecutionResponse,
)

logger = logging.getLogger(__name__)


class ModifyRepositoryStatus(str, Enum):
    """Lifecycle status for a repository-modification run."""

    PROPOSED = "proposed"          # dry-run only; nothing applied
    APPLIED = "applied"            # changes applied and validated
    APPLIED_WITH_WARNINGS = "applied_with_warnings"
    VALIDATION_FAILED = "validation_failed"
    APPLICATION_FAILED = "application_failed"
    TEST_FAILED = "test_failed"
    REVIEW_FAILED = "review_failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    DEBUGGER_FAILED = "debugger_failed"
    REJECTED = "rejected"          # user explicitly rejected proposed changes


@dataclass(frozen=True)
class ModifyRepositoryRequest:
    """Request DTO for repository-aware modification.

    ``repository_path`` is validated against configured allowed roots. The
    user never supplies arbitrary paths; the same RAG security mechanism is
    reused. ``dry_run`` disables any write to disk.
    """

    repository_path: str = ""
    request: str = ""
    dry_run: bool = False
    max_iterations: int = 3
    change_set: Optional[ChangeSet] = None
    repository_root: Optional[str] = None
    approval: ApprovalStatus = ApprovalStatus.PREVIEW
    approval_token: Optional[str] = None
    job_id: Optional[str] = None
    event_emitter: Optional[EventEmitter] = None


@dataclass(frozen=True)
class ChangeIteration:
    """A single pipeline pass through the repository-modification loop."""

    iteration_number: int
    change_set: ChangeSet
    validation: Optional[ValidationReport] = None
    diffs: list[DiffEntry] = field(default_factory=list)
    application: Optional[ApplicationResult] = None
    test_execution: Optional[TestExecutionResponse] = None
    review: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class ModifyRepositoryResult:
    """Result DTO for a repository-modification run."""

    success: bool
    status: ModifyRepositoryStatus
    repository_path: str
    dry_run: bool
    approval_status: ApprovalStatus = ApprovalStatus.PREVIEW
    approval_token: Optional[str] = None
    planning: Optional[ImplementationPlan] = None
    change_set: Optional[ChangeSet] = None
    validation: Optional[ValidationReport] = None
    diffs: list[DiffEntry] = field(default_factory=list)
    application: Optional[ApplicationResult] = None
    iterations: list[ChangeIteration] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def validation_result(self) -> Optional[ValidationReport]:
        return self.validation

    @property
    def diff(self) -> list[DiffEntry]:
        return self.diffs

    @property
    def application_result(self) -> Optional[ApplicationResult]:
        return self.application

    @property
    def applied_files(self) -> list[str]:
        if self.application is not None:
            return list(self.application.applied_files)
        return []


class ModifyRepositoryUseCase:
    """Coordinate repository-aware code modification.

    This class contains NO AI logic. It only orchestrates the injected
    agents and infrastructure components. It never writes files itself.
    """

    def __init__(
        self,
        *,
        config: RagConfig,
        retriever_use_case,
        plan_use_case: GeneratePlanUseCase,
        coder_agent: CoderAgent,
        validator: ChangeValidator,
        applier: ChangeApplier,
        test_generation_use_case: Optional[GenerateTestsUseCase] = None,
        test_execution_use_case: Optional[ExecuteTestsUseCase] = None,
        review_use_case: Optional[ReviewCodeUseCase] = None,
        debugger_agent: Optional[DebuggerAgent] = None,
        approval_store: Optional[ApprovalStore] = None,
        event_emitter: Optional[EventEmitter] = None,
        max_iterations: int = 3,
    ) -> None:
        self._config = config
        self._retriever_use_case = retriever_use_case
        self._plan_use_case = plan_use_case
        self._coder_agent = coder_agent
        self._validator = validator
        self._applier = applier
        self._test_generation_use_case = test_generation_use_case
        self._test_execution_use_case = test_execution_use_case
        self._review_use_case = review_use_case
        self._debugger_agent = debugger_agent
        self._approval_store = approval_store
        self._event_emitter = event_emitter
        self._max_iterations = max(1, int(max_iterations))

    def _emit(
        self,
        emitter: Optional[EventEmitter],
        job_id: Optional[str],
        event_type: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Safely emit an observable progress event if emitter and job_id are present."""
        if emitter is not None and job_id:
            try:
                emitter.emit(
                    WorkflowEvent(
                        event_type=event_type,
                        job_id=job_id,
                        payload=payload or {},
                    )
                )
            except Exception as exc:
                logger.warning("ModifyRepositoryUseCase: failed to emit event %r: %s", event_type, exc)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def approve(
        self,
        token: str,
        repository_path: str = "",
        *,
        job_id: Optional[str] = None,
        event_emitter: Optional[EventEmitter] = None,
    ) -> ModifyRepositoryResult:
        """Approve and apply an existing proposed change ticket."""
        return self.execute(
            ModifyRepositoryRequest(
                repository_path=repository_path,
                approval_token=token,
                approval=ApprovalStatus.APPROVED,
                dry_run=False,
                job_id=job_id,
                event_emitter=event_emitter,
            )
        )

    def reject(
        self,
        token: str,
        repository_path: str = "",
        *,
        job_id: Optional[str] = None,
        event_emitter: Optional[EventEmitter] = None,
    ) -> ModifyRepositoryResult:
        """Reject an existing proposed change ticket."""
        return self.execute(
            ModifyRepositoryRequest(
                repository_path=repository_path,
                approval_token=token,
                approval=ApprovalStatus.REJECTED,
                dry_run=True,
                job_id=job_id,
                event_emitter=event_emitter,
            )
        )

    def execute(self, request: ModifyRepositoryRequest) -> ModifyRepositoryResult:
        """Run the repository-modification workflow."""
        self._validate_request(request)

        repo_path = request.repository_path or request.repository_root or ""
        emitter = request.event_emitter or self._event_emitter
        job_id = request.job_id

        self._emit(
            emitter,
            job_id,
            "modification_started",
            {"repository_path": repo_path, "dry_run": request.dry_run},
        )

        logger.info(
            "ModifyRepositoryUseCase: start repo=%r dry_run=%r approval=%r",
            repo_path,
            request.dry_run,
            request.approval,
        )

        # If an approval_token is supplied, handle the ticket approval/rejection:
        if request.approval_token:
            return self._handle_approval_token(request, repo_path)

        # If a pre-generated change_set is passed directly:
        if request.change_set is not None:
            change_set = self._augment_original(request.change_set, repository_path=repo_path)
            validation = self._validate(change_set)
            self._emit(
                emitter,
                job_id,
                "validation_complete",
                {"valid": validation.valid},
            )
            diffs = self._build_diffs(change_set)
            self._emit(
                emitter,
                job_id,
                "diff_generated",
                {"diff_count": len(diffs)},
            )

            if not validation.valid:
                self._emit(
                    emitter,
                    job_id,
                    "modification_failed",
                    {"reason": "validation_failed"},
                )
                return ModifyRepositoryResult(
                    success=False,
                    status=ModifyRepositoryStatus.VALIDATION_FAILED,
                    approval_status=request.approval,
                    repository_path=repo_path,
                    dry_run=request.dry_run,
                    change_set=change_set,
                    validation=validation,
                    diffs=diffs,
                    error="proposed changes failed validation",
                )

            token = None
            if self._approval_store is not None:
                ticket = self._approval_store.create_ticket(
                    repository_path=repo_path,
                    change_set=change_set,
                    validation=validation,
                    diffs=diffs,
                )
                token = ticket.token

            if request.dry_run:
                self._emit(
                    emitter,
                    job_id,
                    "approval_required",
                    {"approval_status": "preview", "requires_approval": True},
                )
                return ModifyRepositoryResult(
                    success=True,
                    status=ModifyRepositoryStatus.PROPOSED,
                    approval_status=ApprovalStatus.PREVIEW,
                    approval_token=token,
                    repository_path=repo_path,
                    dry_run=True,
                    change_set=change_set,
                    validation=validation,
                    diffs=diffs,
                )

            # Enforce approval == APPROVED before calling ChangeApplier
            if request.approval != ApprovalStatus.APPROVED:
                if request.approval == ApprovalStatus.REJECTED:
                    self._emit(
                        emitter,
                        job_id,
                        "approval_rejected",
                        {"approval_status": "rejected"},
                    )
                else:
                    self._emit(
                        emitter,
                        job_id,
                        "approval_required",
                        {"approval_status": "preview", "requires_approval": True},
                    )
                return ModifyRepositoryResult(
                    success=False,
                    status=(
                        ModifyRepositoryStatus.REJECTED
                        if request.approval == ApprovalStatus.REJECTED
                        else ModifyRepositoryStatus.PROPOSED
                    ),
                    approval_status=request.approval,
                    approval_token=token,
                    repository_path=repo_path,
                    dry_run=request.dry_run,
                    change_set=change_set,
                    validation=validation,
                    diffs=diffs,
                    error=(
                        "ChangeSet was rejected by user"
                        if request.approval == ApprovalStatus.REJECTED
                        else "Explicit approval required before applying repository changes"
                    ),
                )

            stale_error = self._check_stale(change_set, repository_path=repo_path)
            if stale_error:
                self._emit(
                    emitter,
                    job_id,
                    "stale_change_detected",
                    {},
                )
                self._emit(
                    emitter,
                    job_id,
                    "modification_failed",
                    {"reason": "stale_changes"},
                )
                return ModifyRepositoryResult(
                    success=False,
                    status=ModifyRepositoryStatus.APPLICATION_FAILED,
                    approval_status=ApprovalStatus.REJECTED,
                    approval_token=token,
                    repository_path=repo_path,
                    dry_run=request.dry_run,
                    change_set=change_set,
                    validation=validation,
                    diffs=diffs,
                    error=stale_error,
                )

            self._emit(
                emitter,
                job_id,
                "approval_approved",
                {"approval_status": "approved"},
            )
            application = self._apply(change_set)
            status = (
                ModifyRepositoryStatus.APPLIED
                if application.success
                else ModifyRepositoryStatus.APPLICATION_FAILED
            )
            if application.success:
                self._emit(
                    emitter,
                    job_id,
                    "changes_applied",
                    {"file_count": len(application.applied_files)},
                )
            else:
                self._emit(
                    emitter,
                    job_id,
                    "modification_failed",
                    {"reason": "application_failed"},
                )
            if application.success and token and self._approval_store:
                self._approval_store.update_status(token, ApprovalStatus.APPLIED)

            return ModifyRepositoryResult(
                success=application.success,
                status=status,
                approval_status=ApprovalStatus.APPLIED if application.success else ApprovalStatus.APPROVED,
                approval_token=token,
                repository_path=repo_path,
                dry_run=False,
                change_set=change_set,
                validation=validation,
                diffs=diffs,
                application=application,
                error=application.error,
            )

        # --- Stage 1: RAG context --------------------------------------
        context = self._retrieve_context(request)

        # --- Stage 2: Plan ---------------------------------------------
        planning = self._plan(request, context)

        # --- Stage 3: Generate proposed changes -------------------------
        try:
            change_set = self._coder_agent.generate_changes(
                request.request,
                retrieved_context=context,
                plan_summary=self._plan_summary(planning),
            )
        except Exception as exc:
            logger.exception("ModifyRepositoryUseCase: coder failed")
            clean_err = _sanitize_error(str(exc))
            if any(
                term in clean_err.lower()
                for term in ["json", "decode", "parse", "unterminated", "truncate"]
            ):
                err_msg = (
                    "Repository coding could not parse the AI-generated change set. "
                    "The assistant will retry with a stricter structured-output request."
                )
            else:
                err_msg = f"change generation failed: {clean_err}"

            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.VALIDATION_FAILED,
                approval_status=request.approval,
                repository_path=request.repository_path,
                dry_run=request.dry_run,
                planning=planning,
                error=err_msg,
            )

        # Populate original content/hash for modify operations.
        change_set = self._augment_original(change_set, repository_path=request.repository_path)

        # --- Stage 4: Validate -----------------------------------------
        validation = self._validate(change_set)
        self._emit(
            emitter,
            job_id,
            "validation_complete",
            {"valid": validation.valid},
        )
        if not validation.valid:
            self._emit(
                emitter,
                job_id,
                "modification_failed",
                {"reason": "validation_failed"},
            )
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.VALIDATION_FAILED,
                approval_status=request.approval,
                repository_path=request.repository_path,
                dry_run=request.dry_run,
                planning=planning,
                change_set=change_set,
                validation=validation,
                diffs=self._build_diffs(change_set),
                error="proposed changes failed validation",
            )

        diffs = self._build_diffs(change_set)
        self._emit(
            emitter,
            job_id,
            "diff_generated",
            {"diff_count": len(diffs)},
        )

        token = None
        if self._approval_store is not None:
            ticket = self._approval_store.create_ticket(
                repository_path=repo_path,
                change_set=change_set,
                validation=validation,
                diffs=diffs,
            )
            token = ticket.token

        # --- Dry-run: do not write -------------------------------------
        if request.dry_run:
            self._emit(
                emitter,
                job_id,
                "approval_required",
                {"approval_status": "preview", "requires_approval": True},
            )
            return ModifyRepositoryResult(
                success=True,
                status=ModifyRepositoryStatus.PROPOSED,
                approval_status=ApprovalStatus.PREVIEW,
                approval_token=token,
                repository_path=request.repository_path,
                dry_run=True,
                planning=planning,
                change_set=change_set,
                validation=validation,
                diffs=diffs,
            )

        # Enforce approval == APPROVED before calling ChangeApplier
        if request.approval != ApprovalStatus.APPROVED:
            if request.approval == ApprovalStatus.REJECTED:
                self._emit(
                    emitter,
                    job_id,
                    "approval_rejected",
                    {"approval_status": "rejected"},
                )
            else:
                self._emit(
                    emitter,
                    job_id,
                    "approval_required",
                    {"approval_status": "preview", "requires_approval": True},
                )
            return ModifyRepositoryResult(
                success=False,
                status=(
                    ModifyRepositoryStatus.REJECTED
                    if request.approval == ApprovalStatus.REJECTED
                    else ModifyRepositoryStatus.PROPOSED
                ),
                approval_status=request.approval,
                approval_token=token,
                repository_path=request.repository_path,
                dry_run=request.dry_run,
                planning=planning,
                change_set=change_set,
                validation=validation,
                diffs=diffs,
                error=(
                    "ChangeSet was rejected by user"
                    if request.approval == ApprovalStatus.REJECTED
                    else "Explicit approval required before applying repository changes"
                ),
            )

        stale_error = self._check_stale(change_set, repository_path=request.repository_path)
        if stale_error:
            self._emit(
                emitter,
                job_id,
                "stale_change_detected",
                {},
            )
            self._emit(
                emitter,
                job_id,
                "modification_failed",
                {"reason": "stale_changes"},
            )
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.APPLICATION_FAILED,
                approval_status=ApprovalStatus.REJECTED,
                approval_token=token,
                repository_path=request.repository_path,
                dry_run=request.dry_run,
                planning=planning,
                change_set=change_set,
                validation=validation,
                diffs=diffs,
                error=stale_error,
            )

        self._emit(
            emitter,
            job_id,
            "approval_approved",
            {"approval_status": "approved"},
        )

        # --- Stage 5+: Apply + validate + self-correct -----------------
        result = self._apply_and_validate(
            request,
            planning,
            change_set,
            validation,
            diffs,
        )
        if result.success and token and self._approval_store:
            self._approval_store.update_status(token, ApprovalStatus.APPLIED)
        return result

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def _apply_and_validate(
        self,
        request: ModifyRepositoryRequest,
        planning: Optional[ImplementationPlan],
        change_set: ChangeSet,
        validation: ValidationReport,
        diffs: list[DiffEntry],
    ) -> ModifyRepositoryResult:
        """Apply, run tests/review, and self-correct up to max_iterations."""
        emitter = request.event_emitter or self._event_emitter
        job_id = request.job_id
        current_set = change_set
        iterations: list[ChangeIteration] = []
        iteration_number = 1

        while True:
            # Apply the current change set.
            application = self._apply(current_set)
            iteration = ChangeIteration(
                iteration_number=iteration_number,
                change_set=current_set,
                validation=validation,
                diffs=self._build_diffs(current_set),
                application=application,
            )
            iterations.append(iteration)

            if not application.success:
                self._emit(
                    emitter,
                    job_id,
                    "modification_failed",
                    {"reason": "application_failed", "iteration": iteration_number},
                )
                return ModifyRepositoryResult(
                    success=False,
                    status=ModifyRepositoryStatus.APPLICATION_FAILED,
                    repository_path=request.repository_path,
                    dry_run=False,
                    planning=planning,
                    change_set=current_set,
                    validation=validation,
                    diffs=iteration.diffs,
                    application=application,
                    iterations=iterations,
                    error=application.error,
                )

            self._emit(
                emitter,
                job_id,
                "changes_applied",
                {"file_count": len(application.applied_files), "iteration": iteration_number},
            )

            # Validate the applied result via tests + review.
            feedback = self._validate_applied(current_set, application)
            iteration = ChangeIteration(
                iteration_number=iteration_number,
                change_set=current_set,
                validation=validation,
                diffs=iteration.diffs,
                application=application,
                test_execution=feedback.test_execution,
                review=feedback.review,
                error=feedback.error,
            )
            iterations[-1] = iteration

            if feedback.ok:
                return ModifyRepositoryResult(
                    success=True,
                    status=ModifyRepositoryStatus.APPLIED,
                    repository_path=request.repository_path,
                    dry_run=False,
                    planning=planning,
                    change_set=current_set,
                    validation=validation,
                    diffs=iteration.diffs,
                    application=application,
                    iterations=iterations,
                )

            # Need correction.
            if iteration_number >= self._max_iterations:
                self._emit(
                    emitter,
                    job_id,
                    "modification_failed",
                    {"reason": "max_iterations_reached", "iteration": iteration_number},
                )
                return ModifyRepositoryResult(
                    success=False,
                    status=ModifyRepositoryStatus.MAX_ITERATIONS_REACHED,
                    repository_path=request.repository_path,
                    dry_run=False,
                    planning=planning,
                    change_set=current_set,
                    validation=validation,
                    diffs=iteration.diffs,
                    application=application,
                    iterations=iterations,
                    error=feedback.error,
                )

            if self._debugger_agent is None:
                self._emit(
                    emitter,
                    job_id,
                    "modification_failed",
                    {"reason": "debugger_unavailable"},
                )
                return ModifyRepositoryResult(
                    success=False,
                    status=ModifyRepositoryStatus.DEBUGGER_FAILED,
                    repository_path=request.repository_path,
                    dry_run=False,
                    planning=planning,
                    change_set=current_set,
                    validation=validation,
                    diffs=iteration.diffs,
                    application=application,
                    iterations=iterations,
                    error="debugger not available",
                )

            # Debugger produces a corrected change set.
            try:
                corrected = self._debugger_agent.correct_changes(
                    current_set,
                    feedback=feedback.error or "",
                    retrieved_context=self._retrieve_context(request),
                )
                corrected = self._augment_original(corrected, repository_path=request.repository_path)
            except Exception as exc:
                logger.exception("ModifyRepositoryUseCase: debugger failed")
                self._emit(
                    emitter,
                    job_id,
                    "modification_failed",
                    {"reason": "debugger_failed"},
                )
                return ModifyRepositoryResult(
                    success=False,
                    status=ModifyRepositoryStatus.DEBUGGER_FAILED,
                    repository_path=request.repository_path,
                    dry_run=False,
                    planning=planning,
                    change_set=current_set,
                    validation=validation,
                    diffs=iteration.diffs,
                    application=application,
                    iterations=iterations,
                    error=f"debugger correction failed: {exc}",
                )

            # Re-validate the corrected change set.
            new_validation = self._validate(corrected)
            self._emit(
                emitter,
                job_id,
                "validation_complete",
                {"valid": new_validation.valid, "iteration": iteration_number},
            )
            if not new_validation.valid:
                self._emit(
                    emitter,
                    job_id,
                    "modification_failed",
                    {"reason": "validation_failed", "iteration": iteration_number},
                )
                return ModifyRepositoryResult(
                    success=False,
                    status=ModifyRepositoryStatus.VALIDATION_FAILED,
                    repository_path=request.repository_path,
                    dry_run=False,
                    planning=planning,
                    change_set=corrected,
                    validation=new_validation,
                    diffs=self._build_diffs(corrected),
                    iterations=iterations,
                    error="corrected changes failed validation",
                )

            current_set = corrected
            validation = new_validation
            diffs = self._build_diffs(current_set)

            iteration_number += 1

    # ------------------------------------------------------------------ #
    # Approval & Stale Checking
    # ------------------------------------------------------------------ #

    def _handle_approval_token(
        self, request: ModifyRepositoryRequest, repo_path: str
    ) -> ModifyRepositoryResult:
        """Process an approval or rejection for an existing proposed change ticket."""
        emitter = request.event_emitter or self._event_emitter
        job_id = request.job_id

        if self._approval_store is None:
            self._emit(emitter, job_id, "modification_failed", {"reason": "approval_store_not_configured"})
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.APPLICATION_FAILED,
                approval_status=ApprovalStatus.REJECTED,
                approval_token=request.approval_token,
                repository_path=repo_path,
                dry_run=request.dry_run,
                error="Approval store is not configured",
            )

        ticket = self._approval_store.get_ticket(request.approval_token)
        if ticket is None:
            self._emit(emitter, job_id, "modification_failed", {"reason": "invalid_or_expired_token"})
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.APPLICATION_FAILED,
                approval_status=ApprovalStatus.REJECTED,
                approval_token=request.approval_token,
                repository_path=repo_path,
                dry_run=request.dry_run,
                error=f"Invalid or expired approval token: {request.approval_token}",
            )

        # Check repository path consistency
        target_repo = repo_path or ticket.repository_path
        if repo_path and ticket.repository_path:
            try:
                if Path(repo_path).expanduser().resolve() != Path(ticket.repository_path).expanduser().resolve():
                    self._emit(emitter, job_id, "modification_failed", {"reason": "token_repo_mismatch"})
                    return ModifyRepositoryResult(
                        success=False,
                        status=ModifyRepositoryStatus.APPLICATION_FAILED,
                        approval_status=ApprovalStatus.REJECTED,
                        approval_token=request.approval_token,
                        repository_path=repo_path,
                        dry_run=request.dry_run,
                        error="Approval token does not match repository path",
                    )
            except Exception:
                pass

        # Handle rejection
        if request.approval == ApprovalStatus.REJECTED or ticket.status == ApprovalStatus.REJECTED:
            self._approval_store.update_status(ticket.token, ApprovalStatus.REJECTED)
            self._emit(emitter, job_id, "approval_rejected", {"approval_status": "rejected"})
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.REJECTED,
                approval_status=ApprovalStatus.REJECTED,
                approval_token=request.approval_token,
                repository_path=target_repo,
                dry_run=request.dry_run,
                change_set=ticket.change_set,
                validation=ticket.validation,
                diffs=ticket.diffs,
                error="ChangeSet was rejected by user",
            )

        # Prevent re-applying already-applied changes
        if ticket.status == ApprovalStatus.APPLIED:
            self._emit(emitter, job_id, "modification_failed", {"reason": "already_applied"})
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.APPLICATION_FAILED,
                approval_status=ApprovalStatus.APPLIED,
                approval_token=request.approval_token,
                repository_path=target_repo,
                dry_run=request.dry_run,
                change_set=ticket.change_set,
                validation=ticket.validation,
                diffs=ticket.diffs,
                error="ChangeSet has already been applied",
            )

        # Enforce approval == APPROVED before calling ChangeApplier
        if request.approval != ApprovalStatus.APPROVED:
            self._emit(emitter, job_id, "approval_required", {"approval_status": "preview", "requires_approval": True})
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.PROPOSED,
                approval_status=request.approval,
                approval_token=request.approval_token,
                repository_path=target_repo,
                dry_run=request.dry_run,
                change_set=ticket.change_set,
                validation=ticket.validation,
                diffs=ticket.diffs,
                error="Explicit approval required before applying repository changes",
            )

        # Check for stale repository state
        stale_error = self._check_stale(ticket.change_set, repository_path=target_repo)
        if stale_error:
            self._emit(emitter, job_id, "stale_change_detected", {})
            self._emit(emitter, job_id, "modification_failed", {"reason": "stale_changes"})
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.APPLICATION_FAILED,
                approval_status=ApprovalStatus.REJECTED,
                approval_token=request.approval_token,
                repository_path=target_repo,
                dry_run=request.dry_run,
                change_set=ticket.change_set,
                validation=ticket.validation,
                diffs=ticket.diffs,
                error=stale_error,
            )

        # Re-verify validation
        if not ticket.validation.valid:
            self._emit(emitter, job_id, "modification_failed", {"reason": "validation_failed"})
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.VALIDATION_FAILED,
                approval_status=ApprovalStatus.REJECTED,
                approval_token=request.approval_token,
                repository_path=target_repo,
                dry_run=request.dry_run,
                change_set=ticket.change_set,
                validation=ticket.validation,
                diffs=ticket.diffs,
                error="proposed changes failed validation",
            )

        # Apply approved ChangeSet via ChangeApplier
        self._emit(emitter, job_id, "approval_approved", {"approval_status": "approved"})
        application = self._apply(ticket.change_set)
        status = (
            ModifyRepositoryStatus.APPLIED
            if application.success
            else ModifyRepositoryStatus.APPLICATION_FAILED
        )
        if application.success:
            self._emit(emitter, job_id, "changes_applied", {"file_count": len(application.applied_files)})
            self._approval_store.update_status(ticket.token, ApprovalStatus.APPLIED)
        else:
            self._emit(emitter, job_id, "modification_failed", {"reason": "application_failed"})

        return ModifyRepositoryResult(
            success=application.success,
            status=status,
            approval_status=ApprovalStatus.APPLIED if application.success else ApprovalStatus.APPROVED,
            approval_token=ticket.token,
            repository_path=target_repo,
            dry_run=False,
            change_set=ticket.change_set,
            validation=ticket.validation,
            diffs=ticket.diffs,
            application=application,
            error=application.error,
        )

    def _check_stale(
        self, change_set: ChangeSet, repository_path: str = ""
    ) -> Optional[str]:
        """Verify on-disk state has not changed since the preview was generated."""
        for change in change_set.changes:
            if change.operation == ChangeOperation.MODIFY and change.original_hash:
                _, current_hash = self._read_current(
                    change.file_path, repository_path=repository_path
                )
                if current_hash != change.original_hash:
                    return (
                        f"stale ChangeSet: file {change.file_path} changed on disk "
                        f"(expected hash {change.original_hash[:8]}, found {current_hash[:8]})"
                    )
            elif change.operation == ChangeOperation.CREATE:
                content, _ = self._read_current(
                    change.file_path, repository_path=repository_path
                )
                if content:
                    return f"stale ChangeSet: target file {change.file_path} already exists on disk"
        return None

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _validate_request(self, request: ModifyRepositoryRequest) -> None:
        if not isinstance(request, ModifyRepositoryRequest):
            raise TypeError("request must be a ModifyRepositoryRequest")

        repo_path = request.repository_path or request.repository_root
        if not repo_path or not isinstance(repo_path, str) or not repo_path.strip():
            raise ValueError("repository_path must be a non-empty string")

        if request.approval_token:
            if not isinstance(request.approval_token, str) or not request.approval_token.strip():
                raise ValueError("approval_token must be a non-empty string")
        elif request.change_set is None:
            if not isinstance(request.request, str) or not request.request.strip():
                raise ValueError("request must be a non-empty string")

        # Ensure the repository path resolves inside an allowed root.
        if not self._config.allowed_repository_roots:
            raise ValueError(
                "no allowed repository roots configured; modification is disabled"
            )
        try:
            root = Path(repo_path).expanduser().resolve()
        except OSError as exc:
            raise ValueError(f"invalid repository path: {repo_path}") from exc
        if not self._config.is_within_allowed_root(root):
            raise ValueError(
                f"repository path outside allowed roots: {repo_path}"
            )

    def _retrieve_context(self, request: ModifyRepositoryRequest) -> str:
        """Retrieve relevant repository context via the RAG search use-case."""
        if self._retriever_use_case is None:
            return ""
        try:
            from rag.context import format_retrieved_context

            search_result = self._retriever_use_case.execute(
                self._search_request(request)
            )
            chunks = getattr(search_result, "results", [])
            return format_retrieved_context(chunks)
        except Exception as exc:  # pragma: no cover
            logger.warning("ModifyRepositoryUseCase: retrieval failed: %s", exc)
            return ""

    def _search_request(self, request: ModifyRepositoryRequest):
        # Build a search request for the injected RAG use-case (duck-typed).
        from backend.application.rag_use_cases import SearchRepositoryRequest

        repo_path = request.repository_path or request.repository_root or ""
        repo = Path(repo_path).expanduser().resolve().name
        return SearchRepositoryRequest(query=request.request, repository=repo)

    def _plan(
        self, request: ModifyRepositoryRequest, context: str
    ) -> Optional[ImplementationPlan]:
        if self._plan_use_case is None:
            return None
        try:
            result: GeneratePlanResult = self._plan_use_case.execute(
                GeneratePlanRequest(prompt=request.request)
            )
            return result.plan
        except Exception as exc:
            logger.warning("ModifyRepositoryUseCase: planning failed: %s", exc)
            return None

    @staticmethod
    def _plan_summary(plan: Optional[ImplementationPlan]) -> Optional[str]:
        if plan is None:
            return None
        parts = [
            f"Problem: {plan.problem_summary}",
            f"Type: {plan.project_type}",
            "Requirements:",
            "\n".join(f"- {r}" for r in plan.requirements),
        ]
        return "\n".join(parts)

    def _validate(self, change_set: ChangeSet) -> ValidationReport:
        return self._validator.validate(change_set)

    def _build_diffs(self, change_set: ChangeSet) -> list[DiffEntry]:
        entries: list[DiffEntry] = []
        for change in change_set.changes:
            entries.append(
                DiffEntry(
                    file_path=change.file_path,
                    operation=change.operation.value,
                    diff_text=diff_for_change(change),
                )
            )
        return entries

    def _apply(self, change_set: ChangeSet) -> ApplicationResult:
        return self._applier.apply(change_set, dry_run=False)

    def _augment_original(self, change_set: ChangeSet, repository_path: str = "") -> ChangeSet:
        """Populate original_content/original_hash for modify operations."""
        changed: list[FileChange] = []
        for change in change_set.changes:
            if change.operation != ChangeOperation.MODIFY:
                changed.append(change)
                continue
            content, file_hash = self._read_current(change.file_path, repository_path=repository_path)
            changed.append(
                FileChange(
                    file_path=change.file_path,
                    operation=change.operation,
                    new_content=change.new_content,
                    original_content=content if content else change.original_content,
                    original_hash=file_hash if file_hash else change.original_hash,
                    description=change.description,
                )
            )
        return ChangeSet(changes=changed, summary=change_set.summary)

    def _read_current(self, file_path: str, repository_path: str = "") -> tuple[str, str]:
        """Read the current on-disk content and hash for a repo-relative path."""
        try:
            if repository_path:
                root = Path(repository_path).expanduser().resolve()
            elif self._config and self._config.allowed_repository_roots:
                root = Path(self._config.allowed_repository_roots[0]).expanduser().resolve()
            else:
                return "", ""
            target = (root / file_path).resolve()
            if not target.exists() or not target.is_file():
                return "", ""
            raw = target.read_bytes()
            return raw.decode("utf-8", errors="replace"), hashlib.sha256(raw).hexdigest()
        except Exception:
            return "", ""

    def _validate_applied(self, change_set: ChangeSet, application: ApplicationResult):
        """Run tests + review against the applied changes.

        Returns a small feedback object with ``ok``, ``test_execution``,
        ``review`` and ``error``.
        """

        class _Feedback:
            ok = True
            test_execution = None
            review = None
            error = None

        feedback = _Feedback()

        # --- Generate tests -------------------------------------------
        generated_tests: Optional[str] = None
        if self._test_generation_use_case is not None:
            try:
                test_result: GenerateTestsResult = (
                    self._test_generation_use_case.execute(
                        GenerateTestsRequest(
                            generated_code=self._combined_source(change_set)
                        )
                    )
                )
                generated_tests = test_result.report.generated_test_code
            except Exception as exc:
                logger.warning(
                    "ModifyRepositoryUseCase: test generation failed (continuing): %s",
                    exc,
                )
                feedback.ok = False
                feedback.error = f"test generation failed: {exc}"

        test_execution: Optional[TestExecutionResponse] = None
        if generated_tests and self._test_execution_use_case is not None:
            try:
                files_map = {
                    c.file_path: c.new_content
                    for c in change_set.changes
                    if c.file_path.endswith(".py")
                }
                test_execution = self._test_execution_use_case.execute(
                    TestExecutionRequest(
                        generated_code=self._combined_source(change_set),
                        generated_tests=generated_tests,
                        files=files_map if files_map else None,
                    )
                )
                feedback.test_execution = test_execution
                if not test_execution.success:
                    feedback.ok = False
                    feedback.error = (
                        f"tests failed (exit {test_execution.exit_code}): "
                        f"{test_execution.stderr}"
                    )
            except Exception as exc:
                logger.warning(
                    "ModifyRepositoryUseCase: test execution failed (continuing): %s",
                    exc,
                )
                feedback.ok = False
                feedback.error = f"test execution failed: {exc}"

        # --- Review -----------------------------------------------------
        if self._review_use_case is not None:
            try:
                review_result: ReviewCodeResult = self._review_use_case.execute(
                    ReviewCodeRequest(generated_code=self._combined_source(change_set))
                )
                report = review_result.report
                feedback.review = report.final_summary
                if report.overall_score < 60:
                    feedback.ok = False
                    feedback.error = (
                        feedback.error or ""
                    ) + f"review score too low: {report.overall_score}"
            except Exception as exc:
                logger.warning(
                    "ModifyRepositoryUseCase: review failed (continuing): %s", exc
                )
                feedback.review = str(exc)

        return feedback

    def _combined_source(self, change_set: ChangeSet) -> str:
        """Combine changed Python sources into a single string for analysis.

        This is a pragmatic stand-in for test generation/review; individual
        file-level tests are a future enhancement.
        """
        parts: list[str] = []
        for change in change_set.changes:
            if change.file_path.endswith(".py"):
                parts.append(
                    f"# === {change.file_path} ({change.operation.value}) ===\n"
                    + change.new_content
                )
        if not parts:
            parts.append(change_set.summary or "# no python changes")
        return "\n\n".join(parts)

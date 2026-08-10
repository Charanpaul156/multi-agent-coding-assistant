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
from typing import Optional

from backend.domain.change_models import (
    ApplicationResult,
    ChangeOperation,
    ChangeSet,
    DiffEntry,
    FileChange,
    ValidationReport,
)
from backend.infrastructure.change_applier import ChangeApplier
from backend.infrastructure.change_validation import ChangeValidator
from backend.infrastructure.diff_generator import diff_for_change
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


@dataclass(frozen=True)
class ModifyRepositoryRequest:
    """Request DTO for repository-aware modification.

    ``repository_path`` is validated against configured allowed roots. The
    user never supplies arbitrary paths; the same RAG security mechanism is
    reused. ``dry_run`` disables any write to disk.
    """

    repository_path: str
    request: str
    dry_run: bool = False
    max_iterations: int = 3


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
    planning: Optional[ImplementationPlan] = None
    change_set: Optional[ChangeSet] = None
    validation: Optional[ValidationReport] = None
    diffs: list[DiffEntry] = field(default_factory=list)
    application: Optional[ApplicationResult] = None
    iterations: list[ChangeIteration] = field(default_factory=list)
    error: Optional[str] = None


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
        self._max_iterations = max(1, int(max_iterations))

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def execute(self, request: ModifyRepositoryRequest) -> ModifyRepositoryResult:
        """Run the repository-modification workflow."""
        self._validate_request(request)

        logger.info(
            "ModifyRepositoryUseCase: start repo=%r dry_run=%r",
            request.repository_path,
            request.dry_run,
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
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.VALIDATION_FAILED,
                repository_path=request.repository_path,
                dry_run=request.dry_run,
                planning=planning,
                error=f"change generation failed: {exc}",
            )

        # Populate original content/hash for modify operations.
        change_set = self._augment_original(change_set)

        # --- Stage 4: Validate -----------------------------------------
        validation = self._validate(change_set)
        if not validation.valid:
            return ModifyRepositoryResult(
                success=False,
                status=ModifyRepositoryStatus.VALIDATION_FAILED,
                repository_path=request.repository_path,
                dry_run=request.dry_run,
                planning=planning,
                change_set=change_set,
                validation=validation,
                diffs=self._build_diffs(change_set),
                error="proposed changes failed validation",
            )

        diffs = self._build_diffs(change_set)

        # --- Dry-run: do not write -------------------------------------
        if request.dry_run:
            return ModifyRepositoryResult(
                success=True,
                status=ModifyRepositoryStatus.PROPOSED,
                repository_path=request.repository_path,
                dry_run=True,
                planning=planning,
                change_set=change_set,
                validation=validation,
                diffs=diffs,
            )

        # --- Stage 5+: Apply + validate + self-correct -----------------
        return self._apply_and_validate(
            request,
            planning,
            change_set,
            validation,
            diffs,
        )

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
                corrected = self._augment_original(corrected)
            except Exception as exc:
                logger.exception("ModifyRepositoryUseCase: debugger failed")
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
            if not new_validation.valid:
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
    # Helpers
    # ------------------------------------------------------------------ #

    def _validate_request(self, request: ModifyRepositoryRequest) -> None:
        if not isinstance(request, ModifyRepositoryRequest):
            raise TypeError("request must be a ModifyRepositoryRequest")
        if not isinstance(request.repository_path, str) or not request.repository_path.strip():
            raise ValueError("repository_path must be a non-empty string")
        if not isinstance(request.request, str) or not request.request.strip():
            raise ValueError("request must be a non-empty string")

        # Ensure the repository path resolves inside an allowed root.
        if not self._config.allowed_repository_roots:
            raise ValueError(
                "no allowed repository roots configured; modification is disabled"
            )
        try:
            root = Path(request.repository_path).expanduser().resolve()
        except OSError as exc:
            raise ValueError(f"invalid repository path: {request.repository_path}") from exc
        if not self._config.is_within_allowed_root(root):
            raise ValueError(
                f"repository path outside allowed roots: {request.repository_path}"
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

        repo = Path(request.repository_path).expanduser().resolve().name
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

    def _augment_original(self, change_set: ChangeSet) -> ChangeSet:
        """Populate original_content/original_hash for modify operations."""
        changed: list[FileChange] = []
        for change in change_set.changes:
            if change.operation != ChangeOperation.MODIFY:
                changed.append(change)
                continue
            content, file_hash = self._read_current(change.file_path)
            changed.append(
                FileChange(
                    file_path=change.file_path,
                    operation=change.operation,
                    new_content=change.new_content,
                    original_content=content,
                    original_hash=file_hash,
                    description=change.description,
                )
            )
        return ChangeSet(changes=changed, summary=change_set.summary)

    def _read_current(self, file_path: str) -> tuple[str, str]:
        """Read the current on-disk content and hash for a repo-relative path."""
        roots = self._config.allowed_repository_roots
        root = Path(roots[0]).expanduser().resolve()
        target = (root / file_path).resolve()
        raw = target.read_bytes()
        return raw.decode("utf-8", errors="replace"), hashlib.sha256(raw).hexdigest()

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

        # --- Execute tests ---------------------------------------------
        test_execution: Optional[TestExecutionResponse] = None
        if generated_tests and self._test_execution_use_case is not None:
            try:
                test_execution = self._test_execution_use_case.execute(
                    TestExecutionRequest(
                        generated_code=self._combined_source(change_set),
                        generated_tests=generated_tests,
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

"""Application use-cases for the AI Workflow Orchestrator.

The orchestrator coordinates existing agents and tools but contains NO AI
logic of its own. Each responsibility (planning, coding, test generation,
execution, test execution, review, debugging) remains inside its respective
component.

Framework-agnostic: no FastAPI or Pydantic imports. Only dataclasses.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from agents.planner_agent import ImplementationPlan
from agents.code_reviewer_agent import ReviewReport
from agents.debugger_agent import DebugReport
from backend.application.use_cases import (
    GenerateCodeRequest,
    GenerateCodeResult,
    GenerateCodeUseCase,
)
from backend.application.debugging_use_cases import (
    DebugCodeRequest,
    DebugCodeUseCase,
)
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
from backend.tools.python_executor import (
    ExecutionRequest,
    ExecutionResponse,
    ExecuteCodeUseCase,
)
from backend.tools.test_executor import (
    ExecuteTestsUseCase,
    TestExecutionRequest,
    TestExecutionResponse,
)

logger = logging.getLogger(__name__)


class WorkflowStatus(str, Enum):
    """Lifecycle status for a workflow run."""

    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    PLANNING_FAILED = "planning_failed"
    CODING_FAILED = "coding_failed"
    EXECUTION_FAILED = "execution_failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    DEBUGGER_FAILED = "debugger_failed"


@dataclass(frozen=True)
class WorkflowRequest:
    """Request DTO for running the full multi-agent workflow."""

    prompt: str


@dataclass
class WorkflowIteration:
    """A single pipeline pass through the workflow.

    Iteration 1 is the initial generation pipeline
    (Planner -> Coder -> Test Generator -> Executor -> Test Executor ->
    Reviewer). Subsequent iterations reuse the previous debugger's
    ``corrected_code`` and do NOT re-run the Planner or Coder.

    ``debug_report`` is populated after the iteration is evaluated and the
    Debugger runs (if required). It is mutable so the orchestrator can attach
    the diagnostic result to the iteration that produced the failure.
    """

    iteration_number: int
    generated_code: str
    generated_tests: Optional[str] = None
    execution: Optional[ExecutionResponse] = None
    test_execution: Optional[TestExecutionResponse] = None
    review: Optional[ReviewReport] = None
    debug_report: Optional[DebugReport] = None
    test_error: Optional[str] = None
    review_error: Optional[str] = None


@dataclass(frozen=True)
class WorkflowResult:
    """Result DTO capturing the full pipeline state.

    ``iterations`` preserves the full history of every pipeline pass so the
    agentic self-correction loop is fully inspectable. The top-level fields
    (``generated_code``, ``execution``, ``review``, etc.) mirror the FINAL /
    current iteration for convenient API and UI access.

    ``test_error`` surfaces test-generation/execution failure (distinct from
    ``error`` which is used for reviewer warnings / debugger failures).
    """

    success: bool
    workflow_status: WorkflowStatus
    planning: Optional[ImplementationPlan] = None
    generated_code: Optional[str] = None
    generated_tests: Optional[str] = None
    execution: Optional[ExecutionResponse] = None
    test_execution: Optional[TestExecutionResponse] = None
    review: Optional[ReviewReport] = None
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    test_error: Optional[str] = None
    iterations: List[WorkflowIteration] = field(default_factory=list)


class RunWorkflowUseCase:
    """Application use-case: coordinate the full agent workflow.

    Responsibilities (orchestration ONLY):
        Planner -> Coder -> Test Generator -> Application Executor
        -> Test Executor -> Reviewer -> (if needed) Debugger -> repeat

    This class must NEVER:
        - contain AI prompts / LLM logic
        - generate code
        - generate tests
        - review code
        - debug code
        - execute code
        - execute tests

    Self-correction loop:
        After each iteration the orchestrator decides whether debugging is
        required (application failure, test failure, or reviewer critical
        finding). If so, the Debugger produces ``corrected_code`` which is
        validated through a fresh Test Generator -> Executor -> Test Executor
        -> Reviewer pass. The loop terminates when validation succeeds, the
        maximum iteration count is reached, or the Debugger fails.

    Failure strategy:
        - Planner failure            -> STOP, return failure immediately
        - Coder failure              -> STOP, return failure immediately
        - Test Generator failure     -> CONTINUE; keep code, set
          ``generated_tests=None`` / ``test_error``
        - Application execution failure -> CONTINUE to Reviewer
        - Test execution failure     -> CONTINUE to Reviewer
        - Reviewer failure           -> do NOT fail the workflow; return
          success with review=None and a warning
        - Debugger failure           -> STOP correction, return latest state
        - Maximum iterations reached -> STOP, return latest state with an
          explicit ``max_iterations_reached`` status
    """

    # A reviewer score below this threshold is treated as a critical
    # correctness signal that warrants self-correction.
    CRITICAL_SCORE_THRESHOLD = 60

    def __init__(
        self,
        *,
        plan_use_case: GeneratePlanUseCase,
        coder_use_case: GenerateCodeUseCase,
        test_generation_use_case: GenerateTestsUseCase,
        execute_use_case: ExecuteCodeUseCase,
        test_execution_use_case: ExecuteTestsUseCase,
        review_use_case: ReviewCodeUseCase,
        debug_use_case: Optional[DebugCodeUseCase] = None,
        max_iterations: int = 3,
    ) -> None:
        self._plan_use_case = plan_use_case
        self._coder_use_case = coder_use_case
        self._test_generation_use_case = test_generation_use_case
        self._execute_use_case = execute_use_case
        self._test_execution_use_case = test_execution_use_case
        self._review_use_case = review_use_case
        self._debug_use_case = debug_use_case
        self._max_iterations = max(1, int(max_iterations))

    def execute(self, request: WorkflowRequest) -> WorkflowResult:
        """Run the full workflow and return a (possibly partial) result."""

        if not isinstance(request, WorkflowRequest):
            raise TypeError("request must be a WorkflowRequest")
        if not isinstance(request.prompt, str):
            raise TypeError("prompt must be a string")
        if not request.prompt.strip():
            raise ValueError("prompt must be non-empty")

        logger.info("Workflow Started (prompt=%r)", request.prompt)
        start = time.perf_counter()

        # --- Stage 1: Planner ------------------------------------------------
        try:
            plan_result: GeneratePlanResult = self._plan_use_case.execute(
                GeneratePlanRequest(prompt=request.prompt)
            )
            planning = plan_result.plan
            logger.info("Planner Finished")
        except Exception as exc:
            logger.exception("Planner FAILED")
            return self._fail(
                "planning",
                WorkflowStatus.PLANNING_FAILED,
                request.prompt,
                start,
                error=str(exc),
            )

        # --- Stage 2: Coder (initial generation only) -----------------------
        try:
            code_result: GenerateCodeResult = self._coder_use_case.execute(
                GenerateCodeRequest(prompt=request.prompt)
            )
            current_code = code_result.generated_code
            logger.info("Coder Finished")
        except Exception as exc:
            logger.exception("Coder FAILED")
            return self._fail(
                "coding",
                WorkflowStatus.CODING_FAILED,
                request.prompt,
                start,
                planning=planning,
                error=str(exc),
            )

        # --- Self-correction loop -------------------------------------------
        iterations: List[WorkflowIteration] = []
        iteration_number = 1
        final_status: Optional[WorkflowStatus] = None
        final_error: Optional[str] = None

        while True:
            iteration = self._run_iteration(
                generated_code=current_code,
                iteration_number=iteration_number,
            )
            iterations.append(iteration)

            feedback = self._needs_debugging(iteration)

            # No critical failure -> workflow is complete.
            if feedback is None:
                has_warnings = (
                    iteration.test_error is not None
                    or iteration.review_error is not None
                    or iteration.review is None
                )
                final_status = (
                    WorkflowStatus.COMPLETED_WITH_WARNINGS
                    if has_warnings
                    else WorkflowStatus.COMPLETED
                )
                final_error = iteration.review_error
                break

            # Critical failure but no debugger wired and/or iteration budget
            # exhausted.
            if iteration_number >= self._max_iterations:
                logger.warning(
                    "Workflow reached maximum iterations (%d)", self._max_iterations
                )
                final_status = WorkflowStatus.MAX_ITERATIONS_REACHED
                break

            if self._debug_use_case is None:
                logger.warning("Debugger not available; stopping correction loop")
                final_status = WorkflowStatus.DEBUGGER_FAILED
                final_error = "Debugger not available"
                break

            # Run the Debugger to produce a corrected version.
            try:
                debug_result = self._debug_use_case.execute(
                    DebugCodeRequest(
                        generated_code=iteration.generated_code,
                        generated_tests=iteration.generated_tests,
                        execution_stdout=(
                            iteration.execution.stdout
                            if iteration.execution
                            else None
                        ),
                        execution_stderr=(
                            iteration.execution.stderr
                            if iteration.execution
                            else None
                        ),
                        execution_exit_code=(
                            iteration.execution.exit_code
                            if iteration.execution
                            else None
                        ),
                        test_stdout=(
                            iteration.test_execution.stdout
                            if iteration.test_execution
                            else None
                        ),
                        test_stderr=(
                            iteration.test_execution.stderr
                            if iteration.test_execution
                            else None
                        ),
                        test_exit_code=(
                            iteration.test_execution.exit_code
                            if iteration.test_execution
                            else None
                        ),
                        reviewer_feedback=feedback,
                    )
                )
                iteration.debug_report = debug_result.report
                current_code = debug_result.report.corrected_code
                logger.info(
                    "Debugger produced correction for iteration %d",
                    iteration_number,
                )
            except Exception as exc:
                logger.exception("Debugger FAILED")
                final_status = WorkflowStatus.DEBUGGER_FAILED
                final_error = str(exc)
                break

            iteration_number += 1

        last = iterations[-1]
        elapsed_ms = (time.perf_counter() - start) * 1000

        success = final_status in (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.COMPLETED_WITH_WARNINGS,
        )

        return WorkflowResult(
            success=success,
            workflow_status=final_status or WorkflowStatus.COMPLETED,
            planning=planning,
            generated_code=last.generated_code,
            generated_tests=last.generated_tests,
            execution=last.execution,
            test_execution=last.test_execution,
            review=last.review,
            execution_time_ms=elapsed_ms,
            error=final_error,
            test_error=last.test_error,
            iterations=iterations,
        )

    def _run_iteration(
        self,
        *,
        generated_code: str,
        iteration_number: int,
    ) -> WorkflowIteration:
        """Run one full pipeline pass and return a WorkflowIteration.

        The Planner and Coder are NOT re-run here; initial generation and
        later corrections are supplied via ``generated_code``.
        """
        logger.info("Workflow iteration %d: starting", iteration_number)

        # --- Test Generator -------------------------------------------------
        generated_tests: Optional[str] = None
        test_error: Optional[str] = None
        try:
            test_result: GenerateTestsResult = self._test_generation_use_case.execute(
                GenerateTestsRequest(generated_code=generated_code)
            )
            generated_tests = test_result.report.generated_test_code
            logger.info("Workflow iteration %d: Test Generator finished", iteration_number)
        except Exception as exc:
            logger.warning(
                "Workflow iteration %d: Test Generator FAILED (continuing): %s",
                iteration_number,
                exc,
            )
            generated_tests = None
            test_error = str(exc)

        # --- Application Execution ------------------------------------------
        execution: Optional[ExecutionResponse] = None
        try:
            execution = self._execute_use_case.execute(
                ExecutionRequest(generated_code=generated_code)
            )
            logger.info("Workflow iteration %d: Application Execution finished", iteration_number)
        except Exception as exc:
            logger.warning(
                "Workflow iteration %d: Application Execution FAILED (continuing): %s",
                iteration_number,
                exc,
            )
            execution = None

        # --- Test Execution -------------------------------------------------
        test_execution: Optional[TestExecutionResponse] = None
        if generated_tests is not None:
            try:
                test_execution = self._test_execution_use_case.execute(
                    TestExecutionRequest(
                        generated_code=generated_code,
                        generated_tests=generated_tests,
                    )
                )
                logger.info("Workflow iteration %d: Test Execution finished", iteration_number)
            except Exception as exc:
                logger.warning(
                    "Workflow iteration %d: Test Execution FAILED (continuing): %s",
                    iteration_number,
                    exc,
                )
                test_execution = None
        else:
            logger.info(
                "Workflow iteration %d: Test Execution SKIPPED (no tests)",
                iteration_number,
            )

        # --- Reviewer -------------------------------------------------------
        review: Optional[ReviewReport] = None
        review_error: Optional[str] = None
        try:
            review_result: ReviewCodeResult = self._review_use_case.execute(
                ReviewCodeRequest(
                    generated_code=generated_code,
                    stdout=execution.stdout if execution else None,
                    stderr=execution.stderr if execution else None,
                    exit_code=execution.exit_code if execution else None,
                    generated_tests=generated_tests,
                    test_stdout=(
                        test_execution.stdout if test_execution else None
                    ),
                    test_stderr=(
                        test_execution.stderr if test_execution else None
                    ),
                    test_exit_code=(
                        test_execution.exit_code if test_execution else None
                    ),
                )
            )
            review = review_result.report
            logger.info("Workflow iteration %d: Reviewer finished", iteration_number)
        except Exception as exc:
            logger.warning(
                "Workflow iteration %d: Reviewer FAILED (continuing): %s",
                iteration_number,
                exc,
            )
            review = None
            review_error = str(exc)

        return WorkflowIteration(
            iteration_number=iteration_number,
            generated_code=generated_code,
            generated_tests=generated_tests,
            execution=execution,
            test_execution=test_execution,
            review=review,
            debug_report=None,
            test_error=test_error,
            review_error=review_error,
        )

    def _needs_debugging(self, iteration: WorkflowIteration) -> Optional[str]:
        """Determine whether self-correction is required.

        Returns a reviewer-feedback string if debugging should run, otherwise
        ``None``.

        CRITICAL triggers:
            - application execution failure (non-zero exit code)
            - test execution failure (non-zero exit code / failed tests)
            - reviewer logic/security issues
            - reviewer overall_score below ``CRITICAL_SCORE_THRESHOLD``

        NON-CRITICAL (do NOT trigger debugging):
            - PEP8 / style, minor performance, maintainability, optional
              refactoring suggestions.
        """
        feedback_parts: List[str] = []

        execution = iteration.execution
        if execution is not None and (
            not execution.success or execution.exit_code != 0
        ):
            feedback_parts.append(
                "Application execution failed "
                f"(exit code {execution.exit_code}):\n{execution.stderr}"
            )

        test_execution = iteration.test_execution
        if test_execution is not None and (
            not test_execution.success
            or test_execution.exit_code != 0
            or (test_execution.failed or 0) > 0
        ):
            feedback_parts.append(
                "Test execution failed "
                f"(exit code {test_execution.exit_code}, "
                f"failed={test_execution.failed}):\n{test_execution.stdout}"
            )

        review = iteration.review
        if review is not None:
            logic_issues = getattr(review, "logic_issues", None) or []
            security_concerns = getattr(review, "security_concerns", None) or []
            if logic_issues:
                feedback_parts.append(
                    "Critical logic issues:\n" + "\n".join(logic_issues)
                )
            if security_concerns:
                feedback_parts.append(
                    "Security concerns:\n" + "\n".join(security_concerns)
                )
            overall_score = getattr(review, "overall_score", None)
            if isinstance(overall_score, int) and not isinstance(overall_score, bool):
                if overall_score < self.CRITICAL_SCORE_THRESHOLD:
                    feedback_parts.append(
                        f"Overall review score too low: {overall_score}"
                    )

        return "\n".join(feedback_parts) if feedback_parts else None

    def _fail(
        self,
        stage: str,
        status: WorkflowStatus,
        prompt: str,
        start: float,
        *,
        planning: Optional[ImplementationPlan] = None,
        error: Optional[str] = None,
    ) -> WorkflowResult:
        """Build a partial failure result for a critical stage."""

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.warning("Workflow FAILED at %s (%.2f ms)", stage, elapsed_ms)
        return WorkflowResult(
            success=False,
            workflow_status=status,
            planning=planning,
            generated_code=None,
            generated_tests=None,
            execution=None,
            test_execution=None,
            review=None,
            execution_time_ms=elapsed_ms,
            error=error,
            test_error=None,
            iterations=[],
        )

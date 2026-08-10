"""API routes.

This module defines HTTP endpoints.
Business logic must remain in the application layer.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import (
    get_debug_code_use_case,
    get_execute_code_use_case,
    get_generate_code_use_case,
    get_generate_plan_use_case,
    get_generate_tests_use_case,
    get_index_repository_use_case,
    get_modify_repository_use_case,
    get_rag_status_use_case,
    get_review_code_use_case,
    get_run_workflow_use_case,
    get_search_repository_use_case,
)
from backend.application.use_cases import GenerateCodeRequest, GenerateCodeUseCase
from backend.application.debugging_use_cases import (
    DebugCodeRequest,
    DebugCodeUseCase,
)
from backend.application.planning_use_cases import GeneratePlanRequest, GeneratePlanUseCase
from backend.application.review_use_cases import ReviewCodeRequest, ReviewCodeUseCase
from backend.application.test_generation_use_cases import (
    GenerateTestsRequest,
    GenerateTestsUseCase,
)
from backend.application.workflow_use_cases import (
    RunWorkflowUseCase,
    WorkflowRequest,
)

from backend.tools.python_executor import ExecutionRequest, ExecuteCodeUseCase
from backend.application.rag_use_cases import (
    IndexRepositoryRequest,
    IndexRepositoryUseCase,
    SearchRepositoryRequest,
    SearchRepositoryUseCase,
    GetRagStatusUseCase,
)
from backend.application.modify_repository_use_cases import (
    ModifyRepositoryRequest,
    ModifyRepositoryUseCase,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class GenerateCodePayload(BaseModel):
    prompt: str = Field(..., min_length=1)


class GenerateCodeResponse(BaseModel):
    success: bool
    generated_code: str


class ExecuteCodePayload(BaseModel):
    generated_code: str


class ExecuteCodeApiResponse(BaseModel):
    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    exit_code: int


@router.post(
    "/execute-code",
    response_model=ExecuteCodeApiResponse,
    tags=["ai"],
)
def execute_code(
    payload: ExecuteCodePayload,
    use_case: ExecuteCodeUseCase = Depends(get_execute_code_use_case),
) -> ExecuteCodeApiResponse:
    """Execute AI-generated Python code."""

    code = (payload.generated_code or "").strip()
    if not code:
        logger.warning("POST /execute-code: empty generated_code")
        raise HTTPException(status_code=400, detail="generated_code must not be empty")

    logger.info("POST /execute-code: request received")
    try:
        logger.info("POST /execute-code: use-case started")
        result = use_case.execute(ExecutionRequest(generated_code=code))
        logger.info("POST /execute-code: use-case finished")
        return ExecuteCodeApiResponse(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            execution_time_ms=result.execution_time_ms,
            exit_code=result.exit_code,
        )
    except ValueError as exc:
        logger.warning("POST /execute-code: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except TimeoutError:
        logger.warning("POST /execute-code: execution timeout")
        raise HTTPException(status_code=408, detail="Execution timed out") from None
    except Exception as exc:
        logger.exception("POST /execute-code: error")
        raise HTTPException(status_code=500, detail="Unexpected execution failure") from exc


@router.get("/health", tags=["system"])
def health_check() -> dict:
    """Health endpoint."""

    return {"status": "ok"}


class GeneratePlanPayload(BaseModel):
    prompt: str = Field(..., min_length=1)


class ImplementationPlanModel(BaseModel):
    problem_summary: str
    project_type: str
    requirements: list[str]
    modules: list[str]
    functions: list[str]
    classes: list[str]
    external_libraries: list[str]
    database_needed: bool
    api_needed: list[str]
    algorithm: str
    edge_cases: list[str]
    estimated_complexity: str
    future_improvements: list[str]


class GeneratePlanApiResponse(BaseModel):
    success: bool
    plan: ImplementationPlanModel


@router.post(
    "/generate-plan",
    response_model=GeneratePlanApiResponse,
    tags=["ai"],
)
def generate_plan(
    payload: GeneratePlanPayload,
    use_case: GeneratePlanUseCase = Depends(get_generate_plan_use_case),
) -> GeneratePlanApiResponse:
    """Generate a structured implementation plan from a natural language prompt."""

    prompt = (payload.prompt or "").strip()
    if not prompt:
        logger.warning("POST /generate-plan: empty prompt")
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    logger.info("POST /generate-plan: request received")
    try:
        logger.info("POST /generate-plan: use-case started")
        result = use_case.execute(GeneratePlanRequest(prompt=prompt))
        logger.info("POST /generate-plan: use-case finished")

        plan = result.plan
        return GeneratePlanApiResponse(
            success=True,
            plan=ImplementationPlanModel(
                problem_summary=plan.problem_summary,
                project_type=plan.project_type,
                requirements=plan.requirements,
                modules=plan.modules,
                functions=plan.functions,
                classes=plan.classes,
                external_libraries=plan.external_libraries,
                database_needed=plan.database_needed,
                api_needed=plan.api_needed,
                algorithm=plan.algorithm,
                edge_cases=plan.edge_cases,
                estimated_complexity=plan.estimated_complexity,
                future_improvements=plan.future_improvements,
            ),
        )
    except ValueError as exc:
        logger.warning("POST /generate-plan: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /generate-plan: error")
        raise HTTPException(status_code=500, detail="LLM planning failed") from exc


@router.post(
    "/generate-code",
    response_model=GenerateCodeResponse,
    tags=["ai"],
)
def generate_code(
    payload: GenerateCodePayload,
    use_case: GenerateCodeUseCase = Depends(get_generate_code_use_case),
) -> GenerateCodeResponse:
    """Generate Python code from a natural language prompt."""

    prompt = (payload.prompt or "").strip()
    if not prompt:
        logger.warning("POST /generate-code: empty prompt")
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    logger.info("POST /generate-code: request received")
    try:
        logger.info("POST /generate-code: use-case started")
        result = use_case.execute(GenerateCodeRequest(prompt=prompt))
        logger.info("POST /generate-code: use-case finished")
        return GenerateCodeResponse(success=True, generated_code=result.generated_code)
    except ValueError as exc:
        logger.warning("POST /generate-code: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /generate-code: error")
        raise HTTPException(status_code=500, detail="LLM generation failed") from exc


class ReviewCodePayload(BaseModel):
    generated_code: str = Field(..., min_length=1)
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None


class ReviewReportModel(BaseModel):
    overall_score: int
    strengths: list[str]
    weaknesses: list[str]
    pep8_issues: list[str]
    performance_suggestions: list[str]
    security_concerns: list[str]
    logic_issues: list[str]
    maintainability: list[str]
    error_handling: list[str]
    recommendations: list[str]
    final_summary: str


class ReviewCodeApiResponse(BaseModel):
    success: bool
    review: ReviewReportModel


@router.post(
    "/review-code",
    response_model=ReviewCodeApiResponse,
    tags=["ai"],
)
def review_code(
    payload: ReviewCodePayload,
    use_case: ReviewCodeUseCase = Depends(get_review_code_use_case),
) -> ReviewCodeApiResponse:
    """Review AI-generated Python code like a Senior Engineer PR review.

    The ReviewerAgent only analyzes code and optional execution results. It
    never executes, generates, or modifies code.
    """

    code = (payload.generated_code or "").strip()
    if not code:
        logger.warning("POST /review-code: empty generated_code")
        raise HTTPException(status_code=400, detail="generated_code must not be empty")

    logger.info("POST /review-code: request received")
    try:
        logger.info("POST /review-code: use-case started")
        result = use_case.execute(
            ReviewCodeRequest(
                generated_code=code,
                stdout=payload.stdout,
                stderr=payload.stderr,
                exit_code=payload.exit_code,
            )
        )
        logger.info("POST /review-code: use-case finished")

        report = result.report
        return ReviewCodeApiResponse(
            success=True,
            review=ReviewReportModel(
                overall_score=report.overall_score,
                strengths=report.strengths,
                weaknesses=report.weaknesses,
                pep8_issues=report.pep8_issues,
                performance_suggestions=report.performance_suggestions,
                security_concerns=report.security_concerns,
                logic_issues=report.logic_issues,
                maintainability=report.maintainability,
                error_handling=report.error_handling,
                recommendations=report.recommendations,
                final_summary=report.final_summary,
            ),
        )
    except ValueError as exc:
        logger.warning("POST /review-code: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /review-code: error")
        raise HTTPException(status_code=500, detail="LLM review failed") from exc


class DebugCodePayload(BaseModel):
    generated_code: str = Field(..., min_length=1)
    generated_tests: str | None = None
    execution_stdout: str | None = None
    execution_stderr: str | None = None
    execution_exit_code: int | None = None
    test_stdout: str | None = None
    test_stderr: str | None = None
    test_exit_code: int | None = None
    reviewer_feedback: str | None = None


class DebugReportModel(BaseModel):
    issue_detected: bool
    error_type: str
    root_cause: str
    affected_component: str
    explanation: str
    suggested_changes: list[str]
    corrected_code: str
    confidence: float
    final_summary: str


class DebugCodeApiResponse(BaseModel):
    success: bool
    debug_report: DebugReportModel


@router.post(
    "/debug-code",
    response_model=DebugCodeApiResponse,
    tags=["ai"],
)
def debug_code(
    payload: DebugCodePayload,
    use_case: DebugCodeUseCase = Depends(get_debug_code_use_case),
) -> DebugCodeApiResponse:
    """Debug failed AI-generated Python code.

    The DebuggerAgent analyzes the code, tests, execution results and
    reviewer feedback. It never executes code, never writes files, and never
    modifies the repository.
    """

    code = (payload.generated_code or "").strip()
    if not code:
        logger.warning("POST /debug-code: empty generated_code")
        raise HTTPException(status_code=400, detail="generated_code must not be empty")

    logger.info("POST /debug-code: request received")
    try:
        logger.info("POST /debug-code: use-case started")
        result = use_case.execute(
            DebugCodeRequest(
                generated_code=code,
                generated_tests=payload.generated_tests,
                execution_stdout=payload.execution_stdout,
                execution_stderr=payload.execution_stderr,
                execution_exit_code=payload.execution_exit_code,
                test_stdout=payload.test_stdout,
                test_stderr=payload.test_stderr,
                test_exit_code=payload.test_exit_code,
                reviewer_feedback=payload.reviewer_feedback,
            )
        )
        logger.info("POST /debug-code: use-case finished")

        report = result.report
        return DebugCodeApiResponse(
            success=True,
            debug_report=DebugReportModel(
                issue_detected=report.issue_detected,
                error_type=report.error_type,
                root_cause=report.root_cause,
                affected_component=report.affected_component,
                explanation=report.explanation,
                suggested_changes=report.suggested_changes,
                corrected_code=report.corrected_code,
                confidence=report.confidence,
                final_summary=report.final_summary,
            ),
        )
    except ValueError as exc:
        logger.warning("POST /debug-code: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /debug-code: error")
        raise HTTPException(status_code=500, detail="LLM debugging failed") from exc


class GenerateTestsPayload(BaseModel):
    generated_code: str = Field(..., min_length=1)


class TestCaseModel(BaseModel):
    name: str
    description: str
    input_example: str
    expected_behavior: str


class GenerateTestsReportModel(BaseModel):
    test_overview: str
    test_framework: str
    test_cases: list[TestCaseModel]
    edge_cases: list[TestCaseModel]
    negative_cases: list[TestCaseModel]
    coverage_suggestions: list[str]
    generated_test_code: str
    final_summary: str


class GenerateTestsApiResponse(BaseModel):
    success: bool
    report: GenerateTestsReportModel


@router.post(
    "/generate-tests",
    response_model=GenerateTestsApiResponse,
    tags=["ai"],
)
def generate_tests(
    payload: GenerateTestsPayload,
    use_case: GenerateTestsUseCase = Depends(get_generate_tests_use_case),
) -> GenerateTestsApiResponse:
    """Generate a pytest-oriented test suite for AI-generated Python code.

    The TestGeneratorAgent analyzes the code and generates tests only. It
    never executes tests, never executes application code, and never modifies
    the original source code.
    """

    code = (payload.generated_code or "").strip()
    if not code:
        logger.warning("POST /generate-tests: empty generated_code")
        raise HTTPException(status_code=400, detail="generated_code must not be empty")

    logger.info("POST /generate-tests: request received")
    try:
        logger.info("POST /generate-tests: use-case started")
        result = use_case.execute(GenerateTestsRequest(generated_code=code))
        logger.info("POST /generate-tests: use-case finished")

        report = result.report
        return GenerateTestsApiResponse(
            success=True,
            report=GenerateTestsReportModel(
                test_overview=report.test_overview,
                test_framework=report.test_framework,
                test_cases=[
                    TestCaseModel(
                        name=case.name,
                        description=case.description,
                        input_example=case.input_example,
                        expected_behavior=case.expected_behavior,
                    )
                    for case in report.test_cases
                ],
                edge_cases=[
                    TestCaseModel(
                        name=case.name,
                        description=case.description,
                        input_example=case.input_example,
                        expected_behavior=case.expected_behavior,
                    )
                    for case in report.edge_cases
                ],
                negative_cases=[
                    TestCaseModel(
                        name=case.name,
                        description=case.description,
                        input_example=case.input_example,
                        expected_behavior=case.expected_behavior,
                    )
                    for case in report.negative_cases
                ],
                coverage_suggestions=report.coverage_suggestions,
                generated_test_code=report.generated_test_code,
                final_summary=report.final_summary,
            ),
        )
    except ValueError as exc:
        logger.warning("POST /generate-tests: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /generate-tests: error")
        raise HTTPException(status_code=500, detail="LLM test generation failed") from exc


class RunWorkflowPayload(BaseModel):
    prompt: str = Field(..., min_length=1)


class WorkflowExecutionModel(BaseModel):
    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    exit_code: int


class WorkflowTestExecutionModel(BaseModel):
    success: bool
    stdout: str
    stderr: str
    execution_time_ms: float
    exit_code: int
    passed: int | None = None
    failed: int | None = None


class WorkflowDebugReportModel(BaseModel):
    issue_detected: bool
    error_type: str
    root_cause: str
    affected_component: str
    explanation: str
    suggested_changes: list[str]
    corrected_code: str
    confidence: float
    final_summary: str


class WorkflowIterationModel(BaseModel):
    iteration_number: int
    generated_code: str
    generated_tests: str | None = None
    execution: WorkflowExecutionModel | None = None
    test_execution: WorkflowTestExecutionModel | None = None
    review: ReviewReportModel | None = None
    debug_report: WorkflowDebugReportModel | None = None
    test_error: str | None = None
    review_error: str | None = None


class WorkflowResponseModel(BaseModel):
    planning: ImplementationPlanModel | None = None
    generated_code: str | None = None
    generated_tests: str | None = None
    execution: WorkflowExecutionModel | None = None
    test_execution: WorkflowTestExecutionModel | None = None
    review: ReviewReportModel | None = None
    workflow_status: str
    execution_time_ms: float
    error: str | None = None
    test_error: str | None = None
    iterations: list[WorkflowIterationModel] = []


class RunWorkflowApiResponse(BaseModel):
    success: bool
    workflow: WorkflowResponseModel


@router.post(
    "/run-workflow",
    response_model=RunWorkflowApiResponse,
    tags=["ai"],
)
def run_workflow(
    payload: RunWorkflowPayload,
    use_case: RunWorkflowUseCase = Depends(get_run_workflow_use_case),
) -> RunWorkflowApiResponse:
    """Run the full multi-agent workflow:

    Planner -> Coder -> Test Generator -> Application Executor
    -> Test Executor -> Reviewer

    The orchestrator contains no AI logic; it only coordinates existing
    agents and tools. Partial results are returned where appropriate.
    """

    prompt = (payload.prompt or "").strip()
    if not prompt:
        logger.warning("POST /run-workflow: empty prompt")
        raise HTTPException(status_code=400, detail="prompt must not be empty")

    logger.info("POST /run-workflow: request received")
    try:
        logger.info("POST /run-workflow: use-case started")
        result = use_case.execute(WorkflowRequest(prompt=prompt))
        logger.info("POST /run-workflow: use-case finished")

        planning_model = None
        if result.planning is not None:
            plan = result.planning
            planning_model = ImplementationPlanModel(
                problem_summary=plan.problem_summary,
                project_type=plan.project_type,
                requirements=plan.requirements,
                modules=plan.modules,
                functions=plan.functions,
                classes=plan.classes,
                external_libraries=plan.external_libraries,
                database_needed=plan.database_needed,
                api_needed=plan.api_needed,
                algorithm=plan.algorithm,
                edge_cases=plan.edge_cases,
                estimated_complexity=plan.estimated_complexity,
                future_improvements=plan.future_improvements,
            )

        execution_model = None
        if result.execution is not None:
            execution_model = WorkflowExecutionModel(
                success=result.execution.success,
                stdout=result.execution.stdout,
                stderr=result.execution.stderr,
                execution_time_ms=result.execution.execution_time_ms,
                exit_code=result.execution.exit_code,
            )

        test_execution_model = None
        if result.test_execution is not None:
            test_execution_model = WorkflowTestExecutionModel(
                success=result.test_execution.success,
                stdout=result.test_execution.stdout,
                stderr=result.test_execution.stderr,
                execution_time_ms=result.test_execution.execution_time_ms,
                exit_code=result.test_execution.exit_code,
                passed=result.test_execution.passed,
                failed=result.test_execution.failed,
            )

        review_model = None
        if result.review is not None:
            report = result.review
            review_model = ReviewReportModel(
                overall_score=report.overall_score,
                strengths=report.strengths,
                weaknesses=report.weaknesses,
                pep8_issues=report.pep8_issues,
                performance_suggestions=report.performance_suggestions,
                security_concerns=report.security_concerns,
                logic_issues=report.logic_issues,
                maintainability=report.maintainability,
                error_handling=report.error_handling,
                recommendations=report.recommendations,
                final_summary=report.final_summary,
            )

        iteration_models = []
        for it in (result.iterations or []):
            it_exec = None
            if it.execution is not None:
                it_exec = WorkflowExecutionModel(
                    success=it.execution.success,
                    stdout=it.execution.stdout,
                    stderr=it.execution.stderr,
                    execution_time_ms=it.execution.execution_time_ms,
                    exit_code=it.execution.exit_code,
                )
            it_test_exec = None
            if it.test_execution is not None:
                it_test_exec = WorkflowTestExecutionModel(
                    success=it.test_execution.success,
                    stdout=it.test_execution.stdout,
                    stderr=it.test_execution.stderr,
                    execution_time_ms=it.test_execution.execution_time_ms,
                    exit_code=it.test_execution.exit_code,
                    passed=it.test_execution.passed,
                    failed=it.test_execution.failed,
                )
            it_review = None
            if it.review is not None:
                r = it.review
                it_review = ReviewReportModel(
                    overall_score=r.overall_score,
                    strengths=r.strengths,
                    weaknesses=r.weaknesses,
                    pep8_issues=r.pep8_issues,
                    performance_suggestions=r.performance_suggestions,
                    security_concerns=r.security_concerns,
                    logic_issues=r.logic_issues,
                    maintainability=r.maintainability,
                    error_handling=r.error_handling,
                    recommendations=r.recommendations,
                    final_summary=r.final_summary,
                )
            it_debug = None
            if it.debug_report is not None:
                d = it.debug_report
                it_debug = WorkflowDebugReportModel(
                    issue_detected=d.issue_detected,
                    error_type=d.error_type,
                    root_cause=d.root_cause,
                    affected_component=d.affected_component,
                    explanation=d.explanation,
                    suggested_changes=d.suggested_changes,
                    corrected_code=d.corrected_code,
                    confidence=d.confidence,
                    final_summary=d.final_summary,
                )
            iteration_models.append(
                WorkflowIterationModel(
                    iteration_number=it.iteration_number,
                    generated_code=it.generated_code,
                    generated_tests=it.generated_tests,
                    execution=it_exec,
                    test_execution=it_test_exec,
                    review=it_review,
                    debug_report=it_debug,
                    test_error=it.test_error,
                    review_error=it.review_error,
                )
            )

        return RunWorkflowApiResponse(
            success=result.success,
            workflow=WorkflowResponseModel(
                planning=planning_model,
                generated_code=result.generated_code,
                generated_tests=result.generated_tests,
                execution=execution_model,
                test_execution=test_execution_model,
                review=review_model,
                workflow_status=result.workflow_status.value,
                execution_time_ms=result.execution_time_ms,
                error=result.error,
                test_error=result.test_error,
                iterations=iteration_models,
            ),
        )
    except ValueError as exc:
        logger.warning("POST /run-workflow: validation error: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /run-workflow: error")
        raise HTTPException(status_code=500, detail="Workflow execution failed") from exc


# ---------------------------------------------------------------------------
# Repository RAG endpoints
# ---------------------------------------------------------------------------


class IndexRepositoryPayload(BaseModel):
    repository_path: str = Field(..., min_length=1)


class IndexRepositoryApiResponse(BaseModel):
    success: bool
    repository: str
    file_count: int
    chunk_count: int
    error: str | None = None


class SearchResultModel(BaseModel):
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    repository: str
    chunk_index: int
    distance: float | None = None


class SearchRepositoryPayload(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    repository: str | None = None


class SearchRepositoryApiResponse(BaseModel):
    success: bool
    query: str
    results: list[SearchResultModel] = []
    error: str | None = None


class RagStatusApiResponse(BaseModel):
    success: bool
    chunk_count: int
    repositories: list[str] = []


@router.post(
    "/index-repository",
    response_model=IndexRepositoryApiResponse,
    tags=["rag"],
)
def index_repository(
    payload: IndexRepositoryPayload,
    use_case: IndexRepositoryUseCase = Depends(get_index_repository_use_case),
) -> IndexRepositoryApiResponse:
    """Index a repository directory into the RAG vector store.

    The path must resolve inside a configured allowed repository root.
    Arbitrary filesystem access is intentionally rejected.
    """
    path = (payload.repository_path or "").strip()
    if not path:
        logger.warning("POST /index-repository: empty repository_path")
        raise HTTPException(status_code=400, detail="repository_path must not be empty")

    logger.info("POST /index-repository: request received")
    try:
        result = use_case.execute(IndexRepositoryRequest(repository_path=path))
        return IndexRepositoryApiResponse(
            success=result.success,
            repository=result.repository,
            file_count=result.file_count,
            chunk_count=result.chunk_count,
            error=result.error,
        )
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /index-repository: error")
        raise HTTPException(status_code=500, detail="Repository indexing failed") from exc


@router.post(
    "/search-repository",
    response_model=SearchRepositoryApiResponse,
    tags=["rag"],
)
def search_repository(
    payload: SearchRepositoryPayload,
    use_case: SearchRepositoryUseCase = Depends(get_search_repository_use_case),
) -> SearchRepositoryApiResponse:
    """Search the indexed repository and return relevant code chunks."""
    query = (payload.query or "").strip()
    if not query:
        logger.warning("POST /search-repository: empty query")
        raise HTTPException(status_code=400, detail="query must not be empty")

    logger.info("POST /search-repository: request received")
    try:
        result = use_case.execute(
            SearchRepositoryRequest(
                query=query,
                top_k=payload.top_k,
                repository=payload.repository,
            )
        )
        results = [
            SearchResultModel(
                content=chunk.content,
                file_path=chunk.file_path,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                language=chunk.language,
                repository=chunk.repository,
                chunk_index=chunk.chunk_index,
                distance=chunk.distance,
            )
            for chunk in result.results
        ]
        return SearchRepositoryApiResponse(
            success=True,
            query=query,
            results=results,
        )
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /search-repository: error")
        raise HTTPException(
            status_code=500, detail="Repository search failed"
        ) from exc


@router.get(
    "/rag/status",
    response_model=RagStatusApiResponse,
    tags=["rag"],
)
def rag_status(
    use_case: GetRagStatusUseCase = Depends(get_rag_status_use_case),
) -> RagStatusApiResponse:
    """Return the current RAG index status."""
    logger.info("GET /rag/status: request received")
    try:
        result = use_case.execute()
        return RagStatusApiResponse(
            success=True,
            chunk_count=result.chunk_count,
            repositories=result.repositories,
        )
    except Exception as exc:
        logger.exception("GET /rag/status: error")
        raise HTTPException(status_code=500, detail="RAG status failed") from exc


# ---------------------------------------------------------------------------
# Repository-aware code modification endpoints
# ---------------------------------------------------------------------------


class ModifyRepositoryPayload(BaseModel):
    repository: str | None = Field(default=None, description="Repository id (falls back to any indexed repo)")
    request: str = Field(..., min_length=1, description="Natural-language modification request")
    dry_run: bool = True


class ChangeModel(BaseModel):
    file_path: str
    operation: str
    description: str | None = None
    original_hash: str | None = None


class ModifyRepositoryApiResponse(BaseModel):
    success: bool
    dry_run: bool
    changes: list[ChangeModel] = []
    diff: str | None = None
    validation_errors: list[str] = []
    applied_files: list[str] = []
    rollback: dict | None = None
    error: str | None = None
    status: str | None = None


@router.post(
    "/modify-repository",
    response_model=ModifyRepositoryApiResponse,
    tags=["rag"],
)
def modify_repository(
    payload: ModifyRepositoryPayload,
    use_case: ModifyRepositoryUseCase = Depends(get_modify_repository_use_case),
) -> ModifyRepositoryApiResponse:
    """Generate and (optionally) apply repository-aware code changes.

    The assistant retrieves repository context via RAG, plans, produces a
    proposed ChangeSet, validates every path (blocking path traversal and
    protected files), and only applies changes when ``dry_run`` is False.

    Changes are never applied automatically in dry-run mode; the caller must
    explicitly request application after inspecting the diff.
    """
    request_text = (payload.request or "").strip()
    if not request_text:
        logger.warning("POST /modify-repository: empty request")
        raise HTTPException(status_code=400, detail="request must not be empty")

    if payload.dry_run is None:
        raise HTTPException(status_code=400, detail="dry_run must be a boolean")

    logger.info(
        "POST /modify-repository: received (dry_run=%s, repository=%s)",
        payload.dry_run,
        payload.repository,
    )
    try:
        result = use_case.execute(
            ModifyRepositoryRequest(
                repository_path=(payload.repository or "").strip(),
                request=request_text,
                dry_run=payload.dry_run,
            )
        )
        change_set = getattr(result, "change_set", None)
        changes = change_set.changes if change_set is not None else []

        diff = None
        diffs = getattr(result, "diffs", []) or []
        if diffs:
            diff = "\n".join(d.diff_text for d in diffs)

        validation_errors: list[str] = []
        validation = getattr(result, "validation", None)
        if validation is not None:
            for r in getattr(validation, "results", []) or []:
                validation_errors.extend(getattr(r, "messages", []) or [])

        applied_files: list[str] = []
        rollback: dict | None = None
        application = getattr(result, "application", None)
        if application is not None:
            applied_files = list(getattr(application, "applied_files", []) or [])
            records = getattr(application, "rollback_records", []) or []
            rollback = {
                "files": [
                    {
                        "file_path": r.file_path,
                        "was_created": r.was_created,
                    }
                    for r in records
                ]
            }

        return ModifyRepositoryApiResponse(
            success=result.success,
            dry_run=result.dry_run,
            changes=[
                ChangeModel(
                    file_path=c.file_path,
                    operation=c.operation.value,
                    description=c.description,
                    original_hash=c.original_hash,
                )
                for c in changes
            ],
            diff=diff,
            validation_errors=validation_errors,
            applied_files=applied_files,
            rollback=rollback,
            error=result.error,
            status=(result.status.value if result.status is not None else None),
        )
    except TypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("POST /modify-repository: error")
        raise HTTPException(
            status_code=500, detail="Repository modification failed"
        ) from exc

"""Tests for the repository-aware workflow branch.

These tests use fake collaborators only; no LLM, ChromaDB, or filesystem
mutation is involved.
"""

from __future__ import annotations

from agents.planner_agent import ImplementationPlan
from agents.code_reviewer_agent import ReviewReport
from agents.test_generator_agent import TestGenerationReport
from backend.application.modify_repository_use_cases import (
    ModifyRepositoryRequest,
    ModifyRepositoryResult,
    ModifyRepositoryStatus,
)
from backend.application.planning_use_cases import GeneratePlanResult
from backend.application.rag_use_cases import SearchRepositoryResult
from backend.application.review_use_cases import ReviewCodeResult
from backend.application.test_generation_use_cases import GenerateTestsResult
from backend.application.workflow_use_cases import (
    RunWorkflowUseCase,
    WorkflowRequest,
    WorkflowStatus,
)
from backend.domain.change_models import (
    ApplicationResult,
    ChangeOperation,
    ChangeSet,
    ChangeValidationResult,
    DiffEntry,
    FileChange,
    ValidationReport,
)
from backend.tools.python_executor import ExecutionResponse
from backend.tools.test_executor import TestExecutionResponse
from rag.retriever import RetrievedChunk


def _plan() -> ImplementationPlan:
    return ImplementationPlan(
        problem_summary="Update repository feature",
        project_type="python-service",
        requirements=["update auth service"],
        modules=["backend/auth/service.py"],
        functions=["reset_password"],
        classes=[],
        external_libraries=[],
        database_needed=False,
        api_needed=[],
        algorithm="simple",
        edge_cases=[],
        estimated_complexity="low",
        future_improvements=[],
    )


def _test_report() -> TestGenerationReport:
    return TestGenerationReport(
        test_overview="overview",
        test_framework="pytest",
        test_cases=[],
        edge_cases=[],
        negative_cases=[],
        coverage_suggestions=[],
        generated_test_code="def test_repo():\n    assert True\n",
        final_summary="summary",
    )


def _review_report(score: int = 90) -> ReviewReport:
    return ReviewReport(
        overall_score=score,
        strengths=["good"],
        weaknesses=[],
        pep8_issues=[],
        performance_suggestions=[],
        security_concerns=[],
        logic_issues=[],
        maintainability=[],
        error_handling=[],
        recommendations=["ok"],
        final_summary="looks good",
    )


def _repo_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content="def login():\n    pass\n",
        file_path="backend/auth/service.py",
        start_line=1,
        end_line=2,
        language="python",
        repository="repo",
        chunk_index=0,
        distance=0.1,
    )


def _repo_change_set(content: str = "def reset_password():\n    return True\n") -> ChangeSet:
    return ChangeSet(
        changes=[
            FileChange(
                file_path="backend/auth/service.py",
                operation=ChangeOperation.MODIFY,
                new_content=content,
                original_hash="abc123",
                description="Add password reset",
            )
        ],
        summary="update auth service",
    )


class FakePlan:
    def __init__(self, plan=None):
        self.plan = plan or _plan()
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        return GeneratePlanResult(plan=self.plan)


class FakeCode:
    def __init__(self, code="unused"):
        self.code = code

    def execute(self, request):
        return type("R", (), {"generated_code": self.code})()


class FakeSearch:
    def __init__(self):
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        return SearchRepositoryResult(
            success=True,
            query=request.query,
            results=[_repo_chunk()],
        )


class FakeRepoCoder:
    def __init__(self, change_set=None):
        self.change_set = change_set or _repo_change_set()
        self.calls = []

    def generate_changes(
        self,
        prompt,
        *,
        implementation_plan=None,
        retrieved_context=None,
        plan_summary=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "implementation_plan": implementation_plan,
                "retrieved_context": retrieved_context,
                "plan_summary": plan_summary,
            }
        )
        return self.change_set


class FakeRepoModify:
    def __init__(
        self,
        *,
        proposal_success: bool = True,
        application_success: bool = True,
        proposal_valid: bool = True,
        application_valid: bool = True,
        proposal_error: str | None = "failed validation",
        application_error: str | None = "apply failed",
    ):
        self.calls = []
        self.proposal_success = proposal_success
        self.application_success = application_success
        self.proposal_valid = proposal_valid
        self.application_valid = application_valid
        self.proposal_error = proposal_error
        self.application_error = application_error

    def execute(self, request: ModifyRepositoryRequest) -> ModifyRepositoryResult:
        self.calls.append(request)
        is_proposal = request.dry_run
        valid = self.proposal_valid if is_proposal else self.application_valid
        success = self.proposal_success if is_proposal else self.application_success
        error = self.proposal_error if is_proposal else self.application_error

        validation = ValidationReport(
            valid=valid,
            results=[
                ChangeValidationResult(
                    valid=valid,
                    file_path=request.change_set.changes[0].file_path,
                    operation=request.change_set.changes[0].operation.value,
                    messages=[] if valid else ["validation failed"],
                )
            ],
        )
        diff = [
            DiffEntry(
                file_path=change.file_path,
                operation=change.operation.value,
                diff_text=f"diff:{change.file_path}",
            )
            for change in request.change_set.changes
        ]
        applied_files = (
            []
            if request.dry_run or not success
            else [change.file_path for change in request.change_set.changes]
        )
        application = ApplicationResult(
            success=success,
            applied_files=applied_files,
            rollback_records=[],
            dry_run=request.dry_run,
            error=None if success else error,
        )
        status = (
            ModifyRepositoryStatus.PROPOSED
            if request.dry_run
            else (
                ModifyRepositoryStatus.APPLIED
                if success
                else ModifyRepositoryStatus.APPLICATION_FAILED
            )
        )
        return ModifyRepositoryResult(
            success=success,
            status=status,
            repository_path=request.repository_path or request.repository_root or "",
            dry_run=request.dry_run,
            change_set=request.change_set,
            validation=validation,
            diffs=diff,
            application=application,
            error=None if success else error,
        )


class FakeRepoDebugger:
    def __init__(self, corrected_change_set=None):
        self.corrected_change_set = corrected_change_set or _repo_change_set(
            "def reset_password():\n    return 'fixed'\n"
        )
        self.calls = []

    def correct_changes(self, change_set, *, feedback, retrieved_context=None):
        self.calls.append(
            {
                "change_set": change_set,
                "feedback": feedback,
                "retrieved_context": retrieved_context,
            }
        )
        return self.corrected_change_set


class FakeTestGeneration:
    def __init__(self):
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        return GenerateTestsResult(report=_test_report())


class SequenceTestExecution:
    def __init__(self, successes):
        self.successes = list(successes)
        self.calls = 0

    def execute(self, request):
        index = min(self.calls, len(self.successes) - 1)
        success = self.successes[index]
        self.calls += 1
        return TestExecutionResponse(
            success=success,
            stdout="1 passed" if success else "1 failed",
            stderr="" if success else "failure",
            execution_time_ms=1.0,
            exit_code=0 if success else 1,
            passed=1 if success else 0,
            failed=0 if success else 1,
        )


class SequenceReview:
    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0

    def execute(self, request):
        index = min(self.calls, len(self.scores) - 1)
        score = self.scores[index]
        self.calls += 1
        return ReviewCodeResult(report=_review_report(score))


def _build_use_case(
    *,
    repo_modify: FakeRepoModify,
    repo_debugger: FakeRepoDebugger | None = None,
    test_successes=(True,),
    review_scores=(90,),
    max_iterations: int = 3,
) -> RunWorkflowUseCase:
    return RunWorkflowUseCase(
        plan_use_case=FakePlan(),
        coder_use_case=FakeCode(),
        test_generation_use_case=FakeTestGeneration(),
        execute_use_case=type(
            "FakeExecute",
            (),
            {"execute": lambda self, request: ExecutionResponse(True, "", "", 1.0, 0)},
        )(),
        test_execution_use_case=SequenceTestExecution(test_successes),
        review_use_case=SequenceReview(review_scores),
        debug_use_case=None,
        repository_search_use_case=FakeSearch(),
        repository_coder_agent=FakeRepoCoder(),
        repository_modify_use_case=repo_modify,
        repository_debugger_agent=repo_debugger,
        max_iterations=max_iterations,
    )


def _request(*, apply: bool = True) -> WorkflowRequest:
    return WorkflowRequest(
        prompt="add password reset",
        repository_root="C:/repo",
        apply_repository_changes=apply,
    )


def test_repository_workflow_successful_modification() -> None:
    run = _build_use_case(repo_modify=FakeRepoModify())

    result = run.execute(_request(apply=True))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert result.repository_root == "C:/repo"
    assert result.repository_proposal is not None
    assert result.repository_application is not None
    assert result.repository_application.success is True
    assert result.repository_change_set is not None
    assert result.repository_change_set.changes[0].file_path == "backend/auth/service.py"


def test_repository_workflow_validation_failure() -> None:
    run = _build_use_case(
        repo_modify=FakeRepoModify(proposal_success=False, proposal_valid=False)
    )

    result = run.execute(_request(apply=True))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.REPOSITORY_VALIDATION_FAILED
    assert result.repository_proposal is not None
    assert result.repository_proposal.success is False
    assert result.repository_application is None


def test_repository_workflow_application_failure() -> None:
    run = _build_use_case(
        repo_modify=FakeRepoModify(
            proposal_success=True,
            application_success=False,
            application_valid=True,
        )
    )

    result = run.execute(_request(apply=True))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.REPOSITORY_APPLICATION_FAILED
    assert result.repository_proposal is not None
    assert result.repository_application is not None
    assert result.repository_application.success is False


def test_repository_workflow_test_failure_triggers_debugger() -> None:
    debugger = FakeRepoDebugger()
    run = _build_use_case(
        repo_modify=FakeRepoModify(),
        repo_debugger=debugger,
        test_successes=(False, True),
    )

    result = run.execute(_request(apply=True))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert len(debugger.calls) == 1
    assert result.iterations[-1].debug_report is None


def test_repository_workflow_reviewer_failure_triggers_debugger() -> None:
    debugger = FakeRepoDebugger()
    run = _build_use_case(
        repo_modify=FakeRepoModify(),
        repo_debugger=debugger,
        review_scores=(30, 90),
    )

    result = run.execute(_request(apply=True))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert len(debugger.calls) == 1


def test_repository_workflow_debugger_correction() -> None:
    corrected = _repo_change_set("def reset_password():\n    return 'fixed'\n")
    debugger = FakeRepoDebugger(corrected_change_set=corrected)
    run = _build_use_case(
        repo_modify=FakeRepoModify(),
        repo_debugger=debugger,
        test_successes=(False, True),
    )

    result = run.execute(_request(apply=True))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert result.repository_change_set == corrected
    assert result.repository_application is not None
    assert result.repository_application.success is True


def test_repository_workflow_max_iterations_reached() -> None:
    debugger = FakeRepoDebugger()
    run = _build_use_case(
        repo_modify=FakeRepoModify(),
        repo_debugger=debugger,
        test_successes=(False, False),
        max_iterations=1,
    )

    result = run.execute(_request(apply=True))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.MAX_ITERATIONS_REACHED
    assert len(debugger.calls) == 0

"""Unit tests for RunWorkflowUseCase.

Uses fake/mocked use-cases so no LLM/network is required.
"""

from __future__ import annotations

from agents.documentation_agent import DocumentationReport
from agents.planner_agent import ImplementationPlan
from backend.application.documentation_use_cases import (
    GenerateDocumentationRequest,
    GenerateDocumentationResult,
)
from backend.application.test_generation_use_cases import (
    GenerateTestsResult,
)
from backend.application.workflow_use_cases import (
    RunWorkflowUseCase,
    WorkflowRequest,
    WorkflowResult,
    WorkflowStatus,
)
from agents.test_generator_agent import TestGenerationReport
from backend.tools.python_executor import ExecutionResponse
from backend.tools.test_executor import TestExecutionResponse


def make_plan() -> ImplementationPlan:
    return ImplementationPlan(
        problem_summary="Building a calculator",
        project_type="console",
        requirements=["add"],
        modules=["calculator"],
        functions=["add"],
        classes=[],
        external_libraries=[],
        database_needed=False,
        api_needed=[],
        algorithm="n/a",
        edge_cases=[],
        estimated_complexity="low",
        future_improvements=[],
    )


def make_test_report() -> TestGenerationReport:
    return TestGenerationReport(
        test_overview="overview",
        test_framework="pytest",
        test_cases=[],
        edge_cases=[],
        negative_cases=[],
        coverage_suggestions=[],
        generated_test_code="def test_add():\n    assert True\n",
        final_summary="summary",
    )


class FakePlan:
    def __init__(self, plan=None, exc=None):
        self.plan = plan
        self.exc = exc

    def execute(self, request):
        if self.exc is not None:
            raise self.exc

        class _R:
            def __init__(self, plan):
                self.plan = plan

        return _R(self.plan)


class FakeCode:
    def __init__(self, code=None, exc=None):
        self.code = code
        self.exc = exc

    def execute(self, request):
        if self.exc is not None:
            raise self.exc

        class _R:
            def __init__(self, code):
                self.generated_code = code

        return _R(self.code)


class FakeTestGeneration:
    def __init__(self, report=None, exc=None):
        self.report = report
        self.exc = exc

    def execute(self, request):
        if self.exc is not None:
            raise self.exc
        return GenerateTestsResult(report=self.report)


class FakeExecute:
    def __init__(self, resp=None, exc=None):
        self.resp = resp
        self.exc = exc

    def execute(self, request):
        if self.exc is not None:
            raise self.exc
        return self.resp


class FakeTestExecute:
    def __init__(self, resp=None, exc=None):
        self.resp = resp
        self.exc = exc

    def execute(self, request):
        if self.exc is not None:
            raise self.exc
        return self.resp


class FakeReview:
    def __init__(self, report=None, exc=None):
        self.report = report
        self.exc = exc
        self.last_request = None

    def execute(self, request):
        self.last_request = request
        if self.exc is not None:
            raise self.exc

        class _R:
            def __init__(self, report):
                self.report = report

        return _R(self.report)


def make_documentation_report() -> DocumentationReport:
    return DocumentationReport(
        summary="Calculator documentation",
        module_description="Module providing math operations",
        function_docs=[{"name": "add"}],
        class_docs=[],
        usage_examples=["add(1, 2)"],
        markdown_documentation="# Calculator Docs",
    )


class FakeDocumentation:
    def __init__(self, report=None, exc=None):
        self.report = report
        self.exc = exc
        self.calls = 0
        self.last_request = None

    def execute(self, request: GenerateDocumentationRequest) -> GenerateDocumentationResult:
        self.calls += 1
        self.last_request = request
        if self.exc is not None:
            raise self.exc
        return GenerateDocumentationResult(report=self.report)


def build_workflow(
    *,
    test_gen_exc=None,
    execute_exc=None,
    test_exc=None,
    review_exc=None,
    doc_use_case=None,
):
    return RunWorkflowUseCase(
        plan_use_case=FakePlan(plan=make_plan()),
        coder_use_case=FakeCode(code="def add(a,b): return a+b"),
        test_generation_use_case=FakeTestGeneration(
            report=None if test_gen_exc else make_test_report(),
            exc=test_gen_exc,
        ),
        execute_use_case=FakeExecute(
            resp=None
            if execute_exc
            else ExecutionResponse(True, "1", "", 1.0, 0),
            exc=execute_exc,
        ),
        test_execution_use_case=FakeTestExecute(
            resp=None
            if test_exc
            else TestExecutionResponse(True, "1 passed", "", 1.0, 0, 1, 0),
            exc=test_exc,
        ),
        review_use_case=FakeReview(
            report=None if review_exc else object(),
            exc=review_exc,
        ),
        documentation_use_case=doc_use_case,
    )


def test_complete_success() -> None:
    run = build_workflow()
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert isinstance(result, WorkflowResult)
    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert result.planning is not None
    assert result.generated_code is not None
    assert result.generated_tests is not None
    assert result.execution is not None
    assert result.test_execution is not None
    assert result.test_execution.passed == 1
    assert result.review is not None
    assert result.error is None
    assert result.test_error is None


def test_planner_failure() -> None:
    run = build_workflow()
    run._plan_use_case = FakePlan(exc=RuntimeError("planner down"))
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.PLANNING_FAILED
    assert result.planning is None
    assert result.generated_code is None


def test_coder_failure() -> None:
    run = build_workflow()
    run._coder_use_case = FakeCode(exc=RuntimeError("coder down"))
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.CODING_FAILED
    assert result.generated_code is None


def test_test_generation_failure_continues() -> None:
    run = build_workflow(test_gen_exc=RuntimeError("test gen down"))
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED_WITH_WARNINGS
    assert result.generated_tests is None
    assert result.test_execution is None
    assert result.test_error is not None
    # Application execution and review should still happen.
    assert result.generated_code is not None
    assert result.review is not None


def test_application_execution_failure_continues() -> None:
    run = build_workflow(execute_exc=RuntimeError("exec down"))
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.execution is None
    assert result.review is not None


def test_test_execution_failure_continues() -> None:
    run = build_workflow(test_exc=RuntimeError("test exec down"))
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.test_execution is None
    assert result.generated_tests is not None
    assert result.review is not None


def test_reviewer_failure_keeps_results() -> None:
    run = build_workflow(review_exc=RuntimeError("review down"))
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED_WITH_WARNINGS
    assert result.review is None
    assert result.error is not None
    # Previously completed results are preserved.
    assert result.generated_code is not None
    assert result.generated_tests is not None
    assert result.planning is not None


def test_test_execution_skipped_when_no_tests() -> None:
    run = build_workflow(test_gen_exc=RuntimeError("no tests"))
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.generated_tests is None
    assert result.test_execution is None


def test_workflow_status_values() -> None:
    # Ensure the enum string values remain stable/backward-compatible.
    assert WorkflowStatus.COMPLETED.value == "completed"
    assert WorkflowStatus.COMPLETED_WITH_WARNINGS.value == "completed_with_warnings"
    assert WorkflowStatus.PLANNING_FAILED.value == "planning_failed"
    assert WorkflowStatus.CODING_FAILED.value == "coding_failed"


def test_workflow_generates_documentation_on_success() -> None:
    doc_report = make_documentation_report()
    doc_use_case = FakeDocumentation(report=doc_report)
    run = build_workflow(doc_use_case=doc_use_case)
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert result.documentation is doc_report
    assert result.documentation_error is None
    assert doc_use_case.calls == 1
    assert doc_use_case.last_request is not None
    assert doc_use_case.last_request.code == result.generated_code
    assert doc_use_case.last_request.retrieved_context is None


def test_workflow_documentation_failure_is_non_blocking() -> None:
    doc_use_case = FakeDocumentation(exc=RuntimeError("Documentation LLM timeout"))
    run = build_workflow(doc_use_case=doc_use_case)
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED_WITH_WARNINGS
    assert result.documentation is None
    assert result.documentation_error == "Documentation LLM timeout"
    assert doc_use_case.calls == 1
    assert result.generated_code is not None


def test_workflow_without_documentation_use_case() -> None:
    run = build_workflow(doc_use_case=None)
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert result.documentation is None
    assert result.documentation_error is None


def test_workflow_fatal_planning_does_not_call_documentation() -> None:
    doc_use_case = FakeDocumentation(report=make_documentation_report())
    run = RunWorkflowUseCase(
        plan_use_case=FakePlan(exc=RuntimeError("planner failed")),
        coder_use_case=FakeCode(code="pass"),
        test_generation_use_case=FakeTestGeneration(),
        execute_use_case=FakeExecute(),
        test_execution_use_case=FakeTestExecute(),
        review_use_case=FakeReview(),
        documentation_use_case=doc_use_case,
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.PLANNING_FAILED
    assert result.documentation is None
    assert doc_use_case.calls == 0


def test_workflow_fatal_coding_does_not_call_documentation() -> None:
    doc_use_case = FakeDocumentation(report=make_documentation_report())
    run = RunWorkflowUseCase(
        plan_use_case=FakePlan(plan=make_plan()),
        coder_use_case=FakeCode(exc=RuntimeError("coder failed")),
        test_generation_use_case=FakeTestGeneration(),
        execute_use_case=FakeExecute(),
        test_execution_use_case=FakeTestExecute(),
        review_use_case=FakeReview(),
        documentation_use_case=doc_use_case,
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.CODING_FAILED
    assert result.documentation is None
    assert doc_use_case.calls == 0


def test_workflow_documentation_called_only_once_with_self_correction() -> None:
    from backend.application.debugging_use_cases import DebugCodeResult
    from agents.debugger_agent import DebugReport

    class FakeDebugUseCase:
        def __init__(self, corrected_code: str):
            self.corrected_code = corrected_code
            self.calls = 0

        def execute(self, request):
            self.calls += 1
            return DebugCodeResult(
                report=DebugReport(
                    issue_detected=True,
                    error_type="Fix",
                    root_cause="Bug",
                    affected_component="calc",
                    explanation="Fix bug",
                    suggested_changes=[],
                    corrected_code=self.corrected_code,
                    confidence=1.0,
                    final_summary="Fixed",
                )
            )

    class MultiExecute:
        def __init__(self):
            self.calls = 0

        def execute(self, request):
            self.calls += 1
            if self.calls == 1:
                return ExecutionResponse(False, "", "error", 1.0, 1)
            return ExecutionResponse(True, "ok", "", 1.0, 0)

    doc_report = make_documentation_report()
    doc_use_case = FakeDocumentation(report=doc_report)
    debugger = FakeDebugUseCase(corrected_code="def add(a, b): return a + b # fixed")

    run = RunWorkflowUseCase(
        plan_use_case=FakePlan(plan=make_plan()),
        coder_use_case=FakeCode(code="def add(a, b): broken"),
        test_generation_use_case=FakeTestGeneration(report=make_test_report()),
        execute_use_case=MultiExecute(),
        test_execution_use_case=FakeTestExecute(
            resp=TestExecutionResponse(True, "1 passed", "", 1.0, 0, 1, 0)
        ),
        review_use_case=FakeReview(report=object()),
        debug_use_case=debugger,
        documentation_use_case=doc_use_case,
        max_iterations=2,
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert len(result.iterations) == 2
    assert doc_use_case.calls == 1
    assert doc_use_case.last_request is not None
    assert doc_use_case.last_request.code == "def add(a, b): return a + b # fixed"
    assert result.documentation is doc_report

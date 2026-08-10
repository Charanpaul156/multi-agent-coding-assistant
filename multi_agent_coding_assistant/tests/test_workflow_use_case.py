"""Unit tests for RunWorkflowUseCase.

Uses fake/mocked use-cases so no LLM/network is required.
"""

from __future__ import annotations

from agents.planner_agent import ImplementationPlan
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


def build_workflow(
    *,
    test_gen_exc=None,
    execute_exc=None,
    test_exc=None,
    review_exc=None,
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

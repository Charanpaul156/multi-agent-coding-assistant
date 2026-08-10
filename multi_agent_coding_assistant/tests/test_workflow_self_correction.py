"""Unit tests for the workflow self-correction loop.

Tests that the orchestrator correctly triggers the Debugger on critical
failures, validates corrections through the pipeline, enforces the maximum
iteration limit, and does NOT trigger debugging on non-critical feedback.

Uses fake/mocked use-cases so no LLM/network is required.
"""

from __future__ import annotations

from agents.planner_agent import ImplementationPlan
from agents.code_reviewer_agent import ReviewReport
from agents.debugger_agent import DebugReport
from backend.application.debugging_use_cases import (
    DebugCodeRequest,
    DebugCodeResult,
)
from backend.application.workflow_use_cases import (
    RunWorkflowUseCase,
    WorkflowRequest,
    WorkflowStatus,
)
from backend.tools.python_executor import ExecutionResponse
from backend.tools.test_executor import TestExecutionResponse


def make_plan() -> ImplementationPlan:
    return ImplementationPlan(
        problem_summary="Building a calculator",
        project_type="console",
        requirements=["divide"],
        modules=["calculator"],
        functions=["divide"],
        classes=[],
        external_libraries=[],
        database_needed=False,
        api_needed=[],
        algorithm="n/a",
        edge_cases=[],
        estimated_complexity="low",
        future_improvements=[],
    )


def make_review(
    *,
    overall_score: int = 90,
    logic_issues=None,
    security_concerns=None,
) -> ReviewReport:
    return ReviewReport(
        overall_score=overall_score,
        strengths=["ok"],
        weaknesses=[],
        pep8_issues=[],
        performance_suggestions=[],
        security_concerns=security_concerns or [],
        logic_issues=logic_issues or [],
        maintainability=[],
        error_handling=[],
        recommendations=["none"],
        final_summary="review complete",
    )


def make_debug_report(corrected_code: str) -> DebugReport:
    return DebugReport(
        issue_detected=True,
        error_type="ZeroDivisionError",
        root_cause="division by zero",
        affected_component="divide",
        explanation="No guard for zero divisor.",
        suggested_changes=["Add a guard."],
        corrected_code=corrected_code,
        confidence=0.9,
        final_summary="fixed",
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
    def __init__(self, tests=None, exc=None):
        self.tests = tests
        self.exc = exc

    def execute(self, request):
        if self.exc is not None:
            raise self.exc

        class _R:
            def __init__(self, tests):
                self.report = type("T", (), {"generated_test_code": tests})()

        return _R(self.tests)


class FakeExecute:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No more execution responses configured")
        return self.responses.pop(0)


class FakeTestExecute:
    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No more test-execution responses configured")
        return self.responses.pop(0)


class FakeReview:
    def __init__(self, reports=None, exc=None):
        self.reports = list(reports or [])
        self.exc = exc
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if not self.reports:
            raise AssertionError("No more review responses configured")
        return type("R", (), {"report": self.reports.pop(0)})()


class FakeDebug:
    def __init__(self, reports=None, exc=None):
        self.reports = list(reports or [])
        self.exc = exc
        self.calls = 0

    def execute(self, request: DebugCodeRequest):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if not self.reports:
            raise AssertionError("No more debug responses configured")
        return DebugCodeResult(report=self.reports.pop(0))


def _ok_exec() -> ExecutionResponse:
    return ExecutionResponse(True, "ok", "", 1.0, 0)


def _fail_exec() -> ExecutionResponse:
    return ExecutionResponse(False, "", "ZeroDivisionError", 1.0, 1)


def _ok_test() -> TestExecutionResponse:
    return TestExecutionResponse(True, "1 passed", "", 1.0, 0, 1, 0)


def _fail_test() -> TestExecutionResponse:
    return TestExecutionResponse(False, "1 failed", "", 1.0, 1, 0, 1)


def build_workflow(**overrides):
    defaults = dict(
        plan_use_case=FakePlan(plan=make_plan()),
        coder_use_case=FakeCode(code="def divide(a,b): return a/b"),
        test_generation_use_case=FakeTestGeneration(tests="def test_divide():\n    pass\n"),
        execute_use_case=FakeExecute(responses=[_ok_exec()]),
        test_execution_use_case=FakeTestExecute(responses=[_ok_test()]),
        review_use_case=FakeReview(reports=[make_review()]),
    )
    defaults.update(overrides)
    return RunWorkflowUseCase(**defaults)


def test_first_iteration_succeeds_no_debugger() -> None:
    run = build_workflow()
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert len(result.iterations) == 1
    assert result.iterations[0].debug_report is None


def test_runtime_failure_triggers_debugger_then_succeeds() -> None:
    # Iteration 1 fails execution; debugger returns a corrected code that
    # passes on iteration 2.
    run = build_workflow(
        execute_use_case=FakeExecute(responses=[_fail_exec(), _ok_exec()]),
        test_execution_use_case=FakeTestExecute(responses=[_ok_test(), _ok_test()]),
        review_use_case=FakeReview(reports=[make_review(), make_review()]),
        debug_use_case=FakeDebug(
            reports=[make_debug_report("def divide(a,b):\n    if b==0: return None\n    return a/b\n")]
        ),
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert len(result.iterations) == 2
    assert result.iterations[0].debug_report is not None
    assert result.iterations[1].debug_report is None
    assert "if b==0" in result.iterations[1].generated_code


def test_test_failure_triggers_debugger() -> None:
    run = build_workflow(
        execute_use_case=FakeExecute(responses=[_ok_exec(), _ok_exec()]),
        test_execution_use_case=FakeTestExecute(responses=[_fail_test(), _ok_test()]),
        review_use_case=FakeReview(reports=[make_review(), make_review()]),
        debug_use_case=FakeDebug(
            reports=[make_debug_report("def divide(a,b):\n    return a/b\n")]
        ),
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert len(result.iterations) == 2
    assert result.iterations[0].debug_report is not None


def test_reviewer_critical_logic_issue_triggers_debugger() -> None:
    crit_review = make_review(logic_issues=["Division by zero not handled"])
    run = build_workflow(
        execute_use_case=FakeExecute(responses=[_ok_exec(), _ok_exec()]),
        test_execution_use_case=FakeTestExecute(responses=[_ok_test(), _ok_test()]),
        review_use_case=FakeReview(reports=[crit_review, make_review()]),
        debug_use_case=FakeDebug(
            reports=[make_debug_report("def divide(a,b):\n    return a/b\n")]
        ),
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert len(result.iterations) == 2
    assert result.iterations[0].debug_report is not None


def test_reviewer_non_critical_does_not_trigger_debugger() -> None:
    non_crit = make_review(
        overall_score=85,
        logic_issues=[],
        security_concerns=[],
    )
    run = build_workflow(review_use_case=FakeReview(reports=[non_crit]))
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert result.workflow_status == WorkflowStatus.COMPLETED
    assert len(result.iterations) == 1
    assert result.iterations[0].debug_report is None


def test_corrected_code_still_fails_continues_loop() -> None:
    # Iteration 1 fails, debugger returns bad correction, iteration 2 still
    # fails, then iteration 3 succeeds.
    run = build_workflow(
        execute_use_case=FakeExecute(
            responses=[_fail_exec(), _fail_exec(), _ok_exec()]
        ),
        test_execution_use_case=FakeTestExecute(
            responses=[_ok_test(), _ok_test(), _ok_test()]
        ),
        review_use_case=FakeReview(
            reports=[make_review(), make_review(), make_review()]
        ),
        debug_use_case=FakeDebug(
            reports=[
                make_debug_report("def divide(a,b):\n    return a/b\n"),
                make_debug_report("def divide(a,b):\n    if b==0: return None\n    return a/b\n"),
            ]
        ),
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert len(result.iterations) == 3
    assert result.iterations[0].debug_report is not None
    assert result.iterations[1].debug_report is not None
    assert result.iterations[2].debug_report is None


def test_max_iterations_reached() -> None:
    # Every iteration fails; debugger keeps producing corrections but they
    # never pass. Budget is exhausted -> max_iterations_reached.
    run = build_workflow(
        execute_use_case=FakeExecute(
            responses=[_fail_exec(), _fail_exec(), _fail_exec()]
        ),
        test_execution_use_case=FakeTestExecute(
            responses=[_ok_test(), _ok_test(), _ok_test()]
        ),
        review_use_case=FakeReview(
            reports=[make_review(), make_review(), make_review()]
        ),
        debug_use_case=FakeDebug(
            reports=[
                make_debug_report("def divide(a,b):\n    return a/b\n"),
                make_debug_report("def divide(a,b):\n    return a/b\n"),
            ]
        ),
        max_iterations=3,
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.MAX_ITERATIONS_REACHED
    assert len(result.iterations) == 3


def test_debugger_failure_stops_loop() -> None:
    # Iteration 1 fails; debugger raises -> debugger_failed.
    run = build_workflow(
        execute_use_case=FakeExecute(responses=[_fail_exec()]),
        test_execution_use_case=FakeTestExecute(responses=[_ok_test()]),
        review_use_case=FakeReview(reports=[make_review()]),
        debug_use_case=FakeDebug(exc=RuntimeError("debugger down")),
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.DEBUGGER_FAILED
    assert len(result.iterations) == 1


def test_debugger_not_available_stops_loop() -> None:
    # Iteration 1 fails but no debugger is wired -> debugger_failed.
    run = build_workflow(
        execute_use_case=FakeExecute(responses=[_fail_exec()]),
        test_execution_use_case=FakeTestExecute(responses=[_ok_test()]),
        review_use_case=FakeReview(reports=[make_review()]),
        debug_use_case=None,
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is False
    assert result.workflow_status == WorkflowStatus.DEBUGGER_FAILED


def test_reviewer_low_score_triggers_debugger() -> None:
    low_score = make_review(overall_score=40, logic_issues=[], security_concerns=[])
    run = build_workflow(
        execute_use_case=FakeExecute(responses=[_ok_exec(), _ok_exec()]),
        test_execution_use_case=FakeTestExecute(responses=[_ok_test(), _ok_test()]),
        review_use_case=FakeReview(reports=[low_score, make_review()]),
        debug_use_case=FakeDebug(
            reports=[make_debug_report("def divide(a,b):\n    return a/b\n")]
        ),
    )
    result = run.execute(WorkflowRequest(prompt="build calculator"))

    assert result.success is True
    assert len(result.iterations) == 2
    assert result.iterations[0].debug_report is not None

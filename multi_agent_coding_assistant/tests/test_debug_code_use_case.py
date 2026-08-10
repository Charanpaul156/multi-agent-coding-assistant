"""Unit tests for DebugCodeUseCase."""

from __future__ import annotations

import pytest

from agents.debugger_agent import DebugReport
from backend.application.debugging_use_cases import (
    DebugCodeRequest,
    DebugCodeResult,
    DebugCodeUseCase,
)


def _make_report() -> DebugReport:
    return DebugReport(
        issue_detected=True,
        error_type="ZeroDivisionError",
        root_cause="Division by zero",
        affected_component="divide",
        explanation="No guard for zero divisor.",
        suggested_changes=["Add a guard."],
        corrected_code="def divide(a, b):\n    if b == 0: return None\n    return a / b\n",
        confidence=0.9,
        final_summary="Fixed.",
    )


class _FakeDebuggerAgent:
    def __init__(self, report=None, exc=None):
        self.report = report
        self.exc = exc
        self.last_kwargs = None

    def debug_code(self, *args, **kwargs):
        self.last_kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.report


def test_debug_code_valid_request() -> None:
    agent = _FakeDebuggerAgent(report=_make_report())
    use_case = DebugCodeUseCase(debugger_agent=agent)  # type: ignore[arg-type]

    result = use_case.execute(
        DebugCodeRequest(
            generated_code="def divide(a, b):\n    return a / b\n",
            execution_stderr="ZeroDivisionError",
            execution_exit_code=1,
        )
    )

    assert isinstance(result, DebugCodeResult)
    assert result.report.issue_detected is True
    assert agent.last_kwargs["application_exit_code"] == 1
    assert agent.last_kwargs["application_stderr"] == "ZeroDivisionError"


def test_debug_code_empty_code() -> None:
    use_case = DebugCodeUseCase(debugger_agent=_FakeDebuggerAgent(report=_make_report()))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        use_case.execute(DebugCodeRequest(generated_code="   "))


def test_debug_code_non_string_code() -> None:
    use_case = DebugCodeUseCase(debugger_agent=_FakeDebuggerAgent(report=_make_report()))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        use_case.execute(DebugCodeRequest(generated_code=123))  # type: ignore[arg-type]


def test_debug_code_agent_failure() -> None:
    agent = _FakeDebuggerAgent(exc=RuntimeError("debugger down"))
    use_case = DebugCodeUseCase(debugger_agent=agent)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        use_case.execute(DebugCodeRequest(generated_code="def f(): pass\n"))

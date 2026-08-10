"""Application use-cases for debugging.

Use-cases are framework-independent. Application layer must not depend on
FastAPI or Pydantic. Only dataclasses are used here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from agents.debugger_agent import DebugReport, DebuggerAgent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DebugCodeRequest:
    """Request DTO for debugging failed generated code.

    ``generated_code`` is always required. All other inputs are optional so
    the debugger can handle runtime failures, test failures, and
    reviewer-detected logic problems independently.
    """

    generated_code: str
    generated_tests: Optional[str] = None
    execution_stdout: Optional[str] = None
    execution_stderr: Optional[str] = None
    execution_exit_code: Optional[int] = None
    test_stdout: Optional[str] = None
    test_stderr: Optional[str] = None
    test_exit_code: Optional[int] = None
    reviewer_feedback: Optional[str] = None


@dataclass(frozen=True)
class DebugCodeResult:
    """Result DTO for a debugging session."""

    report: DebugReport


class DebugCodeUseCase:
    """Use-case: produce a structured DebugReport for failed generated code.

    The DebuggerAgent analyzes code, tests, execution results and reviewer
    feedback but never executes, generates-on-disk, or modifies code.
    """

    def __init__(self, debugger_agent: DebuggerAgent) -> None:
        self._debugger_agent = debugger_agent

    def execute(self, request: DebugCodeRequest) -> DebugCodeResult:
        """Execute the debugging use-case."""

        if not isinstance(request, DebugCodeRequest):
            raise TypeError("request must be a DebugCodeRequest")

        if not isinstance(request.generated_code, str):
            raise TypeError("generated_code must be a string")
        if not request.generated_code.strip():
            raise ValueError("generated_code must be non-empty")

        logger.info("DebugCodeUseCase: start")
        report = self._debugger_agent.debug_code(
            request.generated_code,
            generated_tests=request.generated_tests,
            application_stdout=request.execution_stdout,
            application_stderr=request.execution_stderr,
            application_exit_code=request.execution_exit_code,
            test_stdout=request.test_stdout,
            test_stderr=request.test_stderr,
            test_exit_code=request.test_exit_code,
            reviewer_feedback=request.reviewer_feedback,
        )
        logger.info("DebugCodeUseCase: finished")
        return DebugCodeResult(report=report)

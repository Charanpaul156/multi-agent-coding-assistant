"""Application use-cases for code review.

Use-cases are framework-independent. Application layer must not depend on
FastAPI or Pydantic. Only dataclasses are used here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from agents.code_reviewer_agent import ReviewReport, ReviewerAgent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewCodeRequest:
    """Request DTO for reviewing generated code.

    The code itself is always required. Execution results (stdout, stderr,
    exit_code) are optional and enable reviewing code both before and after
    execution.

    Test-related fields are optional and used by the full workflow so the
    reviewer can also analyze the generated tests and test execution results.
    The standalone ``POST /review-code`` endpoint continues to work with only
    the application fields.
    """

    generated_code: str
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    exit_code: Optional[int] = None
    generated_tests: Optional[str] = None
    test_stdout: Optional[str] = None
    test_stderr: Optional[str] = None
    test_exit_code: Optional[int] = None


@dataclass(frozen=True)
class ReviewCodeResult:
    """Result DTO for a code review."""

    report: ReviewReport


class ReviewCodeUseCase:
    """Use-case: produce a structured review of generated Python code.

    The ReviewerAgent analyzes the code (and optional execution results) but
    never executes, generates, or modifies code.
    """

    def __init__(self, reviewer_agent: ReviewerAgent) -> None:
        self._reviewer_agent = reviewer_agent

    def execute(self, request: ReviewCodeRequest) -> ReviewCodeResult:
        """Execute the review use-case."""

        if not isinstance(request, ReviewCodeRequest):
            raise TypeError("request must be a ReviewCodeRequest")

        if not isinstance(request.generated_code, str):
            raise TypeError("generated_code must be a string")
        if not request.generated_code.strip():
            raise ValueError("generated_code must be non-empty")

        logger.info("ReviewCodeUseCase: start")
        report = self._reviewer_agent.review_code(
            request.generated_code,
            stdout=request.stdout,
            stderr=request.stderr,
            exit_code=request.exit_code,
            generated_tests=request.generated_tests,
            test_stdout=request.test_stdout,
            test_stderr=request.test_stderr,
            test_exit_code=request.test_exit_code,
        )
        logger.info("ReviewCodeUseCase: finished")
        return ReviewCodeResult(report=report)

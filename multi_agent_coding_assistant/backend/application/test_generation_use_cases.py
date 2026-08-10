"""Application use-cases for test generation.

Use-cases are framework-independent. Application layer must not depend on
FastAPI or Pydantic. Only dataclasses are used here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agents.test_generator_agent import (
    TestGenerationReport,
    TestGeneratorAgent,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerateTestsRequest:
    """Request DTO for generating a test suite.

    The generated Python code is always required.
    """

    generated_code: str


@dataclass(frozen=True)
class GenerateTestsResult:
    """Result DTO for a generated test suite."""

    report: TestGenerationReport


class GenerateTestsUseCase:
    """Use-case: produce a structured pytest test suite for generated code.

    The TestGeneratorAgent analyzes the code and generates tests only. It
    never executes tests, never executes application code, and never modifies
    the original source code.
    """

    def __init__(self, test_generator_agent: TestGeneratorAgent) -> None:
        self._test_generator_agent = test_generator_agent

    def execute(self, request: GenerateTestsRequest) -> GenerateTestsResult:
        """Execute the test-generation use-case."""

        if not isinstance(request, GenerateTestsRequest):
            raise TypeError("request must be a GenerateTestsRequest")

        if not isinstance(request.generated_code, str):
            raise TypeError("generated_code must be a string")
        if not request.generated_code.strip():
            raise ValueError("generated_code must be non-empty")

        logger.info("GenerateTestsUseCase: start")
        report = self._test_generator_agent.generate_tests(
            request.generated_code
        )
        logger.info("GenerateTestsUseCase: finished")
        return GenerateTestsResult(report=report)

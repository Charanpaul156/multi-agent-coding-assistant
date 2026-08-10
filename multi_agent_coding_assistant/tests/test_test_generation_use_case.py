"""Unit tests for GenerateTestsUseCase (application layer).

Uses a fake agent so no LLM/network is required.
"""

from __future__ import annotations

import pytest

from agents.test_generator_agent import TestGenerationReport, TestGeneratorAgentError
from backend.application.test_generation_use_cases import (
    GenerateTestsRequest,
    GenerateTestsResult,
    GenerateTestsUseCase,
)


class _FakeAgent:
    def __init__(self, report=None, exc=None):
        self._report = report
        self._exc = exc

    def generate_tests(self, code):
        if self._exc is not None:
            raise self._exc
        return self._report


def _make_report() -> TestGenerationReport:
    from agents.test_generator_agent import TestCaseInfo

    return TestGenerationReport(
        test_overview="Overview",
        test_framework="pytest",
        test_cases=[
            TestCaseInfo("t1", "d", "in", "expected"),
        ],
        edge_cases=[],
        negative_cases=[],
        coverage_suggestions=[],
        generated_test_code="def test_t1():\n    assert True\n",
        final_summary="Summary",
    )


def test_use_case_valid_request() -> None:
    report = _make_report()
    use_case = GenerateTestsUseCase(test_generator_agent=_FakeAgent(report=report))
    result = use_case.execute(GenerateTestsRequest(generated_code="def add():..."))

    assert isinstance(result, GenerateTestsResult)
    assert result.report is report
    assert result.report.test_framework == "pytest"


def test_use_case_empty_request() -> None:
    use_case = GenerateTestsUseCase(test_generator_agent=_FakeAgent())
    with pytest.raises(ValueError):
        use_case.execute(GenerateTestsRequest(generated_code="   "))


def test_use_case_non_string_request() -> None:
    use_case = GenerateTestsUseCase(test_generator_agent=_FakeAgent())
    with pytest.raises(TypeError):
        use_case.execute(GenerateTestsRequest(generated_code=123))  # type: ignore[arg-type]


def test_use_case_agent_failure_propagates() -> None:
    use_case = GenerateTestsUseCase(
        test_generator_agent=_FakeAgent(exc=TestGeneratorAgentError("bad"))
    )
    with pytest.raises(TestGeneratorAgentError):
        use_case.execute(GenerateTestsRequest(generated_code="def add():..."))


def test_use_case_wrong_request_type() -> None:
    use_case = GenerateTestsUseCase(test_generator_agent=_FakeAgent())
    with pytest.raises(TypeError):
        use_case.execute("not a request")  # type: ignore[arg-type]

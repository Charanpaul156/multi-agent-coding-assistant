"""Unit tests for GenerateDocumentationUseCase (application layer).

Uses a fake DocumentationAgent so no LLM/network or agents are executed.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, Optional

import pytest

from agents.documentation_agent import (
    DocumentationAgentError,
    DocumentationReport,
)
from backend.application.documentation_use_cases import (
    GenerateDocumentationRequest,
    GenerateDocumentationResult,
    GenerateDocumentationUseCase,
)


class _FakeDocumentationAgent:
    """Configurable fake DocumentationAgent for application layer tests."""

    def __init__(
        self,
        report: Optional[DocumentationReport] = None,
        exc: Optional[Exception] = None,
    ) -> None:
        self._report = report
        self._exc = exc
        self.calls = 0
        self.last_code: Optional[str] = None
        self.last_retrieved_context: Optional[str] = None

    def generate_documentation(
        self, code: str, *, retrieved_context: Optional[str] = None
    ) -> DocumentationReport:
        self.calls += 1
        self.last_code = code
        self.last_retrieved_context = retrieved_context
        if self._exc is not None:
            raise self._exc
        if self._report is None:
            raise AssertionError("No DocumentationReport configured on fake agent")
        return self._report


def _make_report() -> DocumentationReport:
    return DocumentationReport(
        summary="Test module summary",
        module_description="Detailed module purpose and overview.",
        function_docs=[
            {
                "name": "add",
                "signature": "def add(a: int, b: int) -> int",
                "description": "Returns the sum of two integers.",
                "parameters": ["a: First integer.", "b: Second integer."],
                "returns": "Sum of a and b.",
            }
        ],
        class_docs=[
            {
                "name": "Accumulator",
                "description": "Accumulates numerical values.",
                "methods": ["add(value)"],
            }
        ],
        usage_examples=["result = add(1, 2)"],
        markdown_documentation="# Module Docs\n\nFull documentation.",
    )


# 1. Valid request
def test_use_case_valid_request() -> None:
    report = _make_report()
    agent = _FakeDocumentationAgent(report=report)
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    code_snippet = "def add(a, b):\n    return a + b\n"
    request = GenerateDocumentationRequest(code=code_snippet)
    result = use_case.execute(request)

    assert isinstance(result, GenerateDocumentationResult)
    assert agent.calls == 1
    assert agent.last_code == code_snippet
    assert agent.last_retrieved_context is None


# 2. Returned DocumentationReport
def test_use_case_returned_documentation_report() -> None:
    expected_report = _make_report()
    agent = _FakeDocumentationAgent(report=expected_report)
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    request = GenerateDocumentationRequest(code="def add(a, b): return a + b")
    result = use_case.execute(request)

    assert result.report is expected_report
    assert result.report.summary == "Test module summary"
    assert result.report.module_description == "Detailed module purpose and overview."
    assert len(result.report.function_docs) == 1
    assert len(result.report.class_docs) == 1
    assert len(result.report.usage_examples) == 1
    assert result.report.markdown_documentation == "# Module Docs\n\nFull documentation."


# 3. retrieved_context forwarding
def test_use_case_retrieved_context_forwarding() -> None:
    report = _make_report()
    agent = _FakeDocumentationAgent(report=report)
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    context = "Retrieved doc: helper math functions."
    code_snippet = "def add(a, b): return a + b"
    request = GenerateDocumentationRequest(
        code=code_snippet,
        retrieved_context=context,
    )
    result = use_case.execute(request)

    assert isinstance(result, GenerateDocumentationResult)
    assert agent.calls == 1
    assert agent.last_code == code_snippet
    assert agent.last_retrieved_context == context


# 4. Invalid request type
@pytest.mark.parametrize("invalid_req", ["not a request", None, 123, {"code": "pass"}])
def test_use_case_invalid_request_type(invalid_req: Any) -> None:
    agent = _FakeDocumentationAgent(report=_make_report())
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="GenerateDocumentationRequest"):
        use_case.execute(invalid_req)  # type: ignore[arg-type]
    assert agent.calls == 0


# 5. Empty code & whitespace code
def test_use_case_empty_code() -> None:
    agent = _FakeDocumentationAgent(report=_make_report())
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        use_case.execute(GenerateDocumentationRequest(code=""))
    assert agent.calls == 0


def test_use_case_whitespace_code() -> None:
    agent = _FakeDocumentationAgent(report=_make_report())
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty"):
        use_case.execute(GenerateDocumentationRequest(code="   \t\n  "))
    assert agent.calls == 0


# 6. Invalid code type
@pytest.mark.parametrize("invalid_code", [None, 123, 45.6, ["def foo(): pass"], {"code": "def foo(): pass"}])
def test_use_case_invalid_code_type(invalid_code: Any) -> None:
    agent = _FakeDocumentationAgent(report=_make_report())
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="string"):
        use_case.execute(GenerateDocumentationRequest(code=invalid_code))  # type: ignore[arg-type]
    assert agent.calls == 0


# 7. Invalid context type
@pytest.mark.parametrize("invalid_context", [123, 45.6, ["context"], {"context": "text"}])
def test_use_case_invalid_context_type(invalid_context: Any) -> None:
    agent = _FakeDocumentationAgent(report=_make_report())
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="string"):
        use_case.execute(
            GenerateDocumentationRequest(
                code="def foo(): pass",
                retrieved_context=invalid_context,  # type: ignore[arg-type]
            )
        )
    assert agent.calls == 0


# 8. Agent exception propagation
def test_use_case_agent_exception_propagation() -> None:
    agent_error = DocumentationAgentError("LLM generation failed")
    agent = _FakeDocumentationAgent(exc=agent_error)
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    request = GenerateDocumentationRequest(code="def add(a, b): return a + b")
    with pytest.raises(DocumentationAgentError, match="LLM generation failed"):
        use_case.execute(request)
    assert agent.calls == 1


def test_use_case_generic_exception_propagation() -> None:
    agent = _FakeDocumentationAgent(exc=RuntimeError("Unexpected agent failure"))
    use_case = GenerateDocumentationUseCase(documentation_agent=agent)  # type: ignore[arg-type]

    request = GenerateDocumentationRequest(code="def add(a, b): return a + b")
    with pytest.raises(RuntimeError, match="Unexpected agent failure"):
        use_case.execute(request)
    assert agent.calls == 1


# Immutability tests
def test_dtos_are_frozen() -> None:
    req = GenerateDocumentationRequest(code="def foo(): pass")
    with pytest.raises(FrozenInstanceError):
        req.code = "new code"  # type: ignore[misc]

    res = GenerateDocumentationResult(report=_make_report())
    with pytest.raises(FrozenInstanceError):
        res.report = _make_report()  # type: ignore[misc]

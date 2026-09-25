"""Unit tests for DocumentationAgent.

These tests mock the LLMClient so no Gemini API key or network access is
required.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any, Dict, List, Optional

import pytest

from agents.documentation_agent import (
    DocumentationAgent,
    DocumentationAgentError,
    DocumentationReport,
)
from backend.infrastructure.llm_client import LLMResponse


def _valid_report_dict() -> Dict[str, Any]:
    return {
        "summary": "Mathematical utility module for arithmetic operations.",
        "module_description": (
            "This module provides functions and classes for performing safe arithmetic "
            "computations and managing calculator state."
        ),
        "function_docs": [
            {
                "name": "add",
                "signature": "def add(a: float, b: float) -> float",
                "description": "Returns the sum of two numbers.",
                "parameters": ["a: First number.", "b: Second number."],
                "returns": "Sum of a and b.",
            }
        ],
        "class_docs": [
            {
                "name": "Calculator",
                "description": "Stateful calculator maintaining running total.",
                "methods": ["add(value)", "reset()"],
            }
        ],
        "usage_examples": [
            "result = add(2, 3)\nassert result == 5",
            "calc = Calculator()\ncalc.add(10)",
        ],
        "markdown_documentation": (
            "# Math Utilities\n\n"
            "## Functions\n"
            "### `add(a, b)`\n"
            "Adds two numbers.\n"
        ),
    }


class _FakeLLMClient:
    """Configurable fake LLM client for deterministic testing."""

    def __init__(
        self,
        responses: Optional[List[str]] = None,
        exc: Optional[Exception] = None,
    ) -> None:
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0
        self.prompts: List[str] = []
        self.system_prompts: List[Optional[str]] = []

    def generate(
        self, prompt: str, *, system_prompt: Optional[str] = None
    ) -> LLMResponse:
        self.calls += 1
        self.prompts.append(prompt)
        self.system_prompts.append(system_prompt)
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No more LLM responses configured")
        return LLMResponse(text=self.responses.pop(0))


def _build_agent(client: _FakeLLMClient) -> DocumentationAgent:
    return DocumentationAgent(client)  # type: ignore[arg-type]


# 1. Valid documentation generation
def test_generate_documentation_valid() -> None:
    expected_data = _valid_report_dict()
    client = _FakeLLMClient(responses=[json.dumps(expected_data)])
    agent = _build_agent(client)

    sample_code = "def add(a, b):\n    return a + b\n"
    report = agent.generate_documentation(sample_code)

    assert isinstance(report, DocumentationReport)
    assert report.summary == expected_data["summary"]
    assert report.module_description == expected_data["module_description"]
    assert len(report.function_docs) == 1
    assert report.function_docs[0]["name"] == "add"
    assert len(report.class_docs) == 1
    assert report.class_docs[0]["name"] == "Calculator"
    assert len(report.usage_examples) == 2
    assert report.markdown_documentation == expected_data["markdown_documentation"]
    assert client.calls == 1
    assert "def add(a, b):" in client.prompts[0]
    assert client.system_prompts[0] is not None
    assert "Documentation" in client.system_prompts[0]


# 2. Empty input
def test_generate_documentation_empty_code() -> None:
    client = _FakeLLMClient()
    agent = _build_agent(client)

    with pytest.raises(ValueError, match="non-empty"):
        agent.generate_documentation("")
    assert client.calls == 0


# 3. Whitespace input
def test_generate_documentation_whitespace_code() -> None:
    client = _FakeLLMClient()
    agent = _build_agent(client)

    with pytest.raises(ValueError, match="non-empty"):
        agent.generate_documentation("   \n\t  ")
    assert client.calls == 0


# 4. Non-string input
@pytest.mark.parametrize("invalid_code", [None, 123, 45.6, ["code"], {"code": "pass"}])
def test_generate_documentation_non_string_code(invalid_code: Any) -> None:
    client = _FakeLLMClient()
    agent = _build_agent(client)

    with pytest.raises(TypeError, match="string"):
        agent.generate_documentation(invalid_code)  # type: ignore[arg-type]
    assert client.calls == 0


def test_generate_documentation_non_string_retrieved_context() -> None:
    client = _FakeLLMClient()
    agent = _build_agent(client)

    with pytest.raises(TypeError, match="string"):
        agent.generate_documentation(
            "def foo(): pass", retrieved_context=123  # type: ignore[arg-type]
        )
    assert client.calls == 0


# 5. Markdown-fenced JSON
def test_generate_documentation_with_markdown_json_fences() -> None:
    raw = f"```json\n{json.dumps(_valid_report_dict())}\n```"
    client = _FakeLLMClient(responses=[raw])
    agent = _build_agent(client)

    report = agent.generate_documentation("def add(a, b): return a + b")
    assert isinstance(report, DocumentationReport)
    assert report.summary == _valid_report_dict()["summary"]
    assert client.calls == 1


def test_generate_documentation_with_markdown_fences() -> None:
    raw = f"```markdown\n{json.dumps(_valid_report_dict())}\n```"
    client = _FakeLLMClient(responses=[raw])
    agent = _build_agent(client)

    report = agent.generate_documentation("def add(a, b): return a + b")
    assert isinstance(report, DocumentationReport)
    assert report.summary == _valid_report_dict()["summary"]
    assert client.calls == 1


def test_generate_documentation_with_generic_code_fences() -> None:
    raw = f"```\n{json.dumps(_valid_report_dict())}\n```"
    client = _FakeLLMClient(responses=[raw])
    agent = _build_agent(client)

    report = agent.generate_documentation("def add(a, b): return a + b")
    assert isinstance(report, DocumentationReport)
    assert report.summary == _valid_report_dict()["summary"]
    assert client.calls == 1


# 6. Malformed JSON
def test_generate_documentation_malformed_json_fails_after_retry() -> None:
    client = _FakeLLMClient(responses=["not json at all", "still not valid json"])
    agent = _build_agent(client)

    with pytest.raises(DocumentationAgentError, match="Invalid documentation response"):
        agent.generate_documentation("def add(a, b): return a + b")
    assert client.calls == 2


def test_generate_documentation_malformed_json_successful_retry() -> None:
    client = _FakeLLMClient(
        responses=["not valid json", json.dumps(_valid_report_dict())]
    )
    agent = _build_agent(client)

    report = agent.generate_documentation("def add(a, b): return a + b")
    assert isinstance(report, DocumentationReport)
    assert client.calls == 2


def test_generate_documentation_single_malformed_response_raises() -> None:
    client = _FakeLLMClient(responses=["random text without json"])
    agent = _build_agent(client)

    with pytest.raises(DocumentationAgentError):
        agent.generate_documentation("def add(a, b): return a + b")


# 7. LLM failure
def test_generate_documentation_llm_failure() -> None:
    client = _FakeLLMClient(exc=RuntimeError("LLM API network error"))
    agent = _build_agent(client)

    with pytest.raises(DocumentationAgentError, match="LLM failure"):
        agent.generate_documentation("def add(a, b): return a + b")


# Additional edge cases & contract tests
def test_generate_documentation_missing_required_fields() -> None:
    data = _valid_report_dict()
    del data["summary"]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(DocumentationAgentError, match="Missing required field.*summary"):
        agent.generate_documentation("def add(a, b): return a + b")


def test_generate_documentation_invalid_field_types() -> None:
    data = _valid_report_dict()
    data["usage_examples"] = "not a list"
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(DocumentationAgentError, match="Expected list"):
        agent.generate_documentation("def add(a, b): return a + b")


def test_generate_documentation_with_retrieved_context() -> None:
    expected_data = _valid_report_dict()
    client = _FakeLLMClient(responses=[json.dumps(expected_data)])
    agent = _build_agent(client)

    context = "Repository context: Math module used across backend."
    code = "def add(a, b): return a + b"
    report = agent.generate_documentation(code, retrieved_context=context)

    assert isinstance(report, DocumentationReport)
    assert client.calls == 1
    assert context in client.prompts[0]
    assert code in client.prompts[0]


def test_documentation_report_frozen() -> None:
    expected_data = _valid_report_dict()
    client = _FakeLLMClient(responses=[json.dumps(expected_data)])
    agent = _build_agent(client)

    report = agent.generate_documentation("def add(a, b): return a + b")
    with pytest.raises(FrozenInstanceError):
        report.summary = "New summary"  # type: ignore[misc]


def test_run_alias() -> None:
    expected_data = _valid_report_dict()
    client = _FakeLLMClient(responses=[json.dumps(expected_data)])
    agent = _build_agent(client)

    report = agent.run("def add(a, b): return a + b")
    assert isinstance(report, DocumentationReport)
    assert report.summary == expected_data["summary"]

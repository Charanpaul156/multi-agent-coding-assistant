"""Unit tests for the DebuggerAgent.

These tests mock the LLMClient so no Gemini API key or network access is
required.
"""

from __future__ import annotations

import json

import pytest

from agents.debugger_agent import (
    DebugReport,
    DebuggerAgent,
    DebuggerAgentError,
)
from backend.infrastructure.llm_client import LLMResponse


def _valid_report_json() -> dict:
    return {
        "issue_detected": True,
        "error_type": "ZeroDivisionError",
        "root_cause": "Division by zero in divide()",
        "affected_component": "divide function",
        "explanation": "The function divides without checking for zero divisor.",
        "suggested_changes": ["Add a zero-divisor guard."],
        "corrected_code": (
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('cannot divide by zero')\n"
            "    return a / b\n"
        ),
        "confidence": 0.9,
        "final_summary": "Fixed the divide-by-zero bug.",
    }


class _FakeLLMClient:
    """A configurable fake LLM client for deterministic tests."""

    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0

    def generate(self, prompt, *, system_prompt=None):
        self.calls += 1
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No more LLM responses configured")
        return LLMResponse(text=self.responses.pop(0))


def _build_agent(client) -> DebuggerAgent:
    # Duck-typed: agent only needs .generate(prompt, system_prompt=...)
    return DebuggerAgent(client)  # type: ignore[arg-type]


def test_debug_valid_runtime_error() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_report_json())])
    agent = _build_agent(client)

    report = agent.debug_code(
        "def divide(a, b):\n    return a / b\n",
        application_stderr="ZeroDivisionError: division by zero",
        application_exit_code=1,
    )

    assert isinstance(report, DebugReport)
    assert report.issue_detected is True
    assert report.error_type == "ZeroDivisionError"
    assert report.root_cause
    assert report.affected_component == "divide function"
    assert report.corrected_code
    assert report.confidence == 0.9
    assert client.calls == 1


def test_debug_with_test_failure() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_report_json())])
    agent = _build_agent(client)

    report = agent.debug_code(
        "def add(a, b):\n    return a - b\n",
        generated_tests="def test_add():\n    assert add(2, 3) == 5",
        test_stdout="1 failed",
        test_exit_code=1,
    )

    assert isinstance(report, DebugReport)
    assert report.issue_detected is True
    assert client.calls == 1


def test_debug_with_reviewer_feedback() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_report_json())])
    agent = _build_agent(client)

    report = agent.debug_code(
        "def f():\n    pass\n",
        reviewer_feedback="Critical logic issue detected.",
    )

    assert isinstance(report, DebugReport)
    assert client.calls == 1


def test_debug_empty_code() -> None:
    agent = _build_agent(_FakeLLMClient())
    with pytest.raises(ValueError):
        agent.debug_code("   ")


def test_debug_non_string_code() -> None:
    agent = _build_agent(_FakeLLMClient())
    with pytest.raises(TypeError):
        agent.debug_code(123)  # type: ignore[arg-type]


def test_debug_malformed_json_successful_retry() -> None:
    client = _FakeLLMClient(
        responses=["not json at all", json.dumps(_valid_report_json())]
    )
    agent = _build_agent(client)

    report = agent.debug_code("def add(a, b):\n    return a + b\n")
    assert isinstance(report, DebugReport)
    assert client.calls == 2


def test_debug_malformed_json_fails_after_retry() -> None:
    client = _FakeLLMClient(responses=["not json at all", "still not json"])
    agent = _build_agent(client)

    with pytest.raises(DebuggerAgentError):
        agent.debug_code("def add(a, b):\n    return a + b\n")
    assert client.calls == 2


def test_debug_missing_required_fields() -> None:
    data = _valid_report_json()
    del data["root_cause"]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(DebuggerAgentError):
        agent.debug_code("def add(a, b):\n    return a + b\n")


def test_debug_llm_failure() -> None:
    client = _FakeLLMClient(exc=RuntimeError("LLM down"))
    agent = _build_agent(client)

    with pytest.raises(DebuggerAgentError):
        agent.debug_code("def add(a, b):\n    return a + b\n")


def test_debug_with_markdown_fences() -> None:
    raw = "```json\n" + json.dumps(_valid_report_json()) + "\n```"
    client = _FakeLLMClient(responses=[raw])
    agent = _build_agent(client)

    report = agent.debug_code("def add(a, b):\n    return a + b\n")
    assert isinstance(report, DebugReport)
    assert client.calls == 1


def test_debug_invalid_confidence() -> None:
    data = _valid_report_json()
    data["confidence"] = 5.0
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(DebuggerAgentError):
        agent.debug_code("def add(a, b):\n    return a + b\n")

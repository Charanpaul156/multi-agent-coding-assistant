"""Unit tests for the ReviewerAgent.

These tests mock the LLMClient so no Gemini API key or network access is
required.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from agents.code_reviewer_agent import (
    ReviewReport,
    ReviewerAgent,
    ReviewerAgentError,
)
from backend.infrastructure.llm_client import LLMResponse


def _valid_review_dict() -> Dict[str, Any]:
    return {
        "overall_score": 98,
        "strengths": ["Clean code", "Proper docstring"],
        "weaknesses": ["None"],
        "pep8_issues": [],
        "performance_suggestions": [],
        "security_concerns": [],
        "logic_issues": [],
        "maintainability": ["High"],
        "error_handling": ["Good"],
        "recommendations": ["Add type hints"],
        "final_summary": "Great implementation.",
    }


class _FakeLLMClient:
    """Configurable fake LLM client for deterministic testing."""

    def __init__(self, responses=None, exc=None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(self, prompt, *, system_prompt=None):
        self.calls += 1
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        if self.exc is not None:
            raise self.exc
        if not self.responses:
            raise AssertionError("No more LLM responses configured")
        return LLMResponse(text=self.responses.pop(0))


def _build_agent(client) -> ReviewerAgent:
    return ReviewerAgent(client)  # type: ignore[arg-type]


def test_review_code_normalizes_none_in_logic_issues_to_empty() -> None:
    data = _valid_review_dict()
    data["logic_issues"] = [
        "None. No issues found.",
        "None. The function correctly implements multiplication.",
    ]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.review_code("def multiply(a, b): return a * b")

    assert isinstance(report, ReviewReport)
    assert report.logic_issues == []


def test_review_code_normalizes_none_in_security_concerns_to_empty() -> None:
    data = _valid_review_dict()
    data["security_concerns"] = [
        "None. No security concerns.",
        "None. The function is a pure mathematical operation.",
    ]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.review_code("def multiply(a, b): return a * b")

    assert isinstance(report, ReviewReport)
    assert report.security_concerns == []


def test_review_code_normalizes_no_issues_and_na() -> None:
    data = _valid_review_dict()
    data["logic_issues"] = ["No issues", "N/A", "no logic issues"]
    data["security_concerns"] = ["No security concerns", "not applicable", "None."]
    data["pep8_issues"] = ["No PEP8 issues", "None observed"]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.review_code("def multiply(a, b): return a * b")

    assert report.logic_issues == []
    assert report.security_concerns == []
    assert report.pep8_issues == []


def test_review_code_preserves_genuine_issues() -> None:
    data = _valid_review_dict()
    data["logic_issues"] = [
        "Division by zero is not handled.",
        "Does not handle empty list input.",
        "No input validation for negative numbers.",
    ]
    data["security_concerns"] = [
        "User input is passed directly to subprocess.",
        "Hardcoded credential detected.",
    ]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.review_code("def divide(a, b): return a / b")

    assert len(report.logic_issues) == 3
    assert "Division by zero is not handled." in report.logic_issues
    assert "Does not handle empty list input." in report.logic_issues
    assert "No input validation for negative numbers." in report.logic_issues
    assert len(report.security_concerns) == 2
    assert "User input is passed directly to subprocess." in report.security_concerns


def test_review_code_invalid_list_types_raise_error() -> None:
    data = _valid_review_dict()
    data["logic_issues"] = "not a list"
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(ReviewerAgentError, match="Expected list"):
        agent.review_code("def add(a, b): return a + b")

    data2 = _valid_review_dict()
    data2["security_concerns"] = [123, 456]
    client2 = _FakeLLMClient(responses=[json.dumps(data2)])
    agent2 = _build_agent(client2)

    with pytest.raises(ReviewerAgentError, match="Expected list"):
        agent2.review_code("def add(a, b): return a + b")


def test_reviewer_prompt_instructs_empty_array_for_no_issues() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_review_dict())])
    agent = _build_agent(client)

    agent.review_code("def multiply(a, b): return a * b")

    assert client.last_system_prompt is not None
    assert "empty array []" in client.last_system_prompt
    assert "None" in client.last_system_prompt
    assert "No issues" in client.last_system_prompt

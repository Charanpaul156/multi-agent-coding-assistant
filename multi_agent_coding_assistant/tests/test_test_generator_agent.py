"""Unit tests for the TestGeneratorAgent.

These tests mock the LLMClient so no Gemini API key or network access is
required.
"""

from __future__ import annotations

import json

import pytest

from agents.test_generator_agent import (
    TestCaseInfo,
    TestGenerationReport,
    TestGeneratorAgent,
    TestGeneratorAgentError,
)
from backend.infrastructure.llm_client import LLMClient, LLMResponse


def _valid_report_json() -> dict:
    return {
        "test_overview": "Tests for the add function.",
        "test_framework": "pytest",
        "test_cases": [
            {
                "name": "test_add_positives",
                "description": "Adds two positive numbers.",
                "input_example": "add(2, 3)",
                "expected_behavior": "returns 5",
            }
        ],
        "edge_cases": [
            {
                "name": "test_add_zero",
                "description": "Adding zero.",
                "input_example": "add(0, 5)",
                "expected_behavior": "returns 5",
            }
        ],
        "negative_cases": [
            {
                "name": "test_add_negatives",
                "description": "Adding negative numbers.",
                "input_example": "add(-1, -2)",
                "expected_behavior": "returns -3",
            }
        ],
        "coverage_suggestions": ["Add boundary tests."],
        "generated_test_code": (
            "def test_add_positives():\n"
            "    assert add(2, 3) == 5\n\n"
            "def test_add_zero():\n"
            "    assert add(0, 5) == 5\n\n"
            "def test_add_negatives():\n"
            "    assert add(-1, -2) == -3\n"
        ),
        "final_summary": "Test suite covers core behavior.",
    }


class _FakeLLMClient:
    """A configurable fake LLM client for deterministic tests."""

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


def _build_agent(client) -> TestGeneratorAgent:
    # Duck-typed: agent only needs .generate(prompt, system_prompt=...)
    return TestGeneratorAgent(client)  # type: ignore[arg-type]


def test_generate_tests_valid_source_code() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_report_json())])
    agent = _build_agent(client)

    report = agent.generate_tests("def add(a, b):\n    return a + b\n")

    assert isinstance(report, TestGenerationReport)
    assert report.test_framework == "pytest"
    assert report.generated_test_code
    assert "def test_add_positives" in report.generated_test_code
    assert report.test_cases[0].name == "test_add_positives"
    assert isinstance(report.test_cases[0], TestCaseInfo)
    assert client.calls == 1


def test_generate_tests_empty_source_code() -> None:
    agent = _build_agent(_FakeLLMClient())
    with pytest.raises(ValueError):
        agent.generate_tests("   ")


def test_generate_tests_non_string_source_code() -> None:
    agent = _build_agent(_FakeLLMClient())
    with pytest.raises(TypeError):
        agent.generate_tests(123)  # type: ignore[arg-type]


def test_generate_tests_malformed_json_fails_after_retry() -> None:
    # First response is malformed, retry is also malformed -> error.
    client = _FakeLLMClient(responses=["not json at all", "still not json"])
    agent = _build_agent(client)

    with pytest.raises(TestGeneratorAgentError):
        agent.generate_tests("def add(a, b):\n    return a + b\n")
    assert client.calls == 2


def test_generate_tests_malformed_json_successful_retry() -> None:
    # First response malformed, retry valid -> success.
    client = _FakeLLMClient(
        responses=["not json at all", json.dumps(_valid_report_json())]
    )
    agent = _build_agent(client)

    report = agent.generate_tests("def add(a, b):\n    return a + b\n")
    assert isinstance(report, TestGenerationReport)
    assert client.calls == 2


def test_generate_tests_missing_required_fields() -> None:
    data = _valid_report_json()
    del data["generated_test_code"]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    with pytest.raises(TestGeneratorAgentError):
        agent.generate_tests("def add(a, b):\n    return a + b\n")


def test_generate_tests_missing_final_summary_falls_back_to_overview() -> None:
    data = _valid_report_json()
    del data["final_summary"]
    client = _FakeLLMClient(responses=[json.dumps(data)])
    agent = _build_agent(client)

    report = agent.generate_tests("def add(a, b):\n    return a + b\n")

    assert isinstance(report, TestGenerationReport)
    assert report.final_summary == report.test_overview


def test_generate_tests_llm_failure() -> None:
    client = _FakeLLMClient(exc=RuntimeError("LLM down"))
    agent = _build_agent(client)

    with pytest.raises(TestGeneratorAgentError):
        agent.generate_tests("def add(a, b):\n    return a + b\n")


def test_generate_tests_with_markdown_fences() -> None:
    raw = "```json\n" + json.dumps(_valid_report_json()) + "\n```"
    client = _FakeLLMClient(responses=[raw])
    agent = _build_agent(client)

    report = agent.generate_tests("def add(a, b):\n    return a + b\n")
    assert isinstance(report, TestGenerationReport)
    assert client.calls == 1


def test_generate_tests_prompt_contains_generated_code_import_guidance() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_report_json())])
    agent = _build_agent(client)

    agent.generate_tests("def multiply(a, b):\n    return a * b\n")

    assert client.last_system_prompt is not None
    # Must instruct that the code is in module 'generated_code'
    assert "generated_code" in client.last_system_prompt
    # Must instruct importing symbols from generated_code
    assert "from generated_code import" in client.last_system_prompt
    # Must warn not to assume shared namespace
    assert "same namespace" in client.last_system_prompt


def test_generate_tests_prompt_contains_pytest_approx_guidance() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_report_json())])
    agent = _build_agent(client)

    agent.generate_tests("def multiply(a, b):\n    return a * b\n")

    assert client.last_system_prompt is not None
    # Must instruct using pytest.approx for floating-point calculations
    assert "pytest.approx" in client.last_system_prompt
    assert "floating-point" in client.last_system_prompt


def test_generate_tests_prompt_contains_nan_and_special_floating_point_guidance() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_report_json())])
    agent = _build_agent(client)

    agent.generate_tests("def multiply(a, b):\n    return a * b\n")

    assert client.last_system_prompt is not None
    # Must instruct that NaN must not be compared using equality
    assert "Never compare NaN values using `==`" in client.last_system_prompt
    assert 'value == float("nan")' in client.last_system_prompt
    # Must instruct using math.isnan
    assert "math.isnan" in client.last_system_prompt
    # Must instruct correct IEEE-754 floating-point semantics
    assert "IEEE-754" in client.last_system_prompt


def test_generate_tests_prompt_contains_avoid_speculative_requirements_guidance() -> None:
    client = _FakeLLMClient(responses=[json.dumps(_valid_report_json())])
    agent = _build_agent(client)

    agent.generate_tests("def multiply(a, b):\n    return a * b\n")

    assert client.last_system_prompt is not None
    # Must instruct avoiding speculative or unsupported requirements
    assert "Avoid Speculative Test Requirements" in client.last_system_prompt
    assert "speculative" in client.last_system_prompt
    # Must instruct not assuming validation/error-handling behavior unless requested or established
    assert "Do not assume validation or error-handling behavior" in client.last_system_prompt
    assert "explicitly requested" in client.last_system_prompt


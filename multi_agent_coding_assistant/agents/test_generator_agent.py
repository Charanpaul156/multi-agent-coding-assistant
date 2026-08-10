"""Test Generator Agent.

This agent is responsible for analyzing generated Python source code and
producing a structured, pytest-oriented test suite.

RESPONSIBILITIES:
- Analyze generated Python code.
- Understand functions/classes and expected behavior.
- Generate pytest-compatible tests.
- Generate normal test cases.
- Generate edge cases.
- Generate negative/error cases where applicable.
- Consider boundary conditions.
- Identify important scenarios that should be tested.

STRICT CONSTRAINTS (must never be violated):
- NEVER execute tests.
- NEVER execute application code.
- NEVER modify the original generated code.
- NEVER generate arbitrary application code outside the test suite.

IMPORTANT: This class must not contain provider-specific logic (e.g. Gemini).
It communicates ONLY through the dependency-injected `LLMClient`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List

from backend.infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class TestGeneratorAgentError(RuntimeError):
    """Base error for test generator agent failures."""

    __test__ = False


@dataclass(frozen=True)
class TestCaseInfo:
    """Structured description of a single test scenario.

    Each scenario captures the name, a human-readable description, an example
    of the input, and the expected behavior verified by the test.
    """

    __test__ = False

    name: str
    description: str
    input_example: str
    expected_behavior: str


@dataclass(frozen=True)
class TestGenerationReport:
    """Structured test suite produced by the TestGeneratorAgent.

    The report separates the various categories of tests and always includes
    ready-to-use pytest code in ``generated_test_code``.
    """

    __test__ = False

    test_overview: str
    test_framework: str
    test_cases: List[TestCaseInfo]
    edge_cases: List[TestCaseInfo]
    negative_cases: List[TestCaseInfo]
    coverage_suggestions: List[str]
    generated_test_code: str
    final_summary: str


class TestGeneratorAgent:
    """Produce a structured TestGenerationReport from generated Python code.

    The agent only analyzes source code and generates tests. It never
    executes tests, never executes application code, never modifies the
    original source, and never generates unrelated application code.
    """

    __test__ = False

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def generate_tests(self, code: str) -> TestGenerationReport:
        """Generate a pytest-oriented test suite for the supplied source code.

        Args:
            code: The generated Python code to analyze (required).

        Returns:
            A structured TestGenerationReport.

        Raises:
            TypeError: If ``code`` is not a string.
            ValueError: If ``code`` is empty.
            TestGeneratorAgentError: If the LLM returns invalid/irrecoverable
                JSON or the test-generation response is invalid.
        """

        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if not code.strip():
            raise ValueError("code must be non-empty")

        system_prompt = (
            "You are a Senior Python Test Engineer.\n"
            "Analyze the supplied source code.\n"
            "Generate tests only.\n"
            "Do not modify the original source code.\n"
            "Never execute tests. Never execute application code.\n"
            "Never generate arbitrary application code outside the test suite.\n"
            "Your only responsibility is analyzing the source code and "
            "generating a pytest-compatible test suite.\n"
            "Return ONLY structured test information as valid JSON.\n"
            "The JSON MUST match the following schema:\n"
            "{\n"
            "  \"test_overview\": string,\n"
            "  \"test_framework\": string,\n"
            "  \"test_cases\": [\n"
            "    {\"name\": string, \"description\": string, "
            "\"input_example\": string, \"expected_behavior\": string}\n"
            "  ],\n"
            "  \"edge_cases\": [\n"
            "    {\"name\": string, \"description\": string, "
            "\"input_example\": string, \"expected_behavior\": string}\n"
            "  ],\n"
            "  \"negative_cases\": [\n"
            "    {\"name\": string, \"description\": string, "
            "\"input_example\": string, \"expected_behavior\": string}\n"
            "  ],\n"
            "  \"coverage_suggestions\": string[],\n"
            "  \"generated_test_code\": string,\n"
            "  \"final_summary\": string\n"
            "}\n"
            "generated_test_code must contain ONLY pytest-oriented test code "
            "with no Markdown fences. Include normal cases, edge cases, and "
            "negative/error cases where applicable.\n"
            "Do not include markdown. Do not include explanations outside JSON."
        )

        user_prompt = "Analyze the following Python code and generate tests:\n\n"
        user_prompt += "```python\n" + code + "\n```\n"

        logger.info("TestGeneratorAgent: generating tests")
        raw_text = self._call_llm(user_prompt, system_prompt)

        try:
            data = self._parse_json(raw_text)
            return self._validate_and_build_report(data)
        except TestGeneratorAgentError as exc:
            # Only malformed JSON should trigger a single correction retry.
            # Validation failures (e.g. missing required fields) and LLM
            # failures must fail fast and never silently return partial data.
            if "malformed" not in str(exc).lower():
                raise
            logger.warning(
                "TestGeneratorAgent: failed first parse (%s); retrying once", exc
            )
            correction_prompt = (
                "The previous response was not valid JSON matching the required "
                "schema. Return ONLY corrected valid JSON matching the schema. "
                "Do not add explanations."
            )
            raw_text_retry = self._call_llm(correction_prompt, system_prompt)
            try:
                data_retry = self._parse_json(raw_text_retry)
                return self._validate_and_build_report(data_retry)
            except Exception as exc2:
                logger.exception("TestGeneratorAgent: retry failed")
                raise TestGeneratorAgentError(
                    "Invalid test-generation response from LLM"
                ) from exc2

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        try:
            response = self._llm_client.generate(
                prompt, system_prompt=system_prompt
            )
            return (response.text or "").strip()
        except TestGeneratorAgentError:
            raise
        except Exception as exc:
            raise TestGeneratorAgentError(
                f"LLM failure during test generation: {exc}"
            ) from exc

    def _parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = _strip_markdown_code_fences(text).strip()

        # Attempt direct JSON parse.
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Last resort: extract the first JSON object.
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed

        raise TestGeneratorAgentError("Malformed JSON from LLM")

    def _validate_and_build_report(
        self, data: Dict[str, Any]
    ) -> TestGenerationReport:
        required_fields = [
            "test_overview",
            "test_framework",
            "test_cases",
            "edge_cases",
            "negative_cases",
            "coverage_suggestions",
            "generated_test_code",
            "final_summary",
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise TestGeneratorAgentError(
                    f"Missing field in test-generation JSON: {field_name}"
                )

        def as_str(value: Any) -> str:
            if not isinstance(value, str):
                raise TestGeneratorAgentError(
                    "Expected string fields in test-generation report"
                )
            return value

        def as_str_list(value: Any) -> List[str]:
            if not isinstance(value, list) or not all(
                isinstance(x, str) for x in value
            ):
                raise TestGeneratorAgentError(
                    "Expected list[string] fields in test-generation report"
                )
            return value

        def as_test_case_list(value: Any) -> List[TestCaseInfo]:
            if not isinstance(value, list):
                raise TestGeneratorAgentError(
                    "Expected list[object] fields in test-generation report"
                )
            cases: List[TestCaseInfo] = []
            for item in value:
                if not isinstance(item, dict):
                    raise TestGeneratorAgentError(
                        "Expected each test case to be an object"
                    )
                for key in ("name", "description", "input_example", "expected_behavior"):
                    if key not in item:
                        raise TestGeneratorAgentError(
                            f"Missing field in test case JSON: {key}"
                        )
                    if not isinstance(item[key], str):
                        raise TestGeneratorAgentError(
                            "Expected string values inside test cases"
                        )
                cases.append(
                    TestCaseInfo(
                        name=item["name"],
                        description=item["description"],
                        input_example=item["input_example"],
                        expected_behavior=item["expected_behavior"],
                    )
                )
            return cases

        generated_test_code = as_str(data["generated_test_code"])
        if (
            "def test_" not in generated_test_code
            and "import pytest" not in generated_test_code
            and "def test" not in generated_test_code
        ):
            # Heuristic: the generated code should look like pytest tests.
            # Keep it lenient but reject clearly non-test content.
            stripped = generated_test_code.strip()
            if (
                not stripped
                or stripped.startswith("def ")
                and "test" not in stripped.lower()
            ):
                raise TestGeneratorAgentError(
                    "generated_test_code does not appear to be pytest test code"
                )

        return TestGenerationReport(
            test_overview=as_str(data["test_overview"]),
            test_framework=as_str(data["test_framework"]),
            test_cases=as_test_case_list(data["test_cases"]),
            edge_cases=as_test_case_list(data["edge_cases"]),
            negative_cases=as_test_case_list(data["negative_cases"]),
            coverage_suggestions=as_str_list(data["coverage_suggestions"]),
            generated_test_code=generated_test_code,
            final_summary=as_str(data["final_summary"]),
        )


def _strip_markdown_code_fences(text: str) -> str:
    """Remove leading/trailing triple-backtick fences if present."""

    stripped = text.strip()

    fence_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped, flags=re.I)
    if fence_match:
        return fence_match.group(1)

    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped

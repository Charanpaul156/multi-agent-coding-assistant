"""Test Generator Agent.

This agent is responsible for analyzing generated Python source code and
producing a structured, pytest-oriented test suite with unit and integration tests.

RESPONSIBILITIES:
- Analyze generated Python code or repository files.
- Understand functions/classes and expected behavior.
- Generate pytest-compatible unit and integration tests.
- Cover edge cases, negative/error cases, and boundary conditions.
- Identify missing test scenarios and provide coverage suggestions.
- Avoid redundant or duplicate test cases.

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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class TestGeneratorAgentError(RuntimeError):
    """Base error for test generator agent failures."""

    __test__ = False


@dataclass(frozen=True)
class TestCaseInfo:
    """Structured description of a single test scenario."""

    __test__ = False

    name: str
    description: str
    input_example: str
    expected_behavior: str


@dataclass(frozen=True)
class TestGenerationReport:
    """Structured test suite produced by the TestGeneratorAgent."""

    __test__ = False

    test_overview: str
    test_framework: str
    test_cases: List[TestCaseInfo]
    edge_cases: List[TestCaseInfo]
    negative_cases: List[TestCaseInfo]
    coverage_suggestions: List[str]
    generated_test_code: str
    final_summary: str
    integration_cases: List[TestCaseInfo] = field(default_factory=list)
    missing_scenarios: List[str] = field(default_factory=list)


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
        """Generate a pytest-oriented test suite for the supplied source code."""
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
            "Generate unit tests, integration tests, boundary conditions, and edge cases.\n"
            "Identify missing test scenarios while avoiding redundant tests.\n"
            "RULES FOR GENERATED TESTS:\n"
            "1. The code under test is available from the Python module: generated_code\n"
            "2. Generated tests must explicitly import the required functions/classes "
            "from generated_code (for example: from generated_code import multiply). "
            "Do not assume test code shares the same namespace as the application code.\n"
            "3. Tests must validate the actual public behavior of the generated code.\n"
            "4. Use pytest where appropriate for test structure and assertions.\n"
            "5. For floating-point calculations, generated tests MUST avoid strict exact "
            "equality (==) when numerical precision can vary. Use pytest.approx(expected) "
            "instead (for example: assert multiply(1e-10, 1e-10) == pytest.approx(1e-20) "
            "instead of assert multiply(1e-10, 1e-10) == 1e-20).\n"
            "6. Special Floating-Point Values and NaN Semantics:\n"
            "   - Follow correct IEEE-754 and Python floating-point semantics.\n"
            "   - Never compare NaN values using `==` or `!=` as a correctness assertion.\n"
            "   - Do not write assertions such as `value == float(\"nan\")` or `value == float(\"NaN\")` "
            "because NaN does not equal itself in Python (float('nan') == float('nan') is always False).\n"
            "   - When testing whether a result is NaN, import `math` and use `math.isnan(value)`.\n"
            "   - When testing infinity, use appropriate direct infinity checks (such as `math.isinf(value)` "
            "or direct comparison with `float('inf')`), but do not confuse infinity with NaN.\n"
            "7. Avoid Speculative Test Requirements:\n"
            "   - Primarily test behavior explicitly requested by the user or established by the code contract.\n"
            "   - Cover reasonable edge cases directly implied by the requirements (such as positive/negative numbers, "
            "zero, integer/float combinations, reasonable floating-point precision).\n"
            "   - Avoid inventing unsupported or speculative requirements or behaviors.\n"
            "   - Do not assume validation or error-handling behavior (e.g. requiring strings, lists, or dicts to raise "
            "TypeError, or None handling, or NaN/Infinity handling) unless:\n"
            "     a) the user explicitly requested it,\n"
            "     b) the source code under test clearly and deliberately implements that specific validation/error-handling, or\n"
            "     c) standard Python language semantics inherently guarantee it for the operation.\n"
            "   - Tests should fail only when the code violates actual specified behavior or basic mathematical/logical "
            "correctness, not because the tests invented requirements the author never intended to support.\n"
            "Return ONLY structured test suite information as valid JSON.\n"
            "The JSON MUST match the following schema:\n"
            "{\n"
            "  \"test_overview\": string,\n"
            "  \"test_framework\": \"pytest\",\n"
            "  \"test_cases\": [\n"
            "    {\n"
            "      \"name\": string,\n"
            "      \"description\": string,\n"
            "      \"input_example\": string,\n"
            "      \"expected_behavior\": string\n"
            "    }\n"
            "  ],\n"
            "  \"edge_cases\": [\n"
            "    {\n"
            "      \"name\": string,\n"
            "      \"description\": string,\n"
            "      \"input_example\": string,\n"
            "      \"expected_behavior\": string\n"
            "    }\n"
            "  ],\n"
            "  \"negative_cases\": [\n"
            "    {\n"
            "      \"name\": string,\n"
            "      \"description\": string,\n"
            "      \"input_example\": string,\n"
            "      \"expected_behavior\": string\n"
            "    }\n"
            "  ],\n"
            "  \"integration_cases\": [\n"
            "    {\n"
            "      \"name\": string,\n"
            "      \"description\": string,\n"
            "      \"input_example\": string,\n"
            "      \"expected_behavior\": string\n"
            "    }\n"
            "  ],\n"
            "  \"missing_scenarios\": string[],\n"
            "  \"coverage_suggestions\": string[],\n"
            "  \"generated_test_code\": string,\n"
            "  \"final_summary\": string\n"
            "}\n"
            "Do not include markdown outside JSON. Do not include explanations outside JSON."
        )

        user_prompt = "Generate a pytest-oriented test suite for the following Python code:\n\n"
        user_prompt += "```python\n" + code + "\n```\n"

        logger.info("TestGeneratorAgent: generating tests")
        raw_text = self._call_llm(user_prompt, system_prompt)

        try:
            data = self._parse_json(raw_text)
            return self._validate_and_build_report(data)
        except TestGeneratorAgentError as exc:
            if "malformed" not in str(exc).lower():
                raise
            logger.warning("TestGeneratorAgent: failed first parse (%s); retrying once", exc)
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
                raise TestGeneratorAgentError("Invalid test-generation response from LLM") from exc2

    def generate_repository_tests(
        self, files: Dict[str, str], target_module: Optional[str] = None
    ) -> TestGenerationReport:
        """Generate integration and unit tests for multiple repository files."""
        if not files:
            raise ValueError("files mapping must not be empty")

        context_lines: List[str] = []
        for path, content in files.items():
            context_lines.append(f"File: {path}\n```\n{content}\n```")
        combined_context = "\n\n".join(context_lines)

        target_str = f" Target module: {target_module}" if target_module else ""
        prompt = f"Generate integration and unit tests for the following repository components.{target_str}\n\n{combined_context}"
        return self.generate_tests(prompt)

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        try:
            response = self._llm_client.generate(prompt, system_prompt=system_prompt)
            return (response.text or "").strip()
        except TestGeneratorAgentError:
            raise
        except Exception as exc:
            raise TestGeneratorAgentError(f"LLM failure during test generation: {exc}") from exc

    def _parse_json(self, text: str) -> Dict[str, Any]:
        cleaned = _strip_markdown_code_fences(text).strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed

        raise TestGeneratorAgentError("Malformed JSON from LLM")

    def _validate_and_build_report(self, data: Dict[str, Any]) -> TestGenerationReport:
        required_fields = [
            "test_overview",
            "test_framework",
            "test_cases",
            "edge_cases",
            "negative_cases",
            "coverage_suggestions",
            "generated_test_code",
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise TestGeneratorAgentError(f"Missing field in test-generation JSON: {field_name}")

        def as_str(value: Any) -> str:
            if not isinstance(value, str):
                raise TestGeneratorAgentError("Expected string fields in test-generation report")
            return value

        def as_str_list(value: Any) -> List[str]:
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise TestGeneratorAgentError("Expected list[string] fields in test-generation report")
            return value

        def as_test_case_list(value: Any) -> List[TestCaseInfo]:
            if not isinstance(value, list):
                raise TestGeneratorAgentError("Expected list[object] fields in test-generation report")
            cases: List[TestCaseInfo] = []
            for item in value:
                if not isinstance(item, dict):
                    raise TestGeneratorAgentError("Expected each test case to be an object")
                for key in ("name", "description", "input_example", "expected_behavior"):
                    if key not in item:
                        raise TestGeneratorAgentError(f"Missing field in test case JSON: {key}")
                    if not isinstance(item[key], str):
                        raise TestGeneratorAgentError("Expected string values inside test cases")
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
            stripped = generated_test_code.strip()
            if not stripped or (stripped.startswith("def ") and "test" not in stripped.lower()):
                raise TestGeneratorAgentError(
                    "generated_test_code does not appear to be pytest test code"
                )

        final_summary = as_str(
            data.get("final_summary") or data.get("test_overview") or ""
        )

        integration_cases = (
            as_test_case_list(data["integration_cases"])
            if "integration_cases" in data and isinstance(data["integration_cases"], list)
            else []
        )
        missing_scenarios = (
            as_str_list(data["missing_scenarios"])
            if "missing_scenarios" in data and isinstance(data["missing_scenarios"], list)
            else []
        )

        return TestGenerationReport(
            test_overview=as_str(data["test_overview"]),
            test_framework=as_str(data["test_framework"]),
            test_cases=as_test_case_list(data["test_cases"]),
            edge_cases=as_test_case_list(data["edge_cases"]),
            negative_cases=as_test_case_list(data["negative_cases"]),
            coverage_suggestions=as_str_list(data["coverage_suggestions"]),
            generated_test_code=generated_test_code,
            final_summary=final_summary,
            integration_cases=integration_cases,
            missing_scenarios=missing_scenarios,
        )


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped, flags=re.I)
    if fence_match:
        return fence_match.group(1)

    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped

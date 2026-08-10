"""Code Reviewer Agent.

This agent is responsible for reviewing generated Python code the same way a
Senior Software Engineer would review a GitHub Pull Request.

RESPONSIBILITIES:
- Analyze generated Python code.
- Analyze optional execution results (stdout, stderr, exit code).
- Provide structured, categorized feedback.

STRICT CONSTRAINTS (must never be violated):
- NEVER execute code.
- NEVER generate code.
- NEVER modify code.
- It is a read-only auditor.

IMPORTANT: This class must not contain provider-specific logic.
It communicates ONLY through the dependency-injected `LLMClient`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


from backend.infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class ReviewerAgentError(RuntimeError):
    """Base error for reviewer agent failures."""


@dataclass(frozen=True)
class ReviewReport:
    """Structured review report produced by the ReviewerAgent.

    Represents the output of a professional code review. Each field captures
    a distinct review category to keep the report structured (no free-form
    text dumping).
    """

    overall_score: int
    strengths: List[str]
    weaknesses: List[str]
    pep8_issues: List[str]
    performance_suggestions: List[str]
    security_concerns: List[str]
    logic_issues: List[str]
    maintainability: List[str]
    error_handling: List[str]
    recommendations: List[str]
    final_summary: str


class ReviewerAgent:
    """Produce a structured ReviewReport from generated code (and optional
    execution results) using an injected LLM client.

    The agent never executes, generates, or modifies code. It only analyzes
    and reports, behaving like a Senior Engineer performing a PR review.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def review_code(
        self,
        code: str,
        *,
        stdout: Optional[str] = None,
        stderr: Optional[str] = None,
        exit_code: Optional[int] = None,
        generated_tests: Optional[str] = None,
        test_stdout: Optional[str] = None,
        test_stderr: Optional[str] = None,
        test_exit_code: Optional[int] = None,
    ) -> ReviewReport:
        """Review generated Python code.

        Args:
            code: The generated Python code to review (required).
            stdout: Optional captured stdout from application execution.
            stderr: Optional captured stderr from application execution.
            exit_code: Optional exit code from application execution.
            generated_tests: Optional generated pytest test code to review.
            test_stdout: Optional captured stdout from test execution.
            test_stderr: Optional captured stderr from test execution.
            test_exit_code: Optional exit code from test execution.

        Returns:
            A structured ReviewReport.

        Raises:
            TypeError: If ``code`` is not a string.
            ValueError: If ``code`` is empty.
            ReviewerAgentError: If the LLM returns invalid/irrecoverable JSON.
        """

        if not isinstance(code, str):
            raise TypeError("code must be a string")
        if not code.strip():
            raise ValueError("code must be non-empty")

        system_prompt = (
            "You are a Senior Software Engineer performing a professional "
            "GitHub Pull Request review.\n"
            "Do not generate source code.\n"
            "Do not rewrite the implementation.\n"
            "Only review and provide structured feedback.\n"
            "Never execute code. Never modify code. Only analyze.\n"
            "Return ONLY structured review information as valid JSON.\n"
            "The JSON MUST match the following schema:\n"
            "{\n"
            "  \"overall_score\": integer (0 to 100),\n"
            "  \"strengths\": string[],\n"
            "  \"weaknesses\": string[],\n"
            "  \"pep8_issues\": string[],\n"
            "  \"performance_suggestions\": string[],\n"
            "  \"security_concerns\": string[],\n"
            "  \"logic_issues\": string[],\n"
            "  \"maintainability\": string[],\n"
            "  \"error_handling\": string[],\n"
            "  \"recommendations\": string[],\n"
            "  \"final_summary\": string\n"
            "}\n"
            "Do not include markdown. Do not include explanations outside JSON."
        )

        user_prompt = "Review the following Python code:\n\n"
        user_prompt += "```python\n" + code + "\n```\n"

        if exit_code is not None or stderr is not None or stdout is not None:
            user_prompt += (
                "\n\nThe code was executed. Include analysis of the execution "
                "results in your review.\n"
                "Execution results:\n"
            )
            user_prompt += json.dumps(
                {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                },
                indent=2,
            )

        if generated_tests is not None:
            user_prompt += (
                "\n\nThe following generation of tests was also produced. "
                "Include analysis of the test quality and coverage in your "
                "review.\n"
                "Generated tests:\n"
            )
            user_prompt += "```python\n" + generated_tests + "\n```\n"

        has_test_results = (
            test_exit_code is not None
            or test_stderr is not None
            or test_stdout is not None
        )
        if has_test_results:
            user_prompt += (
                "\n\nThe tests were executed. Include analysis of the test "
                "execution results in your review.\n"
                "Test execution results:\n"
            )
            user_prompt += json.dumps(
                {
                    "test_stdout": test_stdout,
                    "test_stderr": test_stderr,
                    "test_exit_code": test_exit_code,
                },
                indent=2,
            )

        logger.info("ReviewerAgent: reviewing code")
        raw_text = self._call_llm(user_prompt, system_prompt)

        try:
            data = self._parse_json(raw_text)
            return self._validate_and_build_report(data)
        except ReviewerAgentError:
            raise
        except Exception as exc:
            # malformed JSON or unexpected response shape
            logger.warning(
                "ReviewerAgent: failed first parse (%s); retrying once", exc
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
                logger.exception("ReviewerAgent: retry failed")
                raise ReviewerAgentError(
                    "Invalid review response from LLM"
                ) from exc2

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        response = self._llm_client.generate(prompt, system_prompt=system_prompt)
        return (response.text or "").strip()

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

        raise ReviewerAgentError("Malformed JSON from LLM")

    def _validate_and_build_report(self, data: Dict[str, Any]) -> ReviewReport:
        required_fields = [
            "overall_score",
            "strengths",
            "weaknesses",
            "pep8_issues",
            "performance_suggestions",
            "security_concerns",
            "logic_issues",
            "maintainability",
            "error_handling",
            "recommendations",
            "final_summary",
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise ReviewerAgentError(
                    f"Missing field in review JSON: {field_name}"
                )

        def as_str(value: Any) -> str:
            if not isinstance(value, str):
                raise ReviewerAgentError("Expected string fields in review")
            return value

        def as_str_list(value: Any) -> List[str]:
            if not isinstance(value, list) or not all(
                isinstance(x, str) for x in value
            ):
                raise ReviewerAgentError(
                    "Expected list[string] fields in review"
                )
            return value

        if not isinstance(data["overall_score"], int) or isinstance(
            data["overall_score"], bool
        ):
            raise ReviewerAgentError(
                "Expected integer for overall_score"
            )
        if not (0 <= data["overall_score"] <= 100):
            raise ReviewerAgentError(
                "overall_score must be between 0 and 100"
            )

        return ReviewReport(
            overall_score=data["overall_score"],
            strengths=as_str_list(data["strengths"]),
            weaknesses=as_str_list(data["weaknesses"]),
            pep8_issues=as_str_list(data["pep8_issues"]),
            performance_suggestions=as_str_list(
                data["performance_suggestions"]
            ),
            security_concerns=as_str_list(data["security_concerns"]),
            logic_issues=as_str_list(data["logic_issues"]),
            maintainability=as_str_list(data["maintainability"]),
            error_handling=as_str_list(data["error_handling"]),
            recommendations=as_str_list(data["recommendations"]),
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

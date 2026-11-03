"""Debugger Agent.

This agent is responsible for analyzing failed Python programs and tests and
producing a structured debugging report (``DebugReport``).

It also provides ``correct_changes()`` for repository-aware code modification.
In that mode the agent produces a corrected ``ChangeSet`` (proposed changes)
based on the failed changes and feedback. The agent NEVER writes files itself;
it only proposes corrected changes to be validated and applied by the
infrastructure layer.

RESPONSIBILITIES:
- Analyze generated Python source code and generated tests.
- Analyze application execution results (stdout, stderr, exit code).
- Analyze test execution results (stdout, stderr, exit code).
- Analyze reviewer findings.
- Classify errors (taxonomy) and identify root causes with file/line localization.
- Provide targeted diffs and corrections avoiding unnecessary full-file rewrites.
- Support iterative debugging with bounded retry mechanisms.

STRICT CONSTRAINTS (must never be violated):
- NEVER execute code or shell commands.
- NEVER write files or modify repository files directly.
- NEVER call subprocess, eval(), or exec().
- NEVER invent execution results.

IMPORTANT: This class must not contain provider-specific logic (e.g. Gemini).
It communicates ONLY through the dependency-injected ``LLMClient``.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.infrastructure.llm_client import LLMClient
from backend.domain.change_models import (
    ChangeOperation,
    ChangeSet,
    FileChange,
)

logger = logging.getLogger(__name__)


class DebuggerAgentError(RuntimeError):
    """Base error for debugger agent failures."""


@dataclass(frozen=True)
class DebugReport:
    """Structured debugging report produced by the DebuggerAgent.

    Represents the result of analyzing a failed program/test.
    """

    issue_detected: bool
    error_type: str
    root_cause: str
    affected_component: str
    explanation: str
    suggested_changes: List[str]
    corrected_code: str
    confidence: float
    final_summary: str
    error_category: str = "RuntimeError"
    line_number: Optional[int] = None
    targeted_diff: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class DebuggerAgent:
    """Produce a structured DebugReport from failed code, tests, execution
    results and reviewer feedback using an injected LLM client.

    The agent never executes, generates-on-disk, or modifies code. It only
    analyzes and reports, behaving like a Senior Python Debugging Engineer.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def debug_code(
        self,
        generated_code: str,
        *,
        generated_tests: Optional[str] = None,
        application_stdout: Optional[str] = None,
        application_stderr: Optional[str] = None,
        application_exit_code: Optional[int] = None,
        test_stdout: Optional[str] = None,
        test_stderr: Optional[str] = None,
        test_exit_code: Optional[int] = None,
        reviewer_feedback: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 3,
    ) -> DebugReport:
        """Debug failed generated Python code."""
        if not isinstance(generated_code, str):
            raise TypeError("generated_code must be a string")
        if not generated_code.strip():
            raise ValueError("generated_code must be non-empty")

        system_prompt = (
            "You are a Senior Python Debugging Engineer.\n"
            "You analyze failed Python programs and tests.\n"
            "You identify the root cause, classify the error, locate the failing line, "
            "and propose a minimal, targeted correction.\n"
            "Avoid unnecessary full-file rewrites; keep changes localized.\n"
            "You must not execute code.\n"
            "You must not invent execution results.\n"
            "You never write files and never modify the repository.\n"
            "Return ONLY a debugging result as valid JSON matching this schema:\n"
            "{\n"
            "  \"issue_detected\": boolean,\n"
            "  \"error_type\": string,\n"
            "  \"error_category\": \"SyntaxError\" | \"ImportError\" | \"AttributeError\" | \"TypeError\" | \"ValueError\" | \"IndexError\" | \"KeyError\" | \"ZeroDivisionError\" | \"AssertionError\" | \"TimeoutError\" | \"LogicError\",\n"
            "  \"line_number\": integer | null,\n"
            "  \"root_cause\": string,\n"
            "  \"affected_component\": string,\n"
            "  \"explanation\": string,\n"
            "  \"suggested_changes\": string[],\n"
            "  \"corrected_code\": string,\n"
            "  \"confidence\": number (0.0 to 1.0),\n"
            "  \"final_summary\": string\n"
            "}\n"
            "corrected_code must contain ONLY corrected Python source code "
            "with no Markdown fences. If no correction is needed, set "
            "issue_detected to false and corrected_code to the original code unchanged.\n"
            "Do not include markdown. Do not include explanations outside JSON."
        )

        user_prompt = self._build_user_prompt(
            generated_code=generated_code,
            generated_tests=generated_tests,
            application_stdout=application_stdout,
            application_stderr=application_stderr,
            application_exit_code=application_exit_code,
            test_stdout=test_stdout,
            test_stderr=test_stderr,
            test_exit_code=test_exit_code,
            reviewer_feedback=reviewer_feedback,
        )

        logger.info("DebuggerAgent: debugging (retry %d/%d)", retry_count, max_retries)
        raw_text = self._call_llm(user_prompt, system_prompt)

        try:
            data = self._parse_json(raw_text)
            return self._validate_and_build_report(
                data,
                original_code=generated_code,
                retry_count=retry_count,
                max_retries=max_retries,
                stderr_context=(application_stderr or "") + " " + (test_stderr or ""),
            )
        except DebuggerAgentError as exc:
            if "malformed" not in str(exc).lower():
                raise
            logger.warning("DebuggerAgent: failed first parse (%s); retrying once", exc)
            correction_prompt = (
                "The previous response was not valid JSON matching the required "
                "schema. Return ONLY corrected valid JSON matching the schema. "
                "Do not add explanations."
            )
            raw_text_retry = self._call_llm(correction_prompt, system_prompt)
            try:
                data_retry = self._parse_json(raw_text_retry)
                return self._validate_and_build_report(
                    data_retry,
                    original_code=generated_code,
                    retry_count=retry_count,
                    max_retries=max_retries,
                    stderr_context=(application_stderr or "") + " " + (test_stderr or ""),
                )
            except Exception as exc2:
                logger.exception("DebuggerAgent: retry failed")
                raise DebuggerAgentError("Invalid debugging response from LLM") from exc2

    def correct_changes(
        self,
        change_set: ChangeSet,
        *,
        feedback: str,
        retrieved_context: str | None = None,
    ) -> ChangeSet:
        """Produce a corrected ChangeSet for repository-aware modification."""
        if not isinstance(change_set, ChangeSet):
            raise TypeError("change_set must be a ChangeSet")
        if not change_set.changes:
            raise ValueError("change_set must contain at least one change")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be a non-empty string")

        system_prompt = (
            "You are a Senior Python Debugging Engineer.\n"
            "You analyze a failed set of proposed repository changes.\n"
            "You return ONLY a corrected ChangeSet as valid JSON.\n"
            "You do NOT write files.\n"
            "You never modify .env*, secrets, keys, credentials, tokens, "
            "consent files, deployment credentials, or .git contents.\n"
            "Return a JSON object with the shape:\n"
            "{\"changes\": [ {"
            "\"file_path\": string, \"operation\": \"create\" or \"modify\", "
            "\"new_content\": string (COMPLETE file content), "
            "\"description\": string } ], \"summary\": string}\n"
            "RULES:\n"
            "- Use POSIX '/' paths; never absolute paths or '..'.\n"
            "- Never use 'delete'.\n"
            "- Provide COMPLETE file content for every file, preserving "
            "unrelated existing content.\n"
            "- Keep corrections minimal and targeted to address the feedback.\n"
            "- Do not include Markdown fences. No explanations outside JSON."
        )

        user_prompt = self._build_correct_changes_prompt(
            change_set, feedback, retrieved_context
        )

        logger.info("DebuggerAgent: correcting repository changes")
        raw_text = self._call_llm(user_prompt, system_prompt)

        try:
            data = self._parse_json(raw_text)
            return self._validate_and_build_changes(data)
        except DebuggerAgentError as exc:
            if "malformed" not in str(exc).lower():
                raise
            logger.warning("DebuggerAgent: failed first parse (%s); retrying once", exc)
            retry = (
                "The previous response was not valid JSON matching the "
                "required schema. Return ONLY corrected valid JSON. "
                "Do not add explanations."
            )
            raw_retry = self._call_llm(retry, system_prompt)
            try:
                data_retry = self._parse_json(raw_retry)
                return self._validate_and_build_changes(data_retry)
            except Exception as exc2:
                logger.exception("DebuggerAgent: retry failed")
                raise DebuggerAgentError("Invalid corrected-changes response from LLM") from exc2

    def _build_correct_changes_prompt(
        self,
        change_set: ChangeSet,
        feedback: str,
        retrieved_context: str | None,
    ) -> str:
        parts: List[str] = []
        if retrieved_context:
            parts.append("Relevant repository context:\n" + retrieved_context)

        parts.append(
            "Failed proposed changes:\n"
            + json.dumps(
                [
                    {
                        "file_path": c.file_path,
                        "operation": c.operation.value,
                        "new_content": c.new_content,
                        "description": c.description,
                    }
                    for c in change_set.changes
                ],
                indent=2,
            )
        )
        parts.append("Feedback / failure reason:\n" + feedback)
        parts.append(
            "Produce corrected proposed changes as valid JSON, replacing the "
            "failing changes with corrected full-file content."
        )
        return "\n\n---\n\n".join(parts)

    def _validate_and_build_changes(self, data: Dict[str, Any]) -> ChangeSet:
        if "changes" not in data or not isinstance(data["changes"], list):
            raise DebuggerAgentError(
                "changes field missing or not a list in corrected-changes JSON"
            )
        if not data["changes"]:
            raise DebuggerAgentError("changes list must not be empty")

        summary = data.get("summary", "")
        if not isinstance(summary, str):
            raise DebuggerAgentError("summary must be a string")

        changes: List[FileChange] = []
        for idx, item in enumerate(data["changes"]):
            if not isinstance(item, dict):
                raise DebuggerAgentError(f"change at index {idx} must be an object")
            for key in ("file_path", "operation", "new_content"):
                if key not in item:
                    raise DebuggerAgentError(f"change at index {idx} missing field: {key}")

            file_path = item["file_path"]
            operation_raw = item["operation"]
            new_content = item["new_content"]
            description = item.get("description", "")

            if not isinstance(file_path, str) or not file_path.strip():
                raise DebuggerAgentError(f"change at index {idx} has invalid file_path")
            if not isinstance(new_content, str) or not new_content.strip():
                raise DebuggerAgentError(f"change at index {idx} has invalid new_content")
            if operation_raw not in ("create", "modify"):
                raise DebuggerAgentError(
                    f"change at index {idx} has unsupported operation: {operation_raw}"
                )

            changes.append(
                FileChange(
                    file_path=file_path,
                    operation=ChangeOperation(operation_raw),
                    new_content=new_content,
                    original_content=None,
                    original_hash=None,
                    description=description if isinstance(description, str) else "",
                )
            )

        return ChangeSet(changes=changes, summary=summary)

    def _build_user_prompt(
        self,
        *,
        generated_code: str,
        generated_tests: Optional[str],
        application_stdout: Optional[str],
        application_stderr: Optional[str],
        application_exit_code: Optional[int],
        test_stdout: Optional[str],
        test_stderr: Optional[str],
        test_exit_code: Optional[int],
        reviewer_feedback: Optional[str],
    ) -> str:
        user_prompt = (
            "Analyze the following failed Python program and tests, and "
            "produce a debugging report.\n\n"
        )
        user_prompt += "Failed Python source code:\n"
        user_prompt += "```python\n" + generated_code + "\n```\n"

        if generated_tests is not None:
            user_prompt += f"\n\nGenerated tests:\n```python\n{generated_tests}\n```\n"

        has_app_results = (
            application_exit_code is not None
            or application_stderr is not None
            or application_stdout is not None
        )
        if has_app_results:
            user_prompt += "\n\nApplication execution results:\n"
            user_prompt += json.dumps(
                {
                    "application_stdout": application_stdout,
                    "application_stderr": application_stderr,
                    "application_exit_code": application_exit_code,
                },
                indent=2,
            )

        has_test_results = (
            test_exit_code is not None
            or test_stderr is not None
            or test_stdout is not None
        )
        if has_test_results:
            user_prompt += "\n\nTest execution results:\n"
            user_prompt += json.dumps(
                {
                    "test_stdout": test_stdout,
                    "test_stderr": test_stderr,
                    "test_exit_code": test_exit_code,
                },
                indent=2,
            )

        if reviewer_feedback is not None:
            user_prompt += f"\n\nReviewer findings:\n{reviewer_feedback}\n"

        return user_prompt

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        try:
            response = self._llm_client.generate(prompt, system_prompt=system_prompt)
            return (response.text or "").strip()
        except DebuggerAgentError:
            raise
        except Exception as exc:
            raise DebuggerAgentError(f"LLM failure during debugging: {exc}") from exc

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

        raise DebuggerAgentError("Malformed JSON from LLM")

    def _validate_and_build_report(
        self,
        data: Dict[str, Any],
        original_code: str = "",
        retry_count: int = 0,
        max_retries: int = 3,
        stderr_context: str = "",
    ) -> DebugReport:
        required_fields = [
            "issue_detected",
            "error_type",
            "root_cause",
            "affected_component",
            "explanation",
            "suggested_changes",
            "corrected_code",
            "confidence",
            "final_summary",
        ]
        for field_name in required_fields:
            if field_name not in data:
                raise DebuggerAgentError(f"Missing field in debugging JSON: {field_name}")

        def as_str(value: Any) -> str:
            if not isinstance(value, str):
                raise DebuggerAgentError("Expected string fields in debugging report")
            return value

        def as_str_list(value: Any) -> List[str]:
            if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
                raise DebuggerAgentError("Expected list[string] fields in debugging report")
            return value

        if not isinstance(data["issue_detected"], bool):
            raise DebuggerAgentError("Expected boolean for issue_detected")

        confidence = data["confidence"]
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise DebuggerAgentError("Expected numeric confidence in debugging report")
        confidence = float(confidence)
        if not (0.0 <= confidence <= 1.0):
            raise DebuggerAgentError("confidence must be between 0.0 and 1.0")

        corrected_code = as_str(data["corrected_code"]).strip()
        if not corrected_code:
            raise DebuggerAgentError("corrected_code must be non-empty")

        # Classify error category
        error_type_str = as_str(data["error_type"])
        error_category = self._classify_error(
            data.get("error_category", error_type_str), stderr_context, as_str(data["explanation"])
        )

        # Detect line number
        line_num = data.get("line_number")
        if line_num is None or not isinstance(line_num, int):
            line_num = self._extract_line_number(stderr_context, as_str(data["explanation"]))

        # Generate targeted diff
        diff_str: Optional[str] = None
        if original_code and original_code != corrected_code:
            diff_lines = list(
                difflib.unified_diff(
                    original_code.splitlines(keepends=True),
                    corrected_code.splitlines(keepends=True),
                    fromfile="original.py",
                    tofile="corrected.py",
                )
            )
            diff_str = "".join(diff_lines)

        return DebugReport(
            issue_detected=data["issue_detected"],
            error_type=error_type_str,
            root_cause=as_str(data["root_cause"]),
            affected_component=as_str(data["affected_component"]),
            explanation=as_str(data["explanation"]),
            suggested_changes=as_str_list(data["suggested_changes"]),
            corrected_code=corrected_code,
            confidence=confidence,
            final_summary=as_str(data["final_summary"]),
            error_category=error_category,
            line_number=line_num,
            targeted_diff=diff_str,
            retry_count=retry_count,
            max_retries=max_retries,
        )

    def _classify_error(self, candidate: str, stderr: str, explanation: str) -> str:
        combined = f"{candidate} {stderr} {explanation}".lower()
        if "syntaxerror" in combined or "indentationerror" in combined:
            return "SyntaxError"
        if "importerror" in combined or "modulenotfounderror" in combined:
            return "ImportError"
        if "zerodivisionerror" in combined:
            return "ZeroDivisionError"
        if "typeerror" in combined:
            return "TypeError"
        if "valueerror" in combined:
            return "ValueError"
        if "indexerror" in combined:
            return "IndexError"
        if "keyerror" in combined:
            return "KeyError"
        if "attributeerror" in combined:
            return "AttributeError"
        if "assertionerror" in combined or "assert" in combined:
            return "AssertionError"
        if "timeout" in combined:
            return "TimeoutError"
        return candidate if candidate else "RuntimeError"

    def _extract_line_number(self, stderr: str, explanation: str) -> Optional[int]:
        # Match 'line 42' pattern
        match = re.search(r"line\s+(\d+)", stderr, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        match2 = re.search(r"line\s+(\d+)", explanation, flags=re.IGNORECASE)
        if match2:
            return int(match2.group(1))
        return None


def _strip_markdown_code_fences(text: str) -> str:
    stripped = text.strip()
    fence_match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", stripped, flags=re.I)
    if fence_match:
        return fence_match.group(1)

    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped

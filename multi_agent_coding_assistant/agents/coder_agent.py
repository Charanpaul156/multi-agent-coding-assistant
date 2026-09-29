"""Coder Agent.

This agent is responsible for translating a natural language programming
request into Python source code.

It also provides ``generate_changes()`` for repository-aware code modification.
In that mode the agent returns a structured ``ChangeSet`` (proposed changes)
instead of a standalone code string. The agent NEVER writes files itself; it
only proposes changes to be validated and applied by the infrastructure layer.

IMPORTANT: This class must not contain Gemini-specific logic.
It communicates ONLY through dependency-injected `LLMClient`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, List

from backend.infrastructure.llm_client import LLMClient, _sanitize_error
from agents.language_support import (
    build_coder_system_prompt,
    extract_programming_language,
    get_language_display_name,
    normalize_language,
    validate_code_language,
)

from backend.domain.change_models import (
    ChangeOperation,
    ChangeSet,
    FileChange,
)

logger = logging.getLogger(__name__)


class CoderAgentError(RuntimeError):
    """Base error for coder agent failures."""


@dataclass(frozen=True)
class CoderAgent:
    """Generate Python code from prompts using an injected LLM client."""

    llm_client: LLMClient

    def generate_changes(
        self,
        prompt: str,
        *,
        implementation_plan: Any | None = None,
        retrieved_context: str | None = None,
        plan_summary: str | None = None,
        max_attempts: int = 2,
    ) -> ChangeSet:
        """Propose a structured ChangeSet for repository-aware modification.

        Args:
            prompt: Natural language repository-modification request.
            implementation_plan: Optional structured implementation plan. This
                may be a dataclass, dict, or pre-rendered string.
            retrieved_context: Optional repository context (pre-formatted by
                the RAG layer). The agent never retrieves context internally.
            plan_summary: Optional planner output to guide the changes.

        Returns:
            A ChangeSet containing full-file proposed changes (``create`` or
            ``modify``). ``delete`` is not supported in v1.

        Raises:
            - TypeError/ValueError: invalid input.
            - CoderAgentError: if the LLM returns invalid/irrecoverable JSON.
        """
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

        if plan_summary is None and implementation_plan is not None:
            plan_summary = self._format_implementation_plan(
                implementation_plan
            )

        system_prompt = (
            "You are a senior software engineer performing repository-aware "
            "modification.\n"
            "You do NOT write files. You only PROPOSE changes.\n"
            "You return ONLY valid JSON describing an array of file changes.\n"
            "Each change must be an object with exactly:\n"
            "{\n"
            "  \"file_path\": string (repository-relative POSIX path, e.g. "
            "'backend/auth/service.py'),\n"
            "  \"operation\": \"create\" or \"modify\",\n"
            "  \"new_content\": string (COMPLETE new file content, full file, "
            "no Markdown fences, no truncation),\n"
            "  \"original_hash\": string | null (optional; include when the "
            "current file hash is known),\n"
            "  \"description\": string (concise)\n"
            "}\n"
            "Return a JSON object with the shape:\n"
            "{\"changes\": [ ... above objects ... ], \"summary\": string}\n"
            "RULES:\n"
            "- Use 'create' for new files, 'modify' for existing files.\n"
            "- NEVER use 'delete'.\n"
            "- NEVER modify .env*, secrets, keys, credentials, tokens, "
            "consent files, deployment credentials, or .git contents.\n"
            "- Use POSIX '/' separators; NEVER use absolute paths or '..'.\n"
            "- Provide the COMPLETE file content for modified files, not a "
            "patch or a partial diff. Preserve unrelated existing content.\n"
            "- If the current file hash is known, include original_hash so the "
            "application layer can reject stale changes safely.\n"
            "- Do not include Markdown fences. Do not add explanations "
            "outside JSON."
        )

        user_prompt = self._build_changes_prompt(
            prompt, retrieved_context, plan_summary
        )

        current_prompt = user_prompt
        last_error = ""

        for attempt in range(1, max_attempts + 1):
            logger.info(
                "CoderAgent: generating repository changes (attempt %d/%d)",
                attempt,
                max_attempts,
            )
            raw_text = self._call_llm(
                current_prompt,
                system_prompt,
                response_format="json",
                max_tokens=8192,
            )

            try:
                data = self._parse_json(raw_text)
                return self._validate_and_build_changeset(data)
            except CoderAgentError as exc:
                last_error = str(exc)
                is_recoverable_parse_error = any(
                    k in last_error.lower()
                    for k in ("malformed", "truncated", "unterminated", "json")
                )
                if not is_recoverable_parse_error:
                    logger.warning(
                        "CoderAgent: schema validation error (non-retryable): %s",
                        last_error,
                    )
                    raise

                logger.warning(
                    "Repository coding could not parse the AI-generated change set (attempt %d/%d): %s. "
                    "The assistant will retry with a stricter structured-output request.",
                    attempt,
                    max_attempts,
                    last_error,
                )

                if attempt >= max_attempts:
                    logger.exception(
                        "CoderAgent: retry failed after %d attempts", max_attempts
                    )
                    raise CoderAgentError(
                        f"Repository coding could not parse the AI-generated change set: {last_error}"
                    ) from exc

                is_truncated = (
                    "truncated" in last_error.lower()
                    or "unterminated" in last_error.lower()
                )
                truncation_note = (
                    "\nYour previous response appeared truncated or had unterminated strings. "
                    "Keep file contents complete, properly closed, and within token limits."
                    if is_truncated
                    else ""
                )

                current_prompt = (
                    f"{user_prompt}\n\n"
                    f"[CORRECTION REQUIRED - INVALID JSON OUTPUT]\n"
                    f"Your previous response was not valid JSON ({last_error}).{truncation_note}\n\n"
                    f"Return ONLY a valid JSON object matching the required ChangeSet schema.\n\n"
                    f"Every file content must be a valid JSON string with correctly escaped quotes, "
                    f"backslashes, and newlines.\n\n"
                    f"Do not use Markdown fences.\n"
                    f"Do not add explanations.\n"
                    f"Do not truncate file contents."
                )

        raise CoderAgentError(
            f"Failed to generate valid repository changes after {max_attempts} attempts: {last_error}"
        )

    def generate_code(
        self,
        prompt: str,
        *,
        retrieved_context: str | None = None,
        language: str | None = None,
        max_attempts: int = 2,
    ) -> str:
        """Generate code in the requested programming language.

        Args:
            prompt: Natural language programming request.
            retrieved_context: Optional repository context (pre-formatted by
                the RAG layer). Agents never retrieve context internally.
            language: Optional explicit programming language (e.g., 'java', 'python').
                If omitted or 'auto', language is extracted from the prompt.
                Defaults to Python if unspecified.
            max_attempts: Maximum generation attempts before failing on validation error.

        Returns:
            Source code only (no Markdown fences).
        """

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

        user_content = prompt
        if retrieved_context:
            user_content = f"{retrieved_context}\n\n---\n\nUser request:\n{prompt}"

        target_lang = normalize_language(language) or extract_programming_language(prompt) or "python"
        display_name = get_language_display_name(target_lang)
        system_prompt = build_coder_system_prompt(target_lang)

        current_prompt = user_content
        attempts = 0
        last_failure_reason: str | None = None

        while attempts < max_attempts:
            attempts += 1
            logger.info(
                "CoderAgent: generating code (attempt %d/%d, language: %s)",
                attempts,
                max_attempts,
                display_name,
            )
            response = self.llm_client.generate(current_prompt, system_prompt=system_prompt)
            code = response.text or ""
            cleaned = _strip_markdown_code_fences(code).strip()

            if len(cleaned) < 10:
                last_failure_reason = "LLM returned invalid/too-short code"
                current_prompt = (
                    f"{user_content}\n\n"
                    f"[SYSTEM NOTICE: The previous output was too short or empty. "
                    f"Generate complete, valid {display_name} code to solve the request.]"
                )
                continue

            # Validate language
            is_valid, reason = validate_code_language(cleaned, target_lang)
            if not is_valid:
                logger.warning(
                    "CoderAgent: language validation failed for %s on attempt %d: %s",
                    display_name,
                    attempts,
                    reason,
                )
                last_failure_reason = reason
                current_prompt = (
                    f"{user_content}\n\n"
                    f"[CRITICAL ERROR - LANGUAGE MISMATCH]\n"
                    f"Your previous attempt generated the wrong language: {reason}\n"
                    f"You MUST generate ONLY {display_name} code.\n"
                    f"Do NOT output Python, Java, or any other programming language.\n"
                    f"Keep the code concise, focused, and proportional to the task."
                )
                continue

            return cleaned

        raise CoderAgentError(
            f"Failed to generate valid {display_name} code after {max_attempts} attempts: {last_failure_reason}"
        )

    # ------------------------------------------------------------------ #
    # generate_changes helpers
    # ------------------------------------------------------------------ #

    def _build_changes_prompt(
        self,
        prompt: str,
        retrieved_context: str | None,
        plan_summary: str | None,
    ) -> str:
        """Assemble the user prompt for repository-aware change generation."""
        parts: List[str] = []

        if retrieved_context:
            parts.append(
                "The following are relevant sections from the repository:\n"
                + retrieved_context
            )

        if plan_summary:
            parts.append("Implementation plan:\n" + plan_summary)

        parts.append("User request:\n" + prompt)
        parts.append(
            "Determine which files need to be created or modified to satisfy "
            "the request. Return the proposed changes as valid JSON."
        )
        return "\n\n---\n\n".join(parts)

    def _call_llm(
        self,
        prompt: str,
        system_prompt: str,
        *,
        response_format: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        try:
            kwargs: Dict[str, Any] = {"system_prompt": system_prompt}
            if response_format is not None:
                kwargs["response_format"] = response_format
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            response = self.llm_client.generate(prompt, **kwargs)
            return (response.text or "").strip()
        except TypeError:
            # Fallback if injected fake client only accepts (prompt, system_prompt=...)
            response = self.llm_client.generate(prompt, system_prompt=system_prompt)
            return (response.text or "").strip()
        except CoderAgentError:
            raise
        except Exception as exc:
            sanitized = _sanitize_error(str(exc))
            raise CoderAgentError(
                f"LLM failure during change generation: {sanitized}"
            ) from exc

    def _is_truncated_json(self, text: str, err: Exception) -> bool:
        """Detect if a JSON string appears to be truncated mid-generation."""
        err_msg = str(err).lower()
        if "unterminated string" in err_msg:
            return True
        if "expecting value" in err_msg or "expecting property name" in err_msg:
            stripped = text.strip()
            if (
                stripped.endswith(":")
                or stripped.endswith(",")
                or stripped.endswith('"')
                or stripped.endswith("\\")
            ):
                return True
        if text.count("{") > text.count("}"):
            return True
        if text.count("[") > text.count("]"):
            return True
        return False

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Safely parse JSON from LLM response, detecting malformed or truncated payloads."""
        if not isinstance(text, str) or not text.strip():
            raise CoderAgentError("Empty response from LLM; expected a JSON ChangeSet.")

        cleaned = _strip_markdown_code_fences(text).strip()

        # 1. Attempt standard JSON parsing first on cleaned text
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
            raise CoderAgentError(
                f"Malformed JSON: expected a JSON object (dict), got {type(parsed).__name__}"
            )
        except json.JSONDecodeError as err:
            first_err = err

        # 2. Locate the intended outer JSON object safely without greedy arbitrary regex
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = cleaned[start_idx : end_idx + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
                raise CoderAgentError(
                    f"Malformed JSON: expected a JSON object (dict), got {type(parsed).__name__}"
                )
            except json.JSONDecodeError as err:
                parse_err = err
        else:
            parse_err = first_err

        # 3. Detect malformed vs truncated JSON
        if self._is_truncated_json(cleaned, parse_err):
            raise CoderAgentError(
                "Truncated JSON response from LLM: unterminated string or incomplete brackets."
            )

        sanitized_msg = _sanitize_error(str(parse_err))
        raise CoderAgentError(f"Malformed JSON from LLM: {sanitized_msg}")

    def _validate_and_build_changeset(self, data: Dict[str, Any]) -> ChangeSet:
        """Strictly validate LLM change JSON and build a ChangeSet."""
        if not isinstance(data, dict):
            raise CoderAgentError("ChangeSet payload must be a JSON object (dict)")
        if "changes" not in data or not isinstance(data["changes"], list):
            raise CoderAgentError(
                "changes field missing or not a list in change JSON"
            )
        if not data["changes"]:
            raise CoderAgentError("changes list must not be empty")

        if len(data["changes"]) > 50:
            raise CoderAgentError("ChangeSet exceeds safe limit of 50 files per generation")

        summary = data.get("summary", "")
        if not isinstance(summary, str):
            raise CoderAgentError("summary must be a string")

        changes: List[FileChange] = []
        for idx, item in enumerate(data["changes"]):
            changes.append(self._build_file_change(item, idx))

        return ChangeSet(changes=changes, summary=summary)

    def _build_file_change(self, item: Any, idx: int) -> FileChange:
        if not isinstance(item, dict):
            raise CoderAgentError(
                f"change at index {idx} must be an object"
            )
        for key in ("file_path", "operation", "new_content"):
            if key not in item:
                raise CoderAgentError(
                    f"change at index {idx} missing field: {key}"
                )

        for key in ("description",):
            if key not in item:
                raise CoderAgentError(
                    f"change at index {idx} missing field: {key}"
                )

        file_path = item["file_path"]
        operation_raw = item["operation"]
        new_content = item["new_content"]
        description = item["description"]
        original_hash = item.get("original_hash", None)

        if not isinstance(file_path, str) or not file_path.strip():
            raise CoderAgentError(
                f"change at index {idx} has invalid file_path"
            )
        if (
            file_path.startswith("/")
            or file_path.startswith("\\")
            or file_path.startswith("../")
            or "/../" in file_path
            or file_path.endswith("/..")
            or file_path == ".."
        ):
            raise CoderAgentError(
                f"change at index {idx} has invalid or unsafe file_path: {file_path}"
            )
        if not isinstance(new_content, str) or not new_content.strip():
            raise CoderAgentError(
                f"change at index {idx} has invalid new_content"
            )
        if len(new_content.encode("utf-8")) > 2_000_000:
            raise CoderAgentError(
                f"change at index {idx} exceeds maximum content size limit"
            )
        if operation_raw not in ("create", "modify"):
            raise CoderAgentError(
                f"change at index {idx} has unsupported operation: "
                f"{operation_raw} (only create/modify allowed in v1)"
            )
        if "original_hash" in item and original_hash is not None and not isinstance(
            original_hash, str
        ):
            raise CoderAgentError(
                f"change at index {idx} has invalid original_hash"
            )
        if isinstance(original_hash, str) and not original_hash.strip():
            raise CoderAgentError(
                f"change at index {idx} has invalid original_hash"
            )
        if not isinstance(description, str):
            raise CoderAgentError(
                f"change at index {idx} has invalid description"
            )

        operation = ChangeOperation(operation_raw)
        return FileChange(
            file_path=file_path,
            operation=operation,
            new_content=new_content,
            original_content=None,
            original_hash=original_hash,
            description=description,
        )

    def _format_implementation_plan(self, implementation_plan: Any) -> str:
        """Render implementation plan input for prompt construction."""
        if isinstance(implementation_plan, str):
            return implementation_plan.strip()

        if is_dataclass(implementation_plan):
            try:
                payload = asdict(implementation_plan)
            except Exception:
                return str(implementation_plan)
            return json.dumps(payload, indent=2, sort_keys=True, default=str)

        if isinstance(implementation_plan, dict):
            return json.dumps(implementation_plan, indent=2, sort_keys=True, default=str)

        if hasattr(implementation_plan, "__dict__"):
            try:
                return json.dumps(
                    vars(implementation_plan),
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            except Exception:
                return str(implementation_plan)

        return str(implementation_plan)


def _strip_markdown_code_fences(text: str) -> str:
    """Remove leading/trailing triple-backtick fences if present."""

    stripped = text.strip()

    # Matches ```python ... ``` or ``` ... ```
    fence_match = re.match(r"^```(?:python)?\s*([\s\S]*?)\s*```$", stripped, flags=re.I)
    if fence_match:
        return fence_match.group(1)

    # Fallback: remove start/end fences independently.
    stripped = re.sub(r"^```(?:python)?\s*", "", stripped, flags=re.I)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped


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
from dataclasses import dataclass
from typing import Any, Dict, List

from backend.infrastructure.llm_client import LLMClient

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
        retrieved_context: str | None = None,
        plan_summary: str | None = None,
    ) -> ChangeSet:
        """Propose a structured ChangeSet for repository-aware modification.

        Args:
            prompt: Natural language repository-modification request.
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
            "- Do not include Markdown fences. Do not add explanations "
            "outside JSON."
        )

        user_prompt = self._build_changes_prompt(
            prompt, retrieved_context, plan_summary
        )

        logger.info("CoderAgent: generating repository changes")
        raw_text = self._call_llm(user_prompt, system_prompt)

        try:
            data = self._parse_json(raw_text)
            return self._validate_and_build_changeset(data)
        except CoderAgentError as exc:
            if "malformed" not in str(exc).lower():
                raise
            logger.warning(
                "CoderAgent: failed first parse (%s); retrying once", exc
            )
            retry = (
                "The previous response was not valid JSON matching the "
                "required schema. Return ONLY corrected valid JSON. "
                "Do not add explanations."
            )
            raw_retry = self._call_llm(retry, system_prompt)
            try:
                data_retry = self._parse_json(raw_retry)
                return self._validate_and_build_changeset(data_retry)
            except Exception as exc2:
                logger.exception("CoderAgent: retry failed")
                raise CoderAgentError(
                    "Invalid change-generation response from LLM"
                ) from exc2

    def generate_code(
        self,
        prompt: str,
        *,
        retrieved_context: str | None = None,
    ) -> str:
        """Generate Python code.

        Args:
            prompt: Natural language programming request.
            retrieved_context: Optional repository context (pre-formatted by
                the RAG layer). Agents never retrieve context internally.

        Returns:
            Python code only (no Markdown fences).
        """

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

        if retrieved_context:
            prompt = f"{retrieved_context}\n\n---\n\nUser request:\n{prompt}"

        system_prompt = (
            "You are a senior Python software engineer. "
            "Generate production-quality Python code that satisfies the request. "
            "Return ONLY the raw Python source code. "
            "Do NOT include Markdown code fences, explanations, or comments outside "
            "the code. If you must include comments, include them inside the code." 
        )

        logger.info("CoderAgent: generating code")
        response = self.llm_client.generate(prompt, system_prompt=system_prompt)
        code = response.text

        cleaned = _strip_markdown_code_fences(code).strip()
        if not cleaned or "def " not in cleaned and "import " not in cleaned:
            # Heuristic: still allow small snippets; keep it lenient.
            if len(cleaned) < 10:
                raise CoderAgentError("LLM returned invalid/too-short code")

        return cleaned

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

    def _call_llm(self, prompt: str, system_prompt: str) -> str:
        try:
            response = self.llm_client.generate(
                prompt, system_prompt=system_prompt
            )
            return (response.text or "").strip()
        except CoderAgentError:
            raise
        except Exception as exc:
            raise CoderAgentError(
                f"LLM failure during change generation: {exc}"
            ) from exc

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

        raise CoderAgentError("Malformed JSON from LLM")

    def _validate_and_build_changeset(self, data: Dict[str, Any]) -> ChangeSet:
        """Strictly validate LLM change JSON and build a ChangeSet."""
        if "changes" not in data or not isinstance(data["changes"], list):
            raise CoderAgentError(
                "changes field missing or not a list in change JSON"
            )
        if not data["changes"]:
            raise CoderAgentError("changes list must not be empty")

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

        file_path = item["file_path"]
        operation_raw = item["operation"]
        new_content = item["new_content"]
        description = item.get("description", "")

        if not isinstance(file_path, str) or not file_path.strip():
            raise CoderAgentError(
                f"change at index {idx} has invalid file_path"
            )
        if not isinstance(new_content, str) or not new_content.strip():
            raise CoderAgentError(
                f"change at index {idx} has invalid new_content"
            )
        if operation_raw not in ("create", "modify"):
            raise CoderAgentError(
                f"change at index {idx} has unsupported operation: "
                f"{operation_raw} (only create/modify allowed in v1)"
            )

        operation = ChangeOperation(operation_raw)
        return FileChange(
            file_path=file_path,
            operation=operation,
            new_content=new_content,
            original_content=None,
            original_hash=None,
            description=description if isinstance(description, str) else "",
        )


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


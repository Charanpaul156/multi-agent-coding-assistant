"""Coder Agent.

This agent is responsible for translating a natural language programming
request into Python source code.

IMPORTANT: This class must not contain Gemini-specific logic.
It communicates ONLY through dependency-injected `LLMClient`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from backend.infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class CoderAgentError(RuntimeError):
    """Base error for coder agent failures."""


@dataclass(frozen=True)
class CoderAgent:
    """Generate Python code from prompts using an injected LLM client."""

    llm_client: LLMClient

    def generate_code(self, prompt: str) -> str:
        """Generate Python code.

        Args:
            prompt: Natural language programming request.

        Returns:
            Python code only (no Markdown fences).
        """

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty")

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


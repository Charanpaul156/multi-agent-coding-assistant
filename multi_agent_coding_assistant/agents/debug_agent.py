"""Debug Agent (compatibility alias for DebuggerAgent)."""

from __future__ import annotations

from typing import Any, Optional

from agents.debugger_agent import DebugReport, DebuggerAgent
from backend.infrastructure.llm_client import LLMClient


class DebugAgent:
    """Convenience and backwards-compatible wrapper for DebuggerAgent."""

    def __init__(self, llm_client: Optional[LLMClient] = None) -> None:
        self._llm_client = llm_client

    def run(self, error: str, context: Optional[str] = None) -> DebugReport:
        if self._llm_client is None:
            from backend.api.deps import get_llm_client
            self._llm_client = get_llm_client()

        agent = DebuggerAgent(self._llm_client)
        code_to_debug = context or "# Code unavailable\ndef placeholder():\n    pass\n"
        return agent.debug_code(
            code_to_debug,
            application_stderr=error,
        )

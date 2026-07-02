"""LLM client placeholder.

This module is prepared for future integration with LLM providers such as
OpenAI/Gemini. At the foundation stage, it intentionally contains no
business logic or API calls.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResponse:
    """A minimal, provider-agnostic response shape."""

    text: str


class LLMClient:
    """Abstraction for an LLM provider.

    Future implementation will live here (or in infrastructure-specific
    subclasses).
    """

    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError("LLMClient not implemented yet")


"""Application use-cases.

Use-cases should be framework-agnostic.
FastAPI-specific types (Request/Response) must not be used here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agents.coder_agent import CoderAgent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GenerateCodeRequest:
    """Request DTO for generating code."""

    prompt: str
    retrieved_context: str | None = None


@dataclass(frozen=True)
class GenerateCodeResult:
    """Result DTO for generated code."""

    generated_code: str


class GenerateCodeUseCase:
    """Use-case: generate Python code from a natural language prompt."""

    def __init__(self, coder_agent: CoderAgent) -> None:
        self._coder_agent = coder_agent

    def execute(self, request: GenerateCodeRequest) -> GenerateCodeResult:
        """Execute the use-case."""

        if not isinstance(request.prompt, str):
            raise TypeError("prompt must be a string")
        if not request.prompt.strip():
            raise ValueError("prompt must be non-empty")

        logger.info("GenerateCodeUseCase: start")
        code = self._coder_agent.generate_code(
            request.prompt,
            retrieved_context=request.retrieved_context,
        )
        logger.info("GenerateCodeUseCase: finished")
        return GenerateCodeResult(generated_code=code)


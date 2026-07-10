"""FastAPI dependency injection scaffolding.

At this stage we provide minimal DI wiring for the Coder capability.
Dependencies remain loosely coupled so future agents can reuse the same
infrastructure pieces.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from agents.coder_agent import CoderAgent
from backend.application.use_cases import GenerateCodeUseCase
from backend.infrastructure.llm_client import LLMClient
from backend.tools.python_executor import ExecuteCodeUseCase, PythonExecutor


def placeholder_dependency() -> Any:
    """Return a placeholder dependency.

    Replace this when real orchestration/use-cases are implemented.
    """

    return None


def build_dependency_provider(factory: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap a factory to become a FastAPI dependency provider."""

    def _provider() -> Any:
        return factory()

    return _provider


@lru_cache(maxsize=1)
def _get_llm_client() -> LLMClient:
    return LLMClient()


def get_coder_agent() -> CoderAgent:
    return CoderAgent(llm_client=_get_llm_client())


@lru_cache(maxsize=1)
def get_generate_code_use_case() -> GenerateCodeUseCase:
    return GenerateCodeUseCase(coder_agent=get_coder_agent())


@lru_cache(maxsize=1)
def get_execute_code_use_case() -> ExecuteCodeUseCase:
    return ExecuteCodeUseCase(executor=PythonExecutor(timeout_seconds=10.0))



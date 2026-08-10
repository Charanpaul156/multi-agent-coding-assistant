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
from agents.code_reviewer_agent import ReviewerAgent
from agents.debugger_agent import DebuggerAgent
from agents.test_generator_agent import TestGeneratorAgent
from backend.application.use_cases import GenerateCodeUseCase
from backend.application.debugging_use_cases import DebugCodeUseCase
from backend.application.planning_use_cases import GeneratePlanUseCase
from config.settings import get_settings
from backend.application.review_use_cases import ReviewCodeUseCase
from backend.application.test_generation_use_cases import GenerateTestsUseCase
from backend.application.workflow_use_cases import RunWorkflowUseCase
from agents.planner_agent import PlannerAgent

from backend.infrastructure.llm_client import LLMClient
from backend.tools.python_executor import ExecuteCodeUseCase, PythonExecutor
from backend.tools.test_executor import ExecuteTestsUseCase, TestExecutor

from backend.application.rag_use_cases import (
    GetRagStatusUseCase,
    IndexRepositoryUseCase,
    SearchRepositoryUseCase,
)
from backend.application.modify_repository_use_cases import ModifyRepositoryUseCase
from backend.infrastructure.change_applier import ChangeApplier
from backend.infrastructure.change_validation import ChangeValidator
from rag.chunker import CodeChunker
from rag.config import RagConfig
from rag.embedder import Embedder
from rag.embedders.sentence_transformer import (
    get_sentence_transformer_embedder,
)
from rag.file_loader import RepositoryLoader
from rag.pipeline import RepositoryRagPipeline
from rag.vector_stores.chroma import get_chroma_vector_store


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
def get_planner_agent() -> PlannerAgent:
    return PlannerAgent(llm_client=_get_llm_client())


@lru_cache(maxsize=1)
def get_generate_plan_use_case() -> GeneratePlanUseCase:
    return GeneratePlanUseCase(planner_agent=get_planner_agent())



@lru_cache(maxsize=1)
def get_execute_code_use_case() -> ExecuteCodeUseCase:
    return ExecuteCodeUseCase(executor=PythonExecutor(timeout_seconds=10.0))


@lru_cache(maxsize=1)
def get_reviewer_agent() -> ReviewerAgent:
    return ReviewerAgent(llm_client=_get_llm_client())


@lru_cache(maxsize=1)
def get_review_code_use_case() -> ReviewCodeUseCase:
    return ReviewCodeUseCase(reviewer_agent=get_reviewer_agent())


@lru_cache(maxsize=1)
def get_test_generator_agent() -> TestGeneratorAgent:
    return TestGeneratorAgent(llm_client=_get_llm_client())


@lru_cache(maxsize=1)
def get_generate_tests_use_case() -> GenerateTestsUseCase:
    return GenerateTestsUseCase(test_generator_agent=get_test_generator_agent())


@lru_cache(maxsize=1)
def get_debugger_agent() -> DebuggerAgent:
    return DebuggerAgent(llm_client=_get_llm_client())


@lru_cache(maxsize=1)
def get_debug_code_use_case() -> DebugCodeUseCase:
    return DebugCodeUseCase(debugger_agent=get_debugger_agent())


@lru_cache(maxsize=1)
def get_test_execution_use_case() -> ExecuteTestsUseCase:
    """Provide the test execution use-case.

    Uses a dedicated TestExecutor (never PythonExecutor). The fixed pytest
    command is configured by the application, never accepted from the user.
    """
    return ExecuteTestsUseCase(executor=TestExecutor(timeout_seconds=30.0))


@lru_cache(maxsize=1)
def get_run_workflow_use_case() -> RunWorkflowUseCase:
    """Provide the workflow orchestrator.

    Reuses existing providers; does NOT instantiate agents directly.
    """
    return RunWorkflowUseCase(
        plan_use_case=get_generate_plan_use_case(),
        coder_use_case=get_generate_code_use_case(),
        test_generation_use_case=get_generate_tests_use_case(),
        execute_use_case=get_execute_code_use_case(),
        test_execution_use_case=get_test_execution_use_case(),
        review_use_case=get_review_code_use_case(),
        debug_use_case=get_debug_code_use_case(),
        max_iterations=get_settings().max_iterations,
    )


# ---------------------------------------------------------------------------
# Repository RAG wiring
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_rag_config() -> RagConfig:
    """Construct RagConfig from Settings. Values are not scattered."""
    settings = get_settings()
    return RagConfig(
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        vector_store_path=settings.vector_store_path,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
        top_k=settings.rag_top_k,
        allowed_repository_roots=tuple(settings.allowed_repository_roots),
    )


@lru_cache(maxsize=1)
def _get_embedder() -> Embedder:
    config = _build_rag_config()
    return get_sentence_transformer_embedder(config.embedding_model)


@lru_cache(maxsize=1)
def _get_vector_store() -> Any:
    config = _build_rag_config()
    return get_chroma_vector_store(
        persist_directory=config.vector_store_path,
        embedding_dimension=config.embedding_dimension,
    )


@lru_cache(maxsize=1)
def get_repository_rag_pipeline() -> RepositoryRagPipeline:
    """Provide a cached RAG pipeline (loader + chunker + embedder + store)."""
    config = _build_rag_config()
    return RepositoryRagPipeline(
        config=config,
        loader=RepositoryLoader(config),
        chunker=CodeChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        ),
        embedder=_get_embedder(),
        vector_store=_get_vector_store(),
    )


@lru_cache(maxsize=1)
def get_index_repository_use_case() -> IndexRepositoryUseCase:
    return IndexRepositoryUseCase(pipeline=get_repository_rag_pipeline())


@lru_cache(maxsize=1)
def get_search_repository_use_case() -> SearchRepositoryUseCase:
    return SearchRepositoryUseCase(pipeline=get_repository_rag_pipeline())


@lru_cache(maxsize=1)
def get_rag_status_use_case() -> GetRagStatusUseCase:
    return GetRagStatusUseCase(pipeline=get_repository_rag_pipeline())


# ---------------------------------------------------------------------------
# Repository-aware code modification wiring
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_modify_repository_use_case() -> ModifyRepositoryUseCase:
    """Provide the repository-modification orchestrator.

    Reuses the existing RAG pipeline for retrieval, existing agents/use-cases
    for planning/coding/tests/review/debug, and the dedicated
    ChangeValidator + ChangeApplier for safe application. No filesystem
    modification logic lives in the API layer.
    """
    config = _build_rag_config()
    return ModifyRepositoryUseCase(
        config=config,
        retriever_use_case=get_search_repository_use_case(),
        plan_use_case=get_generate_plan_use_case(),
        coder_agent=get_coder_agent(),
        validator=ChangeValidator(config=config),
        applier=ChangeApplier(config=config),
        test_generation_use_case=get_generate_tests_use_case(),
        test_execution_use_case=get_test_execution_use_case(),
        review_use_case=get_review_code_use_case(),
        debugger_agent=get_debugger_agent(),
        max_iterations=get_settings().max_iterations,
    )



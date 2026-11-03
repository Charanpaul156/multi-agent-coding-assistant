"""Unit tests for the Researcher / RAG Agent."""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from agents.rag_agent import (
    RagAgent,
    RagAgentError,
    ResearchReport,
    SourceReference,
)
from backend.infrastructure.llm_client import LLMResponse
from rag.retriever import RetrievedChunk
from rag.retriever import RepositoryRetriever


class _FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 5, repository: str | None = None):
        return self.chunks[:top_k]


class _FakeLLMClient:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = None
        self.last_system_prompt = None

    def generate(self, prompt: str, *, system_prompt: str | None = None):
        self.last_prompt = prompt
        self.last_system_prompt = system_prompt
        return LLMResponse(text=self.response_text)


def test_rag_agent_research_success() -> None:
    chunks = [
        RetrievedChunk(
            content="import os\nfrom pathlib import Path\ndef load_data():\n    return Path('test.txt').read_text()",
            file_path="src/loader.py",
            start_line=1,
            end_line=4,
            language="python",
            repository="my-repo",
            chunk_index=0,
            distance=0.1,
        )
    ]
    retriever = _FakeRetriever(chunks)
    llm = _FakeLLMClient(
        '{"summary": "Loader module loads data from disk.", "answer": "The `load_data` function in `src/loader.py` reads data.", "architecture_insights": ["Clean modular design"], "confidence": 0.98}'
    )
    agent = RagAgent(llm_client=llm, retriever=retriever)  # type: ignore[arg-type]

    report = agent.research("How is data loaded?")

    assert isinstance(report, ResearchReport)
    assert report.query == "How is data loaded?"
    assert report.summary == "Loader module loads data from disk."
    assert "src/loader.py" in report.answer
    assert len(report.sources) == 1
    assert report.sources[0].file_path == "src/loader.py"
    assert "os" in report.related_dependencies or "pathlib" in report.related_dependencies
    assert report.confidence == 0.98


def test_rag_agent_run_backward_compatibility() -> None:
    chunks = []
    retriever = _FakeRetriever(chunks)
    llm = _FakeLLMClient(
        '{"summary": "No data found.", "answer": "No relevant files.", "architecture_insights": [], "confidence": 0.5}'
    )
    agent = RagAgent(llm_client=llm, retriever=retriever)  # type: ignore[arg-type]

    report = agent.run("Where is auth implemented?")
    assert isinstance(report, ResearchReport)
    assert report.query == "Where is auth implemented?"


def test_rag_agent_empty_query_raises() -> None:
    llm = _FakeLLMClient("{}")
    agent = RagAgent(llm_client=llm)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        agent.research("")
    with pytest.raises(TypeError):
        agent.research(None)  # type: ignore[arg-type]


def test_empty_retrieval_returns_insufficient_evidence_without_llm_call() -> None:
    llm = _FakeLLMClient('{"summary": "invented", "answer": "invented"}')
    agent = RagAgent(llm_client=llm, retriever=_FakeRetriever([]))  # type: ignore[arg-type]

    report = agent.research("Where is auth implemented?")

    assert report.evidence_status == "insufficient"
    assert report.sources == []
    assert "unknown" in report.answer
    assert llm.last_prompt is None


def test_invalid_retrieved_documents_are_omitted() -> None:
    retriever = _FakeRetriever([SimpleNamespace(file_path="", content="no provenance")])
    agent = RagAgent(retriever=retriever)  # type: ignore[arg-type]

    report = agent.research("Find the implementation")

    assert report.evidence_status == "insufficient"
    assert report.sources == []


def test_retrieval_failure_returns_unavailable_report() -> None:
    class BrokenRetriever:
        def retrieve(self, query: str, top_k: int = 5):
            raise RuntimeError("vector store unavailable")

    agent = RagAgent(retriever=BrokenRetriever())

    report = agent.research("Find auth")

    assert report.evidence_status == "unavailable"
    assert report.warnings
    assert report.sources == []


@pytest.mark.parametrize("failure_stage", ["embedding", "vector_store"])
def test_embedding_and_vector_store_failures_return_unavailable_report(
    failure_stage: str,
) -> None:
    class TestEmbedder:
        def embed_query(self, query: str) -> list[float]:
            if failure_stage == "embedding":
                raise RuntimeError("embedding service unavailable")
            return [1.0]

    class TestVectorStore:
        def similarity_search(self, query_embedding, top_k=5, metadata_filter=None):
            raise RuntimeError("vector database unavailable")

    retriever = RepositoryRetriever(TestVectorStore(), TestEmbedder())  # type: ignore[arg-type]
    report = RagAgent(retriever=retriever).research("Find auth")

    assert report.evidence_status == "unavailable"
    assert report.sources == []
    assert report.warnings


def test_llm_failure_preserves_retrieved_source_evidence() -> None:
    class BrokenLLM:
        def generate(self, prompt: str, *, system_prompt: str | None = None):
            raise RuntimeError("provider unavailable")

    chunk = RetrievedChunk(
        content="def login(): pass",
        file_path="src/auth.py",
        start_line=4,
        end_line=4,
        language="python",
        repository="repo",
        chunk_index=0,
        distance=0.1,
    )
    agent = RagAgent(llm_client=BrokenLLM(), retriever=_FakeRetriever([chunk]))  # type: ignore[arg-type]

    report = agent.research("Where is login?")

    assert report.evidence_status == "available"
    assert report.sources[0].file_path == "src/auth.py"
    assert report.warnings
    assert report.confidence == 0.0


def test_report_models_keep_legacy_positional_construction() -> None:
    source = SourceReference("src/app.py", 1, 2, 0.8, "def app(): pass")
    report = ResearchReport("query", "summary", "answer", [source])

    assert source.language == ""
    assert report.sources == [source]
    assert report.evidence_status == "available"


"""Unit tests for the Researcher / RAG Agent."""

from __future__ import annotations

import pytest

from agents.rag_agent import (
    RagAgent,
    RagAgentError,
    ResearchReport,
    SourceReference,
)
from backend.infrastructure.llm_client import LLMResponse
from rag.retriever import RetrievedChunk


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

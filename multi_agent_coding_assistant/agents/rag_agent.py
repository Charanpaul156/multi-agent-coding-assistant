"""Researcher / RAG Agent.

This agent is responsible for repository-wide contextual understanding, semantic
codebase retrieval, source dependency extraction, and synthesizing grounded,
citation-backed answers about a codebase.

RESPONSIBILITIES:
- Semantic codebase retrieval and relevance ranking.
- Filtering low-relevance retrieval results to minimize noise.
- Extracting related code dependencies (imports, module relationships, callers).
- Providing exact source file and location references (path, start/end lines).
- Synthesizing structured research reports for codebase Q&A and architecture inquiries.

STRICT CONSTRAINTS:
- NEVER write or modify files on disk.
- NEVER execute code or shell commands.
- It communicates ONLY through dependency-injected `LLMClient` and `RepositoryRetriever`.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.infrastructure.llm_client import LLMClient, _sanitize_error
from rag.retriever import RepositoryRetriever, RetrievedChunk

logger = logging.getLogger(__name__)


class RagAgentError(RuntimeError):
    """Base error for RAG / Researcher agent failures."""


@dataclass(frozen=True)
class SourceReference:
    """Exact source location reference cited in research."""

    file_path: str
    start_line: int
    end_line: int
    language: str
    snippet: str
    relevance_score: float = 1.0


@dataclass(frozen=True)
class ResearchReport:
    """Structured report produced by the Researcher / RAG Agent."""

    query: str
    answer: str
    summary: str
    sources: List[SourceReference] = field(default_factory=list)
    related_dependencies: List[str] = field(default_factory=list)
    architecture_insights: List[str] = field(default_factory=list)
    confidence: float = 1.0


class RagAgent:
    """Intelligent codebase researcher and repository query answering agent."""

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: Optional[RepositoryRetriever] = None,
    ) -> None:
        self._llm_client = llm_client
        self._retriever = retriever

    @property
    def retriever(self) -> Optional[RepositoryRetriever]:
        return self._retriever

    def run(self, query: str, **kwargs: Any) -> ResearchReport:
        """Execute codebase research query (backward-compatible entrypoint)."""
        return self.research(query, **kwargs)

    def research(
        self,
        query: str,
        *,
        repository: Optional[str] = None,
        top_k: int = 5,
        min_relevance: float = 0.0,
        retrieved_context: Optional[str] = None,
    ) -> ResearchReport:
        """Search codebase and synthesize a structured, grounded answer.

        Args:
            query: Natural language query about the codebase or architecture.
            repository: Optional repository identifier filter.
            top_k: Number of semantic chunks to retrieve.
            min_relevance: Minimum relevance score (0.0 to 1.0) to include a chunk.
            retrieved_context: Optional pre-retrieved context string if retriever is omitted.

        Returns:
            A structured ResearchReport with citations, dependencies, and insights.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must be non-empty")

        chunks: List[RetrievedChunk] = []
        if self._retriever is not None:
            try:
                raw_chunks = self._retriever.retrieve(
                    query, top_k=top_k, repository=repository
                )
                chunks = self._filter_and_rank_chunks(raw_chunks, min_relevance=min_relevance)
            except Exception as exc:
                logger.warning("RAG retriever failed: %s; falling back to context", exc)

        sources = self._build_source_references(chunks)
        dependencies = self._extract_dependencies(chunks)

        # Build prompt for LLM synthesis
        context_text = self._format_chunks_for_llm(chunks)
        if retrieved_context and not context_text:
            context_text = retrieved_context

        system_prompt = (
            "You are a Senior Principal Software Architect and Repository Researcher.\n"
            "Your task is to analyze codebase context and answer technical queries with precision.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer based ONLY on the provided codebase context and factual architecture.\n"
            "2. Always cite specific files, functions, classes, and line numbers where appropriate.\n"
            "3. Identify related module dependencies, callers, and architectural patterns.\n"
            "4. Return ONLY valid JSON matching the exact schema below.\n"
            "5. Do NOT include markdown code fences (```json) outside the JSON.\n\n"
            "SCHEMA:\n"
            "{\n"
            '  "summary": "1-2 sentence concise executive answer",\n'
            '  "answer": "Comprehensive, technical, markdown-formatted answer with file references",\n'
            '  "architecture_insights": ["Key architectural insight 1", "Insight 2"],\n'
            '  "confidence": 0.95\n'
            "}"
        )

        user_prompt = f"Technical Query: {query}\n\n"
        if context_text:
            user_prompt += f"--- Codebase Context ---\n{context_text}\n\n"
        if dependencies:
            user_prompt += f"--- Detected Dependencies ---\n{', '.join(dependencies[:20])}\n\n"
        user_prompt += "Synthesize a structured research report answering the technical query."

        try:
            response = self._llm_client.generate(user_prompt, system_prompt=system_prompt)
            data = self._parse_json(response.text or "")
        except Exception as exc:
            sanitized = _sanitize_error(str(exc))
            logger.warning("RagAgent LLM synthesis failed: %s; using deterministic fallback", sanitized)
            data = {
                "summary": f"Research results for '{query}' with {len(sources)} referenced sources.",
                "answer": f"Found {len(sources)} relevant sections in the repository matching the query.\n\n"
                + "\n".join(f"- `{s.file_path}` (lines {s.start_line}-{s.end_line})" for s in sources),
                "architecture_insights": [f"Referenced {len(sources)} files across the repository."],
                "confidence": 0.7 if sources else 0.3,
            }

        return ResearchReport(
            query=query,
            summary=str(data.get("summary", "")),
            answer=str(data.get("answer", "")),
            sources=sources,
            related_dependencies=dependencies,
            architecture_insights=list(data.get("architecture_insights", [])),
            confidence=float(data.get("confidence", 0.9)),
        )

    def _filter_and_rank_chunks(
        self, chunks: List[RetrievedChunk], min_relevance: float
    ) -> List[RetrievedChunk]:
        """Filter out low-relevance chunks and rank by normalized similarity."""
        scored_chunks: List[tuple[float, RetrievedChunk]] = []
        for chunk in chunks:
            # If distance is provided, convert to similarity score in [0.0, 1.0]
            if chunk.distance is not None:
                sim = 1.0 / (1.0 + max(0.0, float(chunk.distance)))
            else:
                sim = 1.0

            if sim >= min_relevance:
                scored_chunks.append((sim, chunk))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored_chunks]

    def _build_source_references(
        self, chunks: List[RetrievedChunk]
    ) -> List[SourceReference]:
        refs: List[SourceReference] = []
        for c in chunks:
            sim = 1.0
            if c.distance is not None:
                sim = round(1.0 / (1.0 + max(0.0, float(c.distance))), 3)
            # Truncate snippet preview for reference
            snippet = "\n".join(c.content.splitlines()[:6])
            refs.append(
                SourceReference(
                    file_path=c.file_path,
                    start_line=c.start_line,
                    end_line=c.end_line,
                    language=c.language,
                    snippet=snippet,
                    relevance_score=sim,
                )
            )
        return refs

    def _extract_dependencies(self, chunks: List[RetrievedChunk]) -> List[str]:
        """Extract imported modules, symbols, and cross-file references."""
        deps: set[str] = set()
        for chunk in chunks:
            if chunk.language == "python":
                try:
                    tree = ast.parse(chunk.content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                deps.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                deps.add(node.module)
                except Exception:
                    # Fallback regex for partial chunks
                    for line in chunk.content.splitlines():
                        match = re.match(r"^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)", line)
                        if match:
                            deps.add(match.group(1))
            elif chunk.language in ("javascript", "typescript"):
                for line in chunk.content.splitlines():
                    match = re.search(r'(?:from|require\()\s*[\'"]([^\'"]+)[\'"]', line)
                    if match:
                        deps.add(match.group(1))
        return sorted(deps)

    def _format_chunks_for_llm(self, chunks: List[RetrievedChunk]) -> str:
        parts: List[str] = []
        for idx, c in enumerate(chunks, 1):
            parts.append(
                f"[Source {idx}] File: {c.file_path} (Lines {c.start_line}-{c.end_line}, Lang: {c.language}):\n"
                f"```\n{c.content}\n```"
            )
        return "\n\n".join(parts)

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise RagAgentError("Failed to parse valid JSON response from LLM")

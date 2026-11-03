"""Research / RAG agent.

This agent retrieves relevant repository context, extracts structured evidence,
then asks the LLM to produce a concise engineering answer grounded in the
retrieved snippets. The implementation intentionally stays dependency-light so
it can be used by the orchestrator and tests without a full vector-store setup.
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional

from backend.infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class RagAgentError(RuntimeError):
    """Base error for RAG agent failures."""


@dataclass(frozen=True)
class SourceReference:
    """Single source snippet used to ground an answer."""

    file_path: str
    start_line: int
    end_line: int
    relevance_score: float
    snippet: str = ""
    language: str = ""
    chunk_index: int = 0
    ranking_reasons: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchReport:
    """Structured repository research report."""

    query: str
    summary: str
    answer: str
    sources: List[SourceReference] = field(default_factory=list)
    related_dependencies: List[str] = field(default_factory=list)
    architecture_insights: List[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence_status: str = "available"
    warnings: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class RagAgent:
    """Retrieve code context and synthesize a repository-aware answer."""

    def __init__(
        self,
        *,
        llm_client: Optional[LLMClient] = None,
        retriever: Optional[Any] = None,
        top_k: int = 5,
    ) -> None:
        self._llm_client = llm_client
        self._retriever = retriever
        self._top_k = max(1, min(50, int(top_k)))

    def run(self, query: str) -> ResearchReport:
        """Backward-compatible alias to ``research()``."""
        return self.research(query)

    def research(self, query: str) -> ResearchReport:
        """Retrieve relevant repository content and answer the user query."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip():
            raise ValueError("query must be non-empty")

        if self._retriever is None:
            return self._unavailable_report(query, warning="No retriever is configured.")

        try:
            chunks = list(self._retriever.retrieve(query, top_k=self._top_k) or [])
        except Exception as exc:
            logger.warning("Repository retrieval failed: %s", exc)
            return self._unavailable_report(query, warning="Repository retrieval failed.")

        chunks = [
            chunk
            for chunk in chunks
            if isinstance(getattr(chunk, "file_path", None), str)
            and bool(getattr(chunk, "file_path", "").strip())
            and isinstance(getattr(chunk, "content", None), str)
            and bool(getattr(chunk, "content", "").strip())
        ]
        sources = self._build_sources(chunks)
        if not sources:
            return ResearchReport(
                query=query,
                summary="No relevant repository evidence was retrieved.",
                answer="I could not find repository evidence to answer this query. The answer is unknown from the indexed context.",
                sources=[],
                related_dependencies=[],
                architecture_insights=[],
                confidence=0.0,
                evidence_status="insufficient",
                warnings=["No valid source chunks were returned by retrieval."],
            )

        related_dependencies = self._extract_dependencies(chunks)

        llm_payload = self._build_llm_prompt(query, chunks)
        warnings: List[str] = []
        try:
            response_text = self._generate_response(llm_payload)
            parsed = self._parse_response(response_text)
        except Exception as exc:
            logger.warning("Research answer generation failed: %s", exc)
            parsed = {
                "summary": "Relevant repository evidence was retrieved, but answer synthesis failed.",
                "answer": "The retrieved source snippets are available, but a synthesized answer could not be generated.",
                "architecture_insights": [],
                "confidence": 0.0,
                "assumptions": [],
                "recommendations": [],
            }
            warnings.append("Answer synthesis failed; report contains retrieved evidence only.")

        return ResearchReport(
            query=query,
            summary=self._string_or_default(parsed.get("summary"), "No relevant summary available."),
            answer=self._compose_answer(query, self._string_or_default(parsed.get("answer"), ""), sources),
            sources=sources,
            related_dependencies=related_dependencies,
            architecture_insights=self._string_list(parsed.get("architecture_insights")),
            confidence=self._confidence(parsed.get("confidence")),
            evidence_status="available",
            warnings=warnings,
            assumptions=self._string_list(parsed.get("assumptions")),
            recommendations=self._string_list(parsed.get("recommendations")),
        )

    @staticmethod
    def _unavailable_report(query: str, warning: str) -> ResearchReport:
        return ResearchReport(
            query=query,
            summary="Repository context is currently unavailable.",
            answer="I could not retrieve repository evidence for this query. The answer is unknown from the indexed context.",
            evidence_status="unavailable",
            warnings=[warning],
        )

    @staticmethod
    def _string_or_default(value: Any, default: str) -> str:
        return value.strip() if isinstance(value, str) and value.strip() else default

    @staticmethod
    def _string_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError, OverflowError):
            return 0.0
        return max(0.0, min(1.0, confidence)) if math.isfinite(confidence) else 0.0

    def _generate_response(self, prompt: str) -> str:
        if self._llm_client is None:
            return json.dumps(
                {
                    "summary": "Repository context suggests a likely implementation path.",
                    "answer": "No LLM client is configured, so the answer is based on retrieved context only.",
                    "architecture_insights": [],
                    "confidence": 0.5,
                    "assumptions": [],
                    "recommendations": [],
                }
            )
        response = self._llm_client.generate(
            prompt,
            system_prompt=(
                "You are a repository research assistant. Treat retrieved snippets as the only repository evidence. "
                "Do not invent files, symbols, dependencies, callers, or architecture. Clearly label assumptions and state when evidence is insufficient."
            ),
        )
        if isinstance(response, str):
            return response.strip()
        return str(getattr(response, "text", "") or "").strip()

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        cleaned = response_text.strip()
        if not cleaned:
            return {
                "summary": "No relevant summary available.",
                "answer": "No applicable answer generated.",
                "architecture_insights": [],
                "confidence": 0.0,
                "assumptions": [],
                "recommendations": [],
            }
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        summary = cleaned[:200].strip()
        return {
            "summary": summary or "No relevant summary available.",
            "answer": cleaned,
            "architecture_insights": [],
            "confidence": 0.5,
        }

    def _build_sources(self, chunks: Iterable[Any]) -> List[SourceReference]:
        sources: List[SourceReference] = []
        for chunk in chunks:
            file_path = getattr(chunk, "file_path", "")
            content = getattr(chunk, "content", "")
            if not isinstance(file_path, str) or not file_path.strip() or not isinstance(content, str) or not content.strip():
                logger.warning("Skipping retrieved chunk without a valid path or content")
                continue
            start_line = self._safe_int(getattr(chunk, "start_line", 0))
            end_line = self._safe_int(getattr(chunk, "end_line", 0))
            relevance = self._confidence(getattr(chunk, "ranking_score", None))
            if relevance == 0.0:
                distance = self._confidence(1.0 - self._confidence(getattr(chunk, "distance", 0.0)))
                relevance = distance
            reasons = getattr(chunk, "ranking_reasons", ())
            sources.append(
                SourceReference(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    relevance_score=relevance,
                    snippet=content[:500],
                    language=str(getattr(chunk, "language", "") or ""),
                    chunk_index=self._safe_int(getattr(chunk, "chunk_index", 0)),
                    ranking_reasons=[reason for reason in reasons if isinstance(reason, str)] if isinstance(reasons, (list, tuple)) else [],
                )
            )
        return sources

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    def _extract_dependencies(self, chunks: Iterable[Any]) -> List[str]:
        deps: List[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            content = getattr(chunk, "content", "") or ""
            if not isinstance(content, str):
                continue
            for match in re.findall(r"(?:from\s+([A-Za-z0-9_\.]+)\s+import)|(?:import\s+([A-Za-z0-9_\.]+))", content):
                module = next((m for m in match if m), "")
                if module and module not in seen:
                    deps.append(module)
                    seen.add(module)
        return deps

    def _build_llm_prompt(self, query: str, chunks: Iterable[Any]) -> str:
        rendered_chunks = []
        for index, chunk in enumerate(chunks, start=1):
            score = self._confidence(getattr(chunk, "ranking_score", 0.0))
            reasons = getattr(chunk, "ranking_reasons", ())
            reason_text = ", ".join(reasons) if isinstance(reasons, (list, tuple)) else ""
            rendered_chunks.append(
                f"Source S{index}: {getattr(chunk, 'file_path', '')}"
                f" (lines {self._safe_int(getattr(chunk, 'start_line', 0))}-"
                f"{self._safe_int(getattr(chunk, 'end_line', 0))})\n"
                f"Ranking: {score:.2f}; reasons: {reason_text}\n"
                f"Content:\n{getattr(chunk, 'content', '')}"
            )
        chunk_text = "\n\n".join(rendered_chunks)
        return (
            "Answer using only repository facts supported by the evidence below. Cite source labels (for example S1) for repository claims. "
            "Do not invent files, symbols, dependencies, relationships, or line numbers. Distinguish recommendations from observed facts and say when evidence is insufficient.\n"
            f"Question: {query}\n\n"
            "Return valid JSON with keys: summary, answer, architecture_insights, confidence, assumptions, recommendations.\n"
            "If evidence is incomplete, state the limitation clearly.\n\n"
            f"Repository context:\n{chunk_text}"
        )

    def _compose_answer(self, query: str, answer: str, sources: List[SourceReference]) -> str:
        base = answer.strip() if answer.strip() else "I did not find a direct answer in the repository context."
        if not sources:
            return f"{base}"
        files = ", ".join(sorted({source.file_path for source in sources if source.file_path}))
        return f"{base} Relevant files include: {files}."


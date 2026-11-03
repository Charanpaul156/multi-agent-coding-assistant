"""Repository retriever.

Converts a natural-language query into relevant repository chunks. This is the
single entry point that agents / application layer use to obtain repository
context. It never returns raw vector-database objects; it returns `RetrievedChunk`
objects with clean, agent-safe fields.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from rag.embedder import Embedder
from rag.vector_store import StoredDocument, VectorStore

logger = logging.getLogger(__name__)


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "to",
    "use", "what", "where", "which", "with",
}
_CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".cpp", ".h"}
_DOC_EXTENSIONS = {".md", ".rst", ".txt"}


@dataclass(frozen=True)
class RetrievedChunk:
    """A repository chunk returned by the retriever.

    Attributes:
        content: The chunk text (safe for agents).
        file_path: Repository-relative POSIX path.
        start_line: 1-based inclusive start line.
        end_line: 1-based inclusive end line.
        language: Language label.
        repository: Repository identifier.
        chunk_index: Zero-based chunk index within the file.
        distance: Similarity/distance score (lower is more similar).
        metadata: Full metadata dict (as returned by the vector store).
    """

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    repository: str
    chunk_index: int
    distance: float | None = None
    metadata: dict = None  # type: ignore
    ranking_score: float = 0.0
    ranking_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.metadata is None:
            object.__setattr__(self, "metadata", {})


class RepositoryRetriever:
    """Retrieve relevant repository chunks for a natural-language query."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        top_k: int = 5,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._top_k = max(1, int(top_k))

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        repository: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k relevant chunks for ``query``.

        Args:
            query: Natural-language query.
            top_k: Overrides the default top-k if provided.
            repository: If set, restrict results to a given repository.

        Returns:
            A list of `RetrievedChunk` results, ordered by relevance.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        k = self._top_k if top_k is None else max(1, int(top_k))
        candidate_k = min(max(k * 3, k), 50)
        query_embedding = self._embedder.embed_query(query)

        metadata_filter = {"repository": repository} if repository else None
        results = self._vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=candidate_k,
            metadata_filter=metadata_filter,
        )

        query_terms = self._query_terms(query)
        ranked: list[RetrievedChunk] = []
        seen_content: set[tuple[str, str]] = set()
        for doc in results:
            try:
                chunk = self._to_retrieved_chunk(doc)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                logger.warning("Skipping malformed retrieved document: %s", exc)
                continue
            if not chunk.content.strip():
                continue

            content_key = (chunk.file_path, self._normalize_content(chunk.content))
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            ranked.append(self._rank_chunk(chunk, query_terms))

        ranked.sort(key=lambda chunk: (-chunk.ranking_score, chunk.file_path, chunk.start_line))
        return ranked[:k]

    @staticmethod
    def _query_terms(query: str) -> set[str]:
        normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", query).replace("_", " ")
        return {
            term.lower()
            for term in re.findall(r"[A-Za-z][A-Za-z0-9]*", normalized)
            if len(term) > 1 and term.lower() not in _STOP_WORDS
        }

    @staticmethod
    def _normalize_content(content: str) -> str:
        return " ".join(content.split()).casefold()

    def _rank_chunk(self, chunk: RetrievedChunk, query_terms: set[str]) -> RetrievedChunk:
        content_terms = self._query_terms(chunk.content)
        path_terms = self._query_terms(chunk.file_path.replace("/", " ").replace(".", " "))
        content_matches = query_terms & content_terms
        path_matches = query_terms & path_terms
        lexical_score = len(content_matches) / max(1, len(query_terms))
        path_score = len(path_matches) / max(1, len(query_terms))

        distance = chunk.distance
        try:
            numeric_distance = float(distance) if distance is not None else None
        except (TypeError, ValueError, OverflowError):
            numeric_distance = None
        if numeric_distance is not None and not math.isfinite(numeric_distance):
            numeric_distance = None
        semantic_score = (
            0.0
            if numeric_distance is None
            else max(0.0, min(1.0, 1.0 - numeric_distance))
        )
        extension = "." + chunk.file_path.rsplit(".", 1)[-1].lower() if "." in chunk.file_path else ""
        type_adjustment = 0.04 if extension in _CODE_EXTENSIONS else -0.04 if extension in _DOC_EXTENSIONS else 0.0
        score = max(0.0, min(1.0, 0.60 * semantic_score + 0.28 * lexical_score + 0.12 * path_score + type_adjustment))

        reasons: list[str] = []
        if numeric_distance is not None:
            reasons.append(f"vector similarity {semantic_score:.2f}")
        if content_matches:
            reasons.append("query terms in content: " + ", ".join(sorted(content_matches)))
        if path_matches:
            reasons.append("query terms in path: " + ", ".join(sorted(path_matches)))
        if type_adjustment:
            reasons.append("implementation source preferred" if type_adjustment > 0 else "documentation deprioritized")

        return RetrievedChunk(
            content=chunk.content,
            file_path=chunk.file_path,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            language=chunk.language,
            repository=chunk.repository,
            chunk_index=chunk.chunk_index,
            distance=chunk.distance,
            metadata=chunk.metadata,
            ranking_score=score,
            ranking_reasons=tuple(reasons),
        )

    def _to_retrieved_chunk(self, doc: StoredDocument) -> RetrievedChunk:
        metadata = getattr(doc, "metadata", None) or {}
        if not isinstance(metadata, dict):
            metadata = {}
        content = getattr(doc, "content", "")
        if not isinstance(content, str):
            raise TypeError("document content must be a string")
        file_path = metadata.get("file_path", "")
        language = metadata.get("language", "")
        repository = metadata.get("repository", "")
        return RetrievedChunk(
            content=content,
            file_path=file_path if isinstance(file_path, str) else "",
            start_line=self._safe_line(metadata.get("start_line")),
            end_line=self._safe_line(metadata.get("end_line")),
            language=language if isinstance(language, str) else "",
            repository=repository if isinstance(repository, str) else "",
            chunk_index=self._safe_line(metadata.get("chunk_index")),
            distance=self._safe_distance(getattr(doc, "distance", None)),
            metadata=metadata,
        )

    @staticmethod
    def _safe_line(value: object) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _safe_distance(value: object) -> float | None:
        try:
            distance = float(value) if value is not None else None
        except (TypeError, ValueError, OverflowError):
            return None
        return distance if distance is not None and math.isfinite(distance) else None

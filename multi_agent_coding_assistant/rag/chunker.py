"""Code chunker.

Splits source files into chunks that preserve useful context.

Primary strategy: line-based chunking with overlap. This is language-agnostic,
simple, and reliable for the first version.

For Python, we additionally attempt "function/class-aware" chunking: if the
file contains top-level function/class definitions, we split on those
boundaries so a chunk maps to a coherent unit. If the file is not parseable as
Python or has no definitions, we fall back to the line-based strategy.

Chunks carry metadata: repository, file_path, language, chunk_index,
start_line, end_line, and a file_hash.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path

from rag.file_loader import SourceFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodeChunk:
    """A single indexed chunk of a source file.

    Attributes:
        id: Unique chunk identifier (repository + file_path + chunk_index).
        repository: Repository identifier.
        file_path: Repository-relative POSIX path.
        language: Language label.
        chunk_index: Zero-based index of the chunk within its file.
        start_line: 1-based inclusive start line (within the file).
        end_line: 1-based inclusive end line (within the file).
        content: The chunk text.
        file_hash: SHA-256 of the full file content (parent).
    """

    id: str
    repository: str
    file_path: str
    language: str
    chunk_index: int
    start_line: int
    end_line: int
    content: str
    file_hash: str

    def metadata_dict(self) -> dict:
        """Return metadata suitable for the vector store."""
        return {
            "repository": self.repository,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_index": self.chunk_index,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "file_hash": self.file_hash,
        }


class CodeChunker:
    """Chunk a source file into context-preserving chunks."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be in [0, chunk_size)")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_file(self, source: SourceFile) -> list[CodeChunk]:
        """Split a single SourceFile into chunks."""
        lines = source.content.splitlines()

        # Prefer function/class-aware chunking for Python when possible.
        if source.language == "python":
            boundaries = self._python_boundaries(source.content)
            if boundaries:
                return self._chunk_by_boundaries(source, lines, boundaries)

        return self._chunk_by_lines(source, lines)

    # ------------------------------------------------------------------
    # Line-based chunking (fallback / generic)
    # ------------------------------------------------------------------

    def _chunk_by_lines(
        self,
        source: SourceFile,
        lines: list[str],
    ) -> list[CodeChunk]:
        if not lines:
            return []

        chunks: list[CodeChunk] = []
        step = max(1, self._chunk_size - self._chunk_overlap)
        total = len(lines)

        for chunk_index, start in enumerate(range(0, total, step)):
            end = min(start + self._chunk_size, total)
            content = "\n".join(lines[start:end])
            chunks.append(
                CodeChunk(
                    id=self._make_id(source, chunk_index),
                    repository=source.repository,
                    file_path=source.repo_path,
                    language=source.language,
                    chunk_index=chunk_index,
                    start_line=start + 1,
                    end_line=end,  # inclusive (1-based)
                    content=content,
                    file_hash=source.file_hash,
                )
            )
            if end >= total:
                break

        return chunks

    # ------------------------------------------------------------------
    # Python function/class-aware chunking
    # ------------------------------------------------------------------

    def _python_boundaries(self, content: str) -> list[int] | None:
        """Return 1-based line numbers where top-level defs/classes begin.

        Returns [] if the file has no top-level definitions (caller falls
        back to line-based chunking). Returns None if the file is not
        parseable as Python.
        """
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        boundaries: list[int] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Only consider top-level nodes to keep chunks coherent.
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # A node is top-level if its parent is the module.
                    parent = getattr(node, "body", None)
                    # We detect top-level by checking the node is directly in
                    # the module body.
                    for item in tree.body:
                        if item is node:
                            boundaries.append(node.lineno)
                            break
        boundaries = sorted(set(boundaries))
        return boundaries

    def _chunk_by_boundaries(
        self,
        source: SourceFile,
        lines: list[str],
        boundaries: list[int],
    ) -> list[CodeChunk]:
        """Chunk on top-level definition boundaries, merging small ones.

        Each chunk starts at a definition boundary. If the gap between two
        boundaries (or the header + first definition) exceeds chunk_size, we
        fall back to line-based chunking for that segment.
        """
        total = len(lines)
        if not lines:
            return []

        # Convert boundaries (1-based) to 0-based line indices.
        boundary_idx = [b - 1 for b in boundaries if 0 < b <= total]

        segments: list[tuple[int, int]] = []  # (start, end_exclusive)
        # Segment 0: header lines before the first definition.
        first = boundary_idx[0] if boundary_idx else 0
        if first > 0:
            segments.append((0, first))

        for i, start in enumerate(boundary_idx):
            end = boundary_idx[i + 1] if i + 1 < len(boundary_idx) else total
            # If a segment is too large, split it with the line-based method.
            if end - start > self._chunk_size:
                self._append_line_based_segment(
                    source, lines, start, end, segments
                )
            else:
                segments.append((start, end))

        # Build chunks from segments, merging consecutive very small segments.
        return self._segments_to_chunks(source, lines, segments)

    def _append_line_based_segment(
        self,
        source: SourceFile,
        lines: list[str],
        start: int,
        end: int,
        segments: list[tuple[int, int]],
    ) -> None:
        step = max(1, self._chunk_size - self._chunk_overlap)
        for s in range(start, end, step):
            e = min(s + self._chunk_size, end)
            segments.append((s, e))
            if e >= end:
                break

    def _segments_to_chunks(
        self,
        source: SourceFile,
        lines: list[str],
        segments: list[tuple[int, int]],
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        chunk_index = 0
        for start, end in segments:
            content = "\n".join(lines[start:end]).rstrip()
            if not content.strip():
                continue
            chunks.append(
                CodeChunk(
                    id=self._make_id(source, chunk_index),
                    repository=source.repository,
                    file_path=source.repo_path,
                    language=source.language,
                    chunk_index=chunk_index,
                    start_line=start + 1,
                    end_line=end,  # inclusive
                    content=content,
                    file_hash=source.file_hash,
                )
            )
            chunk_index += 1
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(source: SourceFile, chunk_index: int) -> str:
        return f"{source.repository}::{source.repo_path}::{chunk_index}"

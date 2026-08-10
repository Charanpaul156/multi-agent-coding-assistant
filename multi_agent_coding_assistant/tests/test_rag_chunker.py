"""Tests for CodeChunker."""

from __future__ import annotations

from rag.chunker import CodeChunk, CodeChunker
from rag.file_loader import SourceFile


def _source(content: str, language: str = "python", path: str = "app.py") -> SourceFile:
    line_count = content.count("\n") + 1
    return SourceFile(
        repository="repo",
        repo_path=path,
        abs_path=f"/tmp/{path}",
        language=language,
        content=content,
        line_count=line_count,
        file_hash="abc123",
    )


def test_empty_file_produces_no_chunks() -> None:
    chunker = CodeChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_file(_source(""))
    assert chunks == []


def test_line_based_chunking_splits_long_file() -> None:
    # Non-python file => line-based chunking.
    content = "\n".join(f"line{i}" for i in range(100))
    chunker = CodeChunker(chunk_size=30, chunk_overlap=5)
    chunks = chunker.chunk_file(_source(content, language="javascript", path="a.js"))
    assert len(chunks) > 1
    # Every chunk carries metadata.
    for chunk in chunks:
        assert chunk.repository == "repo"
        assert chunk.file_path == "a.js"
        assert chunk.language == "javascript"
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        assert chunk.file_hash == "abc123"
    # Chunk index is sequential.
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # First chunk starts at line 1.
    assert chunks[0].start_line == 1


def test_python_function_boundary_chunking() -> None:
    content = (
        "import os\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def sub(a, b):\n"
        "    return a - b\n"
        "\n"
        "class Calc:\n"
        "    def mul(self, a, b):\n"
        "        return a * b\n"
    )
    chunker = CodeChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_file(_source(content))
    assert len(chunks) >= 3
    chunks_by_content = {c.content: c for c in chunks}
    # Each definition should appear within some chunk.
    all_text = "\n".join(chunk.content for chunk in chunks)
    assert "def add(a, b):" in all_text
    assert "def sub(a, b):" in all_text
    assert "class Calc:" in all_text


def test_chunk_within_chunk_size() -> None:
    content = "\n".join(f"line{i}" for i in range(50))
    chunker = CodeChunker(chunk_size=20, chunk_overlap=0)
    chunks = chunker.chunk_file(_source(content, language="json", path="c.json"))
    for chunk in chunks:
        nlines = chunk.end_line - chunk.start_line + 1
        assert nlines <= 20


def test_chunk_overlap_validation() -> None:
    try:
        CodeChunker(chunk_size=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        CodeChunker(chunk_size=10, chunk_overlap=10)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_metadata_dict() -> None:
    chunk = CodeChunk(
        id="repo::app.py::0",
        repository="repo",
        file_path="app.py",
        language="python",
        chunk_index=0,
        start_line=1,
        end_line=5,
        content="def f(): pass\n",
        file_hash="hash",
    )
    meta = chunk.metadata_dict()
    assert meta["repository"] == "repo"
    assert meta["file_path"] == "app.py"
    assert meta["language"] == "python"
    assert meta["chunk_index"] == 0
    assert meta["start_line"] == 1
    assert meta["end_line"] == 5
    assert meta["file_hash"] == "hash"


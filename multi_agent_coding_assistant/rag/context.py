"""Context formatting for agents.

Turns retrieved repository chunks into a compact, agent-safe prompt block. The
RAG layer is responsible for formatting; agents never touch vector stores or
retrieval internals.
"""

from __future__ import annotations

from rag.retriever import RetrievedChunk


def format_retrieved_context(
    chunks: list[RetrievedChunk],
    *,
    max_chunks: int | None = None,
    max_chars_per_chunk: int | None = 2000,
) -> str:
    """Format a list of retrieved chunks into a readable context block.

    Args:
        chunks: Retrieved chunks ordered by relevance.
        max_chunks: Optional cap on the number of chunks included.
        max_chars_per_chunk: Optional per-chunk character cap.

    Returns:
        A formatted string suitable for inclusion in an agent prompt. Returns
        an empty string if ``chunks`` is empty.
    """
    if not chunks:
        return ""

    selected = chunks
    if max_chunks is not None and max_chunks > 0:
        selected = chunks[:max_chunks]

    blocks: list[str] = []
    blocks.append("The following are relevant sections from the repository:")

    for chunk in selected:
        content = chunk.content
        if max_chars_per_chunk and max_chars_per_chunk > 0:
            content = content[:max_chars_per_chunk]
        header = f"File: {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
        blocks.append(header)
        blocks.append("```")
        blocks.append(content)
        blocks.append("```")

    return "\n\n".join(blocks)

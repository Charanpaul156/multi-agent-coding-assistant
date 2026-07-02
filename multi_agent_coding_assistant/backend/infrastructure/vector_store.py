"""Vector store placeholder.

Prepared for future LangChain + ChromaDB integration.

No persistence, indexing, or retrieval logic is implemented at this stage.
"""

from __future__ import annotations


class VectorStore:
    """Vector store abstraction for future ChromaDB integration."""

    def add_texts(self, texts: list[str]) -> None:
        raise NotImplementedError("VectorStore not implemented yet")

    def query(self, text: str, top_k: int = 5) -> list[str]:
        raise NotImplementedError("VectorStore not implemented yet")


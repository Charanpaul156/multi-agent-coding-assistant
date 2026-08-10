"""Repository RAG package.

Public API:
    - RagConfig
    - RepositoryLoader, SourceFile
    - CodeChunker, CodeChunk
    - Embedder, DeterministicEmbedder
    - VectorStore, StoredDocument
    - RepositoryRetriever, RetrievedChunk
    - RepositoryRagPipeline
    - format_retrieved_context
"""

from rag.config import RagConfig
from rag.chunker import CodeChunk, CodeChunker
from rag.context import format_retrieved_context
from rag.embedder import DeterministicEmbedder, Embedder
from rag.file_loader import RepositoryLoader, SourceFile
from rag.pipeline import IndexResult, RepositoryRagPipeline, SearchResult
from rag.retriever import RepositoryRetriever, RetrievedChunk
from rag.vector_store import StoredDocument, VectorStore

__all__ = [
    "RagConfig",
    "RepositoryLoader",
    "SourceFile",
    "CodeChunker",
    "CodeChunk",
    "Embedder",
    "DeterministicEmbedder",
    "VectorStore",
    "StoredDocument",
    "RepositoryRetriever",
    "RetrievedChunk",
    "RepositoryRagPipeline",
    "IndexResult",
    "SearchResult",
    "format_retrieved_context",
]

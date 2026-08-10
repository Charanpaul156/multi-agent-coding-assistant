"""RAG configuration.

Centralized configuration for the Repository RAG system.

All tunable values live here (or in `config/settings.py`) so they are never
scattered across the codebase. The `RagConfig` dataclass is framework-agnostic
and can be constructed from `config.settings.Settings` in the DI layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# File filtering
# ---------------------------------------------------------------------------

# Supported source/config/doc file extensions (case-insensitive).
DEFAULT_SUPPORTED_EXTENSIONS: set[str] = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".java",
    ".cpp",
    ".h",
    ".html",
    ".css",
    ".md",
    ".json",
    ".yaml",
    ".yml",
}

# Directory names that are always excluded during traversal.
DEFAULT_EXCLUDED_DIRS: set[str] = {
    ".git",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
    ".tox",
    ".eggs",
    ".ipynb_checkpoints",
}

# File names / patterns that are always excluded (secrets, credentials, etc.).
# Matched against the basename (case-insensitive).
DEFAULT_EXCLUDED_FILES: set[str] = {
    ".env",
    ".env.example",
    ".env.local",
    "id_rsa",
    "id_rsa.pub",
    "id_ed25519",
    "id_ecdsa",
    "credentials",
    "credential",
    "secrets",
    "secret",
    "token",
    "tokens",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "*.crt",
    "*.cer",
    "*.der",
    "service-account.json",
    "firebase-adminsdk.json",
}

# Binary file extensions always excluded.
DEFAULT_BINARY_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".svg",
    ".webp",
    ".tiff",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".7z",
    ".rar",
    ".exe",
    ".dll",
    ".so",
    ".o",
    ".a",
    ".lib",
    ".dylib",
    ".bin",
    ".dat",
    ".class",
    ".jar",
    ".war",
    ".pyc",
    ".pyo",
    ".whl",
    ".ttf",
    ".otf",
    ".woff",
    ".woff2",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wav",
    ".flac",
    ".lock",
}


@dataclass(frozen=True)
class RagConfig:
    """Immutable RAG configuration with sensible defaults.

    Attributes:
        embedding_model: Name of the sentence-transformers model to load.
        embedding_dimension: Dimensionality of the embedding vectors.
        vector_store_path: Directory where the ChromaDB persistence lives.
        chunk_size: Approximate number of tokens/lines per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        top_k: Default number of results returned by retrieval.
        supported_extensions: File extensions eligible for indexing.
        excluded_dirs: Directory basenames always skipped.
        excluded_files: File basenames/globs always skipped.
        binary_extensions: Extensions treated as binary and skipped.
        allowed_repository_roots: Absolute directories that are allowed to be
            indexed. An empty tuple means indexing is disabled (safest).
    """

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dimension: int = DEFAULT_EMBEDDING_DIMENSION
    vector_store_path: str = ".rag_chroma"
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    top_k: int = DEFAULT_TOP_K
    supported_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_SUPPORTED_EXTENSIONS)
    )
    excluded_dirs: set[str] = field(
        default_factory=lambda: set(DEFAULT_EXCLUDED_DIRS)
    )
    excluded_files: set[str] = field(
        default_factory=lambda: set(DEFAULT_EXCLUDED_FILES)
    )
    binary_extensions: set[str] = field(
        default_factory=lambda: set(DEFAULT_BINARY_EXTENSIONS)
    )
    allowed_repository_roots: tuple[str, ...] = ()

    # Derived helpers -------------------------------------------------------

    @property
    def supported_extensions_lower(self) -> set[str]:
        return {ext.lower() for ext in self.supported_extensions}

    @property
    def binary_extensions_lower(self) -> set[str]:
        return {ext.lower() for ext in self.binary_extensions}

    def is_supported_file(self, path: Path) -> bool:
        """Return True if ``path`` has a supported source extension."""
        return path.suffix.lower() in self.supported_extensions_lower

    def is_binary_file(self, path: Path) -> bool:
        """Return True if ``path`` has a known binary extension."""
        return path.suffix.lower() in self.binary_extensions_lower

    def is_excluded_dir(self, name: str) -> bool:
        """Return True if a directory basename is excluded."""
        return name in self.excluded_dirs

    def is_excluded_file(self, name: str) -> bool:
        """Return True if a file basename matches an exclusion.

        Supports exact names and simple ``*.ext`` glob patterns.
        """
        lower = name.lower()
        if lower in {f.lower() for f in self.excluded_files}:
            return True
        for pattern in self.excluded_files:
            if pattern.startswith("*.") and lower.endswith(pattern[1:].lower()):
                return True
        return False

    def is_within_allowed_root(self, path: Path) -> bool:
        """Return True if ``path`` is inside one of the allowed roots.

        Uses ``Path.resolve()`` to neutralize ``..`` and symlinks.
        """
        if not self.allowed_repository_roots:
            return False
        try:
            resolved = path.resolve()
        except OSError:
            return False
        for root in self.allowed_repository_roots:
            try:
                root_path = Path(root).resolve()
            except OSError:
                continue
            try:
                resolved.relative_to(root_path)
                return True
            except ValueError:
                continue
        return False

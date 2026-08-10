"""Repository loader.

Responsible for safely traversing a repository directory and reading the
supported source files while strictly excluding ignored/secret/binary files.

Security:
    - The repository root must resolve inside one of the configured allowed
      roots, otherwise loading is rejected.
    - Symlinks that escape the allowed root are not followed.
    - `.env`, secrets, credentials, keys, `.git`, virtual environments,
      `node_modules`, build artifacts, caches, binary files, etc. are skipped.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from rag.config import RagConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceFile:
    """A single indexed source file read from disk.

    Attributes:
        repository: Repository identifier (e.g. the repository root name).
        repo_path: Path relative to the repository root (POSIX separators).
        abs_path: Absolute path on disk (kept for engineering/tooling only;
            never exposed to agents or the API).
        language: Programming language derived from the file extension.
        content: Full text content of the file.
        line_count: Number of lines in the file.
        file_hash: SHA-256 hash of the file content (for future incremental
            indexing and duplicate detection).
    """

    repository: str
    repo_path: str
    abs_path: str
    language: str
    content: str
    line_count: int
    file_hash: str


# Map of extension -> language label.
EXTENSION_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".h": "c_header",
    ".html": "html",
    ".css": "css",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
}


class RepositoryError(RuntimeError):
    """Base error for repository loading failures."""


class RepositoryNotFoundError(RepositoryError):
    """Raised when the repository path does not exist or is not a directory."""


class RepositoryNotAllowedError(RepositoryError):
    """Raised when the repository path is outside the allowed roots."""


class RepositoryLoader:
    """Load supported source files from a repository directory.

    The loader is deterministic and has no external side effects besides
    reading files. It never opens the vector store or the embedding model.
    """

    def __init__(self, config: RagConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_repository(self, path: str | Path) -> list[SourceFile]:
        """Traverse ``path`` and return all supported source files.

        Raises:
            RepositoryNotFoundError: if ``path`` is missing or not a dir.
            RepositoryNotAllowedError: if ``path`` is outside allowed roots.
        """
        root = self._validate_root(path)

        files: list[SourceFile] = []
        repository_id = root.name

        for dirpath, dirnames, filenames in os.walk(root, topdown=True):
            dirnames[:] = self._filter_dirs(dirnames)
            for filename in filenames:
                full = Path(dirpath) / filename
                if not self._should_include(full):
                    continue
                try:
                    source = self._read_file(root, repository_id, full)
                except OSError as exc:
                    logger.warning("Skipping unreadable file %s: %s", full, exc)
                    continue
                if source is not None:
                    files.append(source)

        logger.info(
            "RepositoryLoader: indexed %d files from %s",
            len(files),
            root,
        )
        return files

    # ------------------------------------------------------------------
    # Root validation
    # ------------------------------------------------------------------

    def _validate_root(self, path: str | Path) -> Path:
        try:
            root = Path(path).expanduser().resolve()
        except OSError as exc:
            raise RepositoryNotFoundError(f"Invalid path: {path}") from exc

        if not root.exists():
            raise RepositoryNotFoundError(f"Repository path does not exist: {path}")
        if not root.is_dir():
            raise RepositoryNotAllowedError(
                f"Repository path is not a directory: {path}"
            )

        if not self._config.allowed_repository_roots:
            raise RepositoryNotAllowedError(
                "No allowed repository roots configured; indexing is disabled."
            )

        if not self._config.is_within_allowed_root(root):
            raise RepositoryNotAllowedError(
                f"Repository path is outside the allowed roots: {path}"
            )

        return root

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_dirs(self, dirnames: list[str]) -> list[str]:
        """Remove excluded directory basenames in-place (mutates the list)."""
        kept: list[str] = []
        for name in dirnames:
            if self._config.is_excluded_dir(name):
                logger.debug("Skipping excluded directory: %s", name)
                continue
            kept.append(name)
        return kept

    def _should_include(self, path: Path) -> bool:
        if self._config.is_excluded_file(path.name):
            logger.debug("Skipping excluded file: %s", path)
            return False
        if self._config.is_binary_file(path):
            logger.debug("Skipping binary file: %s", path)
            return False
        if not self._config.is_supported_file(path):
            logger.debug("Skipping unsupported file: %s", path)
            return False
        # Reject symlinks that escape the allowed root (safety).
        if path.is_symlink():
            try:
                resolved = path.resolve()
            except OSError:
                return False
            if not self._config.is_within_allowed_root(resolved):
                logger.warning("Skipping symlink escaping allowed root: %s", path)
                return False
        return True

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def _read_file(
        self,
        root: Path,
        repository_id: str,
        full: Path,
    ) -> SourceFile | None:
        # Guard against very large files to avoid loading huge binaries / data.
        try:
            if full.stat().st_size > 1_000_000:  # 1 MB cap
                logger.debug("Skipping large file (>1MB): %s", full)
                return None
        except OSError:
            return None

        try:
            raw = full.read_bytes()
        except OSError:
            return None

        # Decode as UTF-8; fall back to a lenient decode.
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = raw.decode("utf-8", errors="replace")
            except Exception:
                logger.debug("Skipping undecodable file: %s", full)
                return None

        # Secondary binary sniff: reject if content has many null bytes.
        if b"\x00" in raw[:4096]:
            logger.debug("Skipping binary file (null bytes): %s", full)
            return None

        rel = full.relative_to(root)
        repo_path = rel.as_posix()
        file_hash = hashlib.sha256(raw).hexdigest()
        # Count logical lines; a single trailing newline should not add an
        # extra empty line.
        stripped_content = content.rstrip("\r\n")
        line_count = stripped_content.count(chr(10)) + 1 if stripped_content else 0

        extension = full.suffix.lower()
        language = EXTENSION_LANGUAGE.get(extension, extension.lstrip(".") or "plain")

        return SourceFile(
            repository=repository_id,
            repo_path=repo_path,
            abs_path=str(full),
            language=language,
            content=content,
            line_count=line_count,
            file_hash=file_hash,
        )

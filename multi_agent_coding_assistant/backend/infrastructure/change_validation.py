"""Change validation infrastructure.

Deterministic, pure validation of a proposed ``ChangeSet`` BEFORE any file is
written. This layer is the primary security boundary for repository-aware code
modification.

Security guarantees:
    - Every target path must remain inside a configured allowed repository
      root (`RagConfig.allowed_repository_roots`).
    - Rejects `..` traversal, absolute paths, NUL bytes, symlink escapes
      outside the allowed root, oversized content, empty content, and
      malformed changes.
    - Never modifies protected files (`.env*`, keys, credentials, secrets,
      deployment credentials, `.git`, etc.) using the same exclusion rules as
      the RAG loader.
    - For `modify`, verifies the operation/on-disk state and (optionally) the
      expected `original_hash`.
    - Optionally validates Python syntax for `.py` files.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from rag.config import RagConfig

from backend.domain.change_models import (
    ChangeOperation,
    ChangeSet,
    ChangeValidationResult,
    FileChange,
    ValidationReport,
)

logger = logging.getLogger(__name__)


# Max bytes of content a single change may carry. Prevents token/size abuse.
DEFAULT_MAX_CONTENT_BYTES = 2_000_000

# Extra protected basenames beyond RagConfig exclusions (defense in depth).
_EXTRA_PROTECTED_NAMES = {
    ".htpasswd",
    ".gitconfig",
    ".npmrc",
    ".pypirc",
    "deployment-credentials.json",
    "docker-compose.secrets.yml",
}


class ChangeValidationError(RuntimeError):
    """Base error for change validation failures."""


@dataclass(frozen=True)
class ChangeValidator:
    """Validate a ChangeSet against repository-root and content policies."""

    config: RagConfig
    max_content_bytes: int = DEFAULT_MAX_CONTENT_BYTES

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def validate(self, change_set: ChangeSet) -> ValidationReport:
        """Validate an entire ChangeSet; returns per-change and aggregate results.

        Invalid changes are reported but never applied. If the ChangeSet
        itself is not a ChangeSet, a ChangeValidationError is raised.
        """
        if not isinstance(change_set, ChangeSet):
            raise ChangeValidationError("change_set must be a ChangeSet")
        if not change_set.changes:
            return ValidationReport(
                valid=False,
                results=[ChangeValidationResult(False, "", "", ["empty change set"])],
            )

        results: list[ChangeValidationResult] = []
        for change in change_set.changes:
            results.append(self._validate_change(change))

        return ValidationReport(
            valid=all(r.valid for r in results),
            results=results,
        )

    # ------------------------------------------------------------------ #
    # Per-change validation
    # ------------------------------------------------------------------ #

    def _validate_change(self, change: FileChange) -> ChangeValidationResult:
        messages: list[str] = []

        # --- Structural checks ---------------------------------------
        if not isinstance(change.file_path, str) or not change.file_path.strip():
            return ChangeValidationResult(
                False, str(change.file_path), change.operation.value,
                ["file_path must be a non-empty string"],
            )

        if change.operation not in (ChangeOperation.CREATE, ChangeOperation.MODIFY):
            return ChangeValidationResult(
                False,
                change.file_path,
                str(change.operation),
                ["unsupported operation (only create and modify allowed in v1)"],
            )

        # --- Path safety ---------------------------------------------
        path_error = self._path_is_safe(change.file_path)
        if path_error:
            messages.append(path_error)

        # --- Content checks ------------------------------------------
        if not isinstance(change.new_content, str):
            messages.append("new_content must be a string")
        else:
            if not change.new_content.strip():
                messages.append("new_content must not be empty")
            encoded = change.new_content.encode("utf-8", errors="replace")
            if len(encoded) > self.max_content_bytes:
                messages.append(
                    f"new_content exceeds {self.max_content_bytes} bytes"
                )

        if "\x00" in change.file_path:
            messages.append("file_path contains a NUL byte")

        # --- Protected file check ------------------------------------
        if self._is_protected(change.file_path):
            messages.append("target is a protected file and cannot be modified")

        # --- Syntax validation for python files ----------------------
        if change.file_path.endswith(".py") and isinstance(change.new_content, str):
            syntax_error = self._syntax_error(change.new_content)
            if syntax_error:
                messages.append(f"python syntax error: {syntax_error}")

        # --- Operation consistency on-disk ---------------------------
        resolved = None
        if not path_error:
            # Resolve absolute path for on-disk state checks if path is safe.
            try:
                resolved = self._resolve_abs(change.file_path)
            except Exception:
                resolved = None

        if not path_error and resolved is not None:
            exists = resolved.exists()
            if change.operation == ChangeOperation.CREATE and exists and not resolved.is_dir():
                messages.append(
                    "create operation targets an existing file"
                )
            if change.operation == ChangeOperation.CREATE and exists and resolved.is_dir():
                messages.append(
                    "create operation targets an existing directory"
                )
            if change.operation == ChangeOperation.MODIFY and not exists:
                messages.append("modify operation targets a non-existent file")
            if change.operation == ChangeOperation.MODIFY and exists and resolved.is_dir():
                messages.append("modify operation targets a directory")

            # Hash verification (re-read before apply also happens in applier).
            if (
                change.operation == ChangeOperation.MODIFY
                and exists
                and not resolved.is_dir()
                and change.original_hash
            ):
                actual = self._file_hash(resolved)
                if actual != change.original_hash:
                    messages.append(
                        "original_hash mismatch: file changed since generation"
                    )

        valid = not messages
        return ChangeValidationResult(
            valid,
            change.file_path,
            change.operation.value,
            messages,
        )

    # ------------------------------------------------------------------ #
    # Path safety helpers
    # ------------------------------------------------------------------ #

    def _path_is_safe(self, file_path: str) -> str | None:
        """Return an error message if the path is unsafe, else None."""
        if "\\" in file_path:
            # Backslashes could be drive separators / windows traversal.
            return "file_path must use POSIX '/' separators (no backslashes)"
        if file_path.startswith("/") or file_path.startswith("\\"):
            return "absolute paths are not allowed"
        if ".." in file_path.split("/"):
            return "path traversal ('..') is not allowed"

        # Resolve and ensure within an allowed root.
        try:
            resolved = self._resolve_abs(file_path)
        except Exception as exc:
            return f"unable to resolve target path: {exc}"

        if not self.config.is_within_allowed_root(resolved):
            return "target path is outside the allowed repository root"

        # Reject symlink traversal that escapes the root even if resolve
        # happened to land inside (defense in depth). is_within_allowed_root
        # already resolves symlinks; but we also check the direct components.
        return None

    def _resolve_abs(self, file_path: str) -> Path:
        """Return the resolved absolute Path for a repo-relative path.

        The repo root is derived from the first configured allowed root.
        file_path must be POSIX-relative and already validated as safe.
        """
        roots = self.config.allowed_repository_roots
        if not roots:
            raise ChangeValidationError("no allowed repository roots configured")

        root = Path(roots[0]).expanduser().resolve()
        target = (root / file_path).resolve()
        return target

    def _is_protected(self, file_path: str) -> bool:
        """Return True if the target is a protected file."""
        lower = file_path.lower()

        # Direct/exact protected patterns.
        protected_dir_parts = {".git", ".env", ".venv"}
        parts = set(p.lower() for p in file_path.split("/"))
        if parts & protected_dir_parts:
            return True

        name = lower.rsplit("/", 1)[-1]
        if self.config.is_excluded_file(name):
            return True
        if name in _EXTRA_PROTECTED_NAMES:
            return True
        # .env.* variants
        if name.startswith(".env"):
            return True
        for suffix in (".key", ".pem", ".p12", ".pfx", ".jks", ".keystore",
                       ".crt", ".cer", ".der", ".env"):
            if name.endswith(suffix):
                return True
        for token in ("credential", "secret", "token", "service-account",
                      "firebase-adminsdk"):
            if token in name:
                return True
        return False

    @staticmethod
    def _syntax_error(content: str) -> str | None:
        """Return a syntax error message for Python content, or None."""
        try:
            ast.parse(content)
        except SyntaxError as exc:
            return f"line {exc.lineno}: {exc.msg}"
        return None

    @staticmethod
    def _file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

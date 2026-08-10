"""Domain models for repository-aware code modification.

Framework-agnostic dataclasses. No FastAPI/Pydantic imports.

These types represent the structured change representation produced by agents
and consumed by the validation / application / rollback layers. Agents must
never write files directly; they only produce ``ChangeSet`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ChangeOperation(str, Enum):
    """Supported file-change operations.

    ``delete`` is intentionally NOT supported in version 1. Deletion is
    destructive and offers the lowest safety margin; only ``create`` and
    ``modify`` are allowed.
    """

    CREATE = "create"
    MODIFY = "modify"


@dataclass(frozen=True)
class FileChange:
    """A single proposed file change.

    Attributes:
        file_path: Repository-relative POSIX path (e.g. ``backend/auth/service.py``).
        operation: ``create`` or ``modify``.
        new_content: Complete new file content (full-file strategy for v1).
        original_content: Optional original content (primarily for ``modify``).
        original_hash: Optional SHA-256 of the expected current content. When
            provided it is verified immediately before applying; a mismatch
            rejects the transaction.
        description: Human-readable description of the change.
    """

    file_path: str
    operation: ChangeOperation
    new_content: str
    original_content: str | None = None
    original_hash: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ChangeSet:
    """A set of proposed changes produced by an agent."""

    changes: list[FileChange] = field(default_factory=list)
    summary: str = ""


@dataclass(frozen=True)
class ChangeValidationResult:
    """Result of validating a single change."""

    valid: bool
    file_path: str
    operation: str
    messages: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate result of validating a ChangeSet."""

    valid: bool
    results: list[ChangeValidationResult] = field(default_factory=list)


@dataclass(frozen=True)
class DiffEntry:
    """A human-readable unified diff for a single file."""

    file_path: str
    operation: str
    diff_text: str


@dataclass(frozen=True)
class RollbackRecord:
    """Enough information to restore a file to its pre-transaction state.

    ``file_path`` is the absolute path on disk. For created files
    ``was_created`` is True and ``original_content``/``original_hash`` are
    None (restore means removing the file).
    """

    file_path: str
    original_content: str | None
    original_hash: str | None
    was_created: bool


@dataclass(frozen=True)
class ApplicationResult:
    """Result of applying a validated ChangeSet."""

    success: bool
    applied_files: list[str] = field(default_factory=list)
    rollback_records: list[RollbackRecord] = field(default_factory=list)
    dry_run: bool = False
    error: str | None = None

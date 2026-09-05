"""Human-readable unified diff generation.

Pure, deterministic helpers that turn old/new file content into a unified
diff suitable for review in the API and Streamlit UI. Frameworks-agnostic.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Iterable

from backend.domain.change_models import FileChange


@dataclass(frozen=True)
class DiffGenerator:
    """Generate unified, reviewable diffs without modifying the repository."""

    def generate(
        self,
        change: FileChange,
        *,
        original_content: str | None = None,
    ) -> str:
        """Generate a diff for one change.

        ``original_content`` can be supplied by the caller when the previous
        content is held outside the change model. For a create operation, the
        original content is always treated as empty.
        """
        old_content = (
            ""
            if change.operation.value == "create"
            else original_content
            if original_content is not None
            else getattr(change, "original_content", None)
        )
        return generate_diff(change.file_path, old_content, change.new_content)

    def generate_many(
        self,
        changes: Iterable[FileChange],
        *,
        original_contents: dict[str, str | None] | None = None,
    ) -> list[str]:
        """Generate one unified diff per proposed file change."""
        originals = original_contents or {}
        return [
            self.generate(change, original_content=originals.get(change.file_path))
            for change in changes
        ]


def generate_diff(
    file_path: str,
    old_content: str | None,
    new_content: str | None,
) -> str:
    """Return a unified diff string for a single file.

    Args:
        file_path: Repository-relative POSIX path used in the diff header.
        old_content: Original content (empty/None for a create).
        new_content: New content.

    Returns:
        A unified diff string beginning with ``--- <file_path>`` and
        ``+++ <file_path>`` lines.
    """
    old_lines = (old_content or "").splitlines(keepends=True)
    new_lines = (new_content or "").splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=file_path,
        tofile=file_path,
    )
    return "".join(diff)


def diff_for_change(change: FileChange) -> str:
    """Generate a diff for a single FileChange object."""
    return DiffGenerator().generate(change)

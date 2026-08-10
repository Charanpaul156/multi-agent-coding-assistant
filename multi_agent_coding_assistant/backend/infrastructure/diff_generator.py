"""Human-readable unified diff generation.

Pure, deterministic helpers that turn old/new file content into a unified
diff suitable for review in the API and Streamlit UI. Frameworks-agnostic.
"""

from __future__ import annotations

import difflib

from backend.domain.change_models import FileChange


def generate_diff(
    file_path: str,
    operation: str,
    old_content: str | None,
    new_content: str | None,
) -> str:
    """Return a unified diff string for a single file.

    Args:
        file_path: Repository-relative POSIX path used in the diff header.
        operation: ``create`` or ``modify`` (informational only).
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
    old_content = change.original_content or ""
    return generate_diff(
        change.file_path,
        change.operation.value,
        old_content,
        change.new_content,
    )

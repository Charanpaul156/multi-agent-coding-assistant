"""Application boundary for read-only Git workspace inspection."""

from __future__ import annotations

from pathlib import Path

from backend.infrastructure.git_service import (
    GitCommitInfo,
    GitService,
    GitStatusResult,
)


class GitWorkspaceUseCases:
    """Delegate the four approved inspection operations to GitService."""

    def __init__(self, git_service: GitService) -> None:
        self._git_service = git_service

    def get_status(self, repository_path: str | Path) -> GitStatusResult:
        return self._git_service.get_status(repository_path)

    def get_current_branch(self, repository_path: str | Path) -> str:
        return self._git_service.get_current_branch(repository_path)

    def get_diff(
        self,
        repository_path: str | Path,
        *,
        staged: bool = False,
        max_bytes: int,
    ) -> str:
        return self._git_service.get_diff(
            repository_path, staged=staged, max_bytes=max_bytes
        )

    def get_commit_history(
        self, repository_path: str | Path, max_count: int
    ) -> list[GitCommitInfo]:
        return self._git_service.get_commit_history(repository_path, max_count)

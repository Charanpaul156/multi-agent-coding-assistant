"""Bounded, read-only Git inspection for configured repository roots."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from rag.config import RagConfig

logger = logging.getLogger(__name__)

STATUS_OUTPUT_LIMIT = 1_048_576
DIFF_OUTPUT_LIMIT = 1_048_576
HISTORY_OUTPUT_LIMIT = 262_144
STDERR_OUTPUT_LIMIT = 65_536
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_DIFF_BYTES = 1_048_576
MAX_HISTORY_COUNT = 100


class GitServiceError(RuntimeError):
    """Safe, classified error from a Git workspace inspection."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class GitStatusEntry:
    """One repository-relative working-tree or index change."""

    path: str
    index_status: str
    worktree_status: str
    kind: str = "tracked"
    original_path: str | None = None


@dataclass(frozen=True)
class GitStatusResult:
    """Current branch and structured working-tree status."""

    branch: str
    is_clean: bool
    entries: list[GitStatusEntry]


@dataclass(frozen=True)
class GitCommitInfo:
    """Recent commit metadata."""

    hash: str
    author: str
    date: str
    message: str


class GitService:
    """Expose only status, branch, diff, and bounded history inspection."""

    def __init__(
        self,
        config: RagConfig,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._config = config
        self._timeout_seconds = timeout_seconds

    def get_current_branch(self, repository_path: str | Path) -> str:
        repo = self._validated_repository(repository_path)
        output = self._run_git(repo, ["rev-parse", "--abbrev-ref", "HEAD"], 1024)
        branch = output.decode("utf-8", errors="replace").strip()
        if not branch or "\x00" in branch:
            raise GitServiceError("invalid_git_output")
        return branch

    def get_status(self, repository_path: str | Path) -> GitStatusResult:
        repo = self._validated_repository(repository_path)
        output = self._run_git(
            repo,
            [
                "status",
                "--porcelain=v2",
                "-z",
                "--untracked-files=all",
                "--no-ahead-behind",
            ],
            STATUS_OUTPUT_LIMIT,
        )
        entries = self._parse_status(output)
        branch = self.get_current_branch(repo)
        return GitStatusResult(branch=branch, is_clean=not entries, entries=entries)

    def get_diff(
        self,
        repository_path: str | Path,
        *,
        staged: bool = False,
        max_bytes: int = MAX_DIFF_BYTES,
    ) -> str:
        if not 1 <= max_bytes <= MAX_DIFF_BYTES:
            raise GitServiceError("invalid_request")
        repo = self._validated_repository(repository_path)
        self._validate_repository_configuration(repo, reject_filters=True)
        args = ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--no-relative"]
        if staged:
            args.append("--cached")
        output = self._run_git(repo, args, max_bytes)
        return output.decode("utf-8", errors="replace")

    def get_commit_history(
        self, repository_path: str | Path, max_count: int = 10
    ) -> list[GitCommitInfo]:
        if not 1 <= max_count <= MAX_HISTORY_COUNT:
            raise GitServiceError("invalid_request")
        repo = self._validated_repository(repository_path)
        fmt = "%H%x00%an%x00%aI%x00%s%x00"
        output = self._run_git(
            repo,
            [
                "log",
                f"--max-count={max_count}",
                f"--format={fmt}",
                "--no-decorate",
                "--no-color",
            ],
            HISTORY_OUTPUT_LIMIT,
        )
        values = output.split(b"\x00")
        commits: list[GitCommitInfo] = []
        for offset in range(0, len(values) - 1, 4):
            fields = values[offset : offset + 4]
            if len(fields) < 4 or not any(fields):
                continue
            if not all(fields[:3]):
                raise GitServiceError("invalid_git_output")
            commits.append(
                GitCommitInfo(
                    hash=fields[0].decode("utf-8", errors="replace"),
                    author=fields[1].decode("utf-8", errors="replace"),
                    date=fields[2].decode("utf-8", errors="replace"),
                    message=fields[3].decode("utf-8", errors="replace"),
                )
            )
        return commits

    def _validated_repository(self, repository_path: str | Path) -> Path:
        if not self._config.allowed_repository_roots:
            raise GitServiceError("roots_not_configured")
        raw_path = os.fspath(repository_path)
        if not raw_path or not raw_path.strip():
            raise GitServiceError("invalid_path")
        if any(part == ".." for part in raw_path.replace("\\", "/").split("/")):
            raise GitServiceError("invalid_path")
        try:
            candidate = Path(raw_path).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError):
            raise GitServiceError("repository_unavailable") from None
        if not candidate.is_dir() or not self._config.is_within_allowed_root(candidate):
            raise GitServiceError("repository_unavailable")

        # Search only within the configured root boundary. A .git file denotes
        # a linked worktree and is explicitly unsupported; symlink markers fail closed.
        current = candidate
        while self._config.is_within_allowed_root(current):
            marker = current / ".git"
            if marker.is_symlink():
                raise GitServiceError("repository_unavailable")
            if getattr(marker, "is_junction", lambda: False)():
                raise GitServiceError("repository_unavailable")
            if marker.exists():
                if marker.is_file():
                    raise GitServiceError("worktree_unsupported")
                if marker.is_dir():
                    self._validate_repository_configuration(
                        current, reject_filters=False
                    )
                    return current
                raise GitServiceError("repository_unavailable")
            parent = current.parent
            if parent == current:
                break
            current = parent
        raise GitServiceError("repository_unavailable")

    @staticmethod
    def _safe_environment() -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.upper().startswith("GIT_")
        }
        env.update(
            {
                "GIT_OPTIONAL_LOCKS": "0",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0",
                "GIT_ATTR_NOSYSTEM": "1",
                "GIT_PAGER": "cat",
                "PAGER": "cat",
            }
        )
        return env

    @staticmethod
    def _validate_repository_configuration(
        repository: Path, *, reject_filters: bool
    ) -> None:
        """Reject config includes and, for diffs, external filter drivers."""
        config_path = repository / ".git" / "config"
        try:
            if config_path.is_symlink() or getattr(
                config_path, "is_junction", lambda: False
            )():
                raise GitServiceError("unsafe_git_configuration")
            if not config_path.exists():
                return
            if config_path.stat().st_size > 1_048_576:
                raise GitServiceError("unsafe_git_configuration")
            config_text = config_path.read_text(encoding="utf-8", errors="replace")
        except GitServiceError:
            raise
        except OSError:
            raise GitServiceError("repository_unavailable") from None

        section_pattern = re.compile(r"^\s*\[\s*([^\s\]]+)", re.IGNORECASE)
        for line in config_text.splitlines():
            match = section_pattern.match(line)
            if not match:
                continue
            section = match.group(1).casefold()
            # Includes can load arbitrary external config files. Filter drivers
            # can execute configured commands during worktree diff conversion.
            if section in {"include", "includeif"} or (
                reject_filters and section == "filter"
            ):
                raise GitServiceError("unsafe_git_configuration")

    def _run_git(self, repository: Path, args: list[str], stdout_limit: int) -> bytes:
        """Run one fixed built-in command, retaining only bounded pipe output."""
        command = [
            "git",
            "--no-optional-locks",
            f"--git-dir={repository / '.git'}",
            f"--work-tree={repository}",
            "--no-pager",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.pager=cat",
            "-c",
            f"core.attributesFile={os.devnull}",
            "-c",
            "diff.external=",
            "-c",
            "core.quotepath=false",
            *args,
        ]
        try:
            process = subprocess.Popen(
                command,
                cwd=str(repository),
                env=self._safe_environment(),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                bufsize=0,
            )
        except FileNotFoundError:
            raise GitServiceError("git_unavailable") from None
        except PermissionError:
            raise GitServiceError("git_access_denied") from None
        except OSError:
            raise GitServiceError("git_unavailable") from None

        assert process.stdout is not None and process.stderr is not None
        overflow = threading.Event()
        reader_errors: list[BaseException] = []
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()

        def reader(
            stream,
            buffer: bytearray,
            limit: int,
            on_error: Callable[[BaseException], None],
        ) -> None:
            try:
                while True:
                    chunk = stream.read(16_384)
                    if not chunk:
                        return
                    remaining = limit + 1 - len(buffer)
                    if remaining > 0:
                        buffer.extend(chunk[:remaining])
                    if len(buffer) > limit or len(chunk) > remaining:
                        overflow.set()
                        return
            except BaseException as exc:  # surfaced in the calling thread below
                on_error(exc)

        threads = [
            threading.Thread(
                target=reader,
                args=(process.stdout, stdout_buffer, stdout_limit, reader_errors.append),
                name="git-stdout-reader",
            ),
            threading.Thread(
                target=reader,
                args=(process.stderr, stderr_buffer, STDERR_OUTPUT_LIMIT, reader_errors.append),
                name="git-stderr-reader",
            ),
        ]
        started_threads: list[threading.Thread] = []
        deadline = time.monotonic() + self._timeout_seconds
        timed_out = False
        try:
            for thread in threads:
                thread.start()
                started_threads.append(thread)
            while process.poll() is None:
                if overflow.is_set():
                    raise GitServiceError("output_limit_exceeded")
                if time.monotonic() >= deadline:
                    timed_out = True
                    raise GitServiceError("git_timeout")
                time.sleep(0.01)
            for thread in started_threads:
                thread.join(timeout=1.0)
            if any(thread.is_alive() for thread in started_threads):
                raise GitServiceError("git_cleanup_failed")
            if reader_errors:
                raise GitServiceError("git_read_failed") from reader_errors[0]
            if overflow.is_set():
                raise GitServiceError("output_limit_exceeded")
            if process.returncode != 0:
                logger.info("Read-only Git command failed (exit=%s)", process.returncode)
                raise GitServiceError("git_command_failed")
            return bytes(stdout_buffer)
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    logger.warning("Unable to request Git process termination")
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    try:
                        process.kill()
                        process.wait(timeout=1.0)
                    except (OSError, subprocess.TimeoutExpired):
                        logger.error("Git process did not exit after forced termination")
                except OSError:
                    try:
                        process.kill()
                        process.wait(timeout=1.0)
                    except (OSError, subprocess.TimeoutExpired):
                        logger.error("Git process did not exit after forced termination")
            for thread in started_threads:
                thread.join(timeout=1.0)
            for stream in (process.stdout, process.stderr):
                try:
                    stream.close()
                except OSError:
                    pass
            if timed_out:
                logger.info("Read-only Git command exceeded its deadline")

    @staticmethod
    def _relative_path(raw: bytes) -> str:
        path = raw.decode("utf-8", errors="replace").replace("\\", "/")
        if (
            not path
            or path.startswith("/")
            or (len(path) > 1 and path[1] == ":")
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise GitServiceError("invalid_git_output")
        return path

    @classmethod
    def _parse_status(cls, output: bytes) -> list[GitStatusEntry]:
        records = output.split(b"\x00")
        entries: list[GitStatusEntry] = []
        index = 0
        while index < len(records):
            record = records[index]
            index += 1
            if not record or record.startswith(b"# "):
                continue
            if record.startswith(b"? "):
                entries.append(
                    GitStatusEntry(
                        path=cls._relative_path(record[2:]),
                        index_status="?",
                        worktree_status="?",
                        kind="untracked",
                    )
                )
                continue
            if record.startswith((b"1 ", b"2 ")):
                is_rename = record.startswith(b"2 ")
                fields = record.split(b" ", 9 if is_rename else 8)
                expected = 10 if is_rename else 9
                if len(fields) != expected or len(fields[1]) != 2:
                    raise GitServiceError("invalid_git_output")
                original = None
                if is_rename:
                    if index >= len(records) or not records[index]:
                        raise GitServiceError("invalid_git_output")
                    original = cls._relative_path(records[index])
                    index += 1
                entries.append(
                    GitStatusEntry(
                        path=cls._relative_path(fields[-1]),
                        index_status=fields[1][:1].decode("ascii", errors="replace"),
                        worktree_status=fields[1][1:2].decode("ascii", errors="replace"),
                        original_path=original,
                    )
                )
                continue
            if record.startswith(b"u "):
                fields = record.split(b" ", 10)
                if len(fields) != 11 or len(fields[1]) != 2:
                    raise GitServiceError("invalid_git_output")
                entries.append(
                    GitStatusEntry(
                        path=cls._relative_path(fields[-1]),
                        index_status=fields[1][:1].decode("ascii", errors="replace"),
                        worktree_status=fields[1][1:2].decode("ascii", errors="replace"),
                        kind="unmerged",
                    )
                )
                continue
            raise GitServiceError("invalid_git_output")
        return entries

"""Tests for bounded, read-only Git workspace inspection."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path

import pytest

from backend.infrastructure.git_service import (
    DIFF_OUTPUT_LIMIT,
    GitService,
    GitServiceError,
)
from rag.config import RagConfig


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=True,
        text=True,
    )
    return result.stdout


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "initial commit")
    return path


def _service(root: Path, **kwargs) -> GitService:
    return GitService(
        RagConfig(allowed_repository_roots=(str(root),)), **kwargs
    )


def test_read_only_operations_return_status_branch_diff_and_history(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "allowed" / "repo")
    service = _service(tmp_path / "allowed")
    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (repo / "untracked name.txt").write_text("new\n", encoding="utf-8")

    status = service.get_status(repo)
    assert status.branch
    assert status.is_clean is False
    assert {entry.path for entry in status.entries} == {
        "tracked.txt",
        "untracked name.txt",
    }
    assert service.get_current_branch(repo) == status.branch
    assert "tracked.txt" in service.get_diff(repo)
    history = service.get_commit_history(repo, max_count=5)
    assert len(history) == 1
    assert history[0].message == "initial commit"


def test_paths_outside_roots_empty_roots_traversal_and_missing_repo_rejected(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    repo = _make_repo(allowed / "repo")
    outside = _make_repo(tmp_path / "outside")

    with pytest.raises(GitServiceError):
        _service(allowed).get_status(outside)
    with pytest.raises(GitServiceError):
        GitService(RagConfig()).get_status(repo)
    with pytest.raises(GitServiceError):
        _service(allowed).get_status(str(allowed) + "\\..\\allowed\\repo")
    with pytest.raises(GitServiceError):
        _service(allowed).get_status(allowed / "missing")
    with pytest.raises(GitServiceError):
        _service(allowed).get_status(allowed)


def test_symlink_escape_is_rejected_or_skipped_when_unavailable(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = _make_repo(tmp_path / "outside")
    link = allowed / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(GitServiceError):
        _service(allowed).get_status(link)


def test_git_worktree_marker_file_is_explicitly_rejected(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    worktree = root / "worktree"
    worktree.mkdir()
    (worktree / ".git").write_text("gitdir: ../somewhere\n", encoding="utf-8")
    with pytest.raises(GitServiceError, match="worktree_unsupported"):
        _service(root).get_status(worktree)


def test_repository_config_includes_and_diff_filters_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "allowed"
    repo = _make_repo(root / "repo")
    config_path = repo / ".git" / "config"
    with config_path.open("a", encoding="utf-8") as config:
        config.write('\n[filter "external"]\n\tclean = arbitrary-command\n')
    with pytest.raises(GitServiceError, match="unsafe_git_configuration"):
        _service(root).get_diff(repo)

    config_path.write_text(
        config_path.read_text(encoding="utf-8")
        + '\n[include]\n\tpath = C:/outside/config\n',
        encoding="utf-8",
    )
    with pytest.raises(GitServiceError, match="unsafe_git_configuration"):
        _service(root).get_current_branch(repo)


def test_only_fixed_read_only_operations_are_public() -> None:
    public_methods = {
        name
        for name in dir(GitService)
        if not name.startswith("_") and callable(getattr(GitService, name))
    }
    assert public_methods == {
        "get_current_branch",
        "get_status",
        "get_diff",
        "get_commit_history",
    }


def test_read_only_git_command_uses_safe_process_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo(tmp_path / "allowed" / "repo")
    captured = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            captured.update(command=command, kwargs=kwargs)
            self.stdout = io.BytesIO(b"main\n")
            self.stderr = io.BytesIO()
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -1

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "backend.infrastructure.git_service.subprocess.Popen", FakeProcess
    )
    output = _service(tmp_path / "allowed")._run_git(
        repo, ["rev-parse", "--abbrev-ref", "HEAD"], 1024
    )
    assert output == b"main\n"
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["stdin"] == subprocess.DEVNULL
    assert "GIT_OPTIONAL_LOCKS" not in os.environ or captured["kwargs"]["env"][
        "GIT_OPTIONAL_LOCKS"
    ] == "0"
    assert captured["kwargs"]["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
    assert captured["kwargs"]["env"]["GIT_CONFIG_GLOBAL"] == os.devnull
    assert captured["command"][0] == "git"
    assert "--no-optional-locks" in captured["command"]
    assert any(argument.startswith("--git-dir=") for argument in captured["command"])
    assert any(argument.startswith("--work-tree=") for argument in captured["command"])
    assert "--no-pager" in captured["command"]
    assert "core.fsmonitor=false" in captured["command"]
    assert "diff.external=" in captured["command"]
    assert f"core.attributesFile={os.devnull}" in captured["command"]


def test_output_limit_terminates_a_still_running_process(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _make_repo(tmp_path / "allowed" / "repo")
    process_ref = {}

    class FakeProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.BytesIO(b"x" * 128)
            self.stderr = io.BytesIO()
            self.returncode = None
            self.terminated = False
            process_ref["process"] = self

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "backend.infrastructure.git_service.subprocess.Popen", FakeProcess
    )
    with pytest.raises(GitServiceError, match="output_limit_exceeded"):
        _service(tmp_path / "allowed")._run_git(repo, ["status"], 16)
    assert process_ref["process"].terminated is True


def test_timeout_terminates_the_git_process(tmp_path: Path, monkeypatch) -> None:
    repo = _make_repo(tmp_path / "allowed" / "repo")
    process_ref = {}

    class FakeProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode = None
            process_ref["process"] = self

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "backend.infrastructure.git_service.subprocess.Popen", FakeProcess
    )
    with pytest.raises(GitServiceError, match="git_timeout"):
        _service(tmp_path / "allowed", timeout_seconds=0.02)._run_git(
            repo, ["status"], 1024
        )
    assert process_ref["process"].returncode == -15


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_output_limits_terminate_or_reject_excess_output(
    tmp_path: Path, monkeypatch, stream: str
) -> None:
    repo = _make_repo(tmp_path / "allowed" / "repo")

    class FakeProcess:
        def __init__(self, *_args, **_kwargs):
            self.stdout = io.BytesIO(b"x" * 128) if stream == "stdout" else io.BytesIO()
            self.stderr = io.BytesIO(b"x" * 128) if stream == "stderr" else io.BytesIO()
            self.returncode = 0

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        def wait(self, timeout=None):
            return self.returncode

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr(
        "backend.infrastructure.git_service.subprocess.Popen", FakeProcess
    )
    stdout_limit = 16 if stream == "stdout" else DIFF_OUTPUT_LIMIT
    stderr_limit = 16 if stream == "stderr" else 65536
    if stream == "stdout":
        with pytest.raises(GitServiceError, match="output_limit_exceeded"):
            _service(tmp_path / "allowed")._run_git(
                repo, ["status"], stdout_limit
            )
    else:
        # Stderr has a fixed production cap; use an oversized mock reader.
        from backend.infrastructure.git_service import STDERR_OUTPUT_LIMIT

        class LargeErrorProcess(FakeProcess):
            def __init__(self, *_args, **_kwargs):
                self.stdout = io.BytesIO()
                self.stderr = io.BytesIO(b"x" * (STDERR_OUTPUT_LIMIT + 1))
                self.returncode = 0

        monkeypatch.setattr(
            "backend.infrastructure.git_service.subprocess.Popen", LargeErrorProcess
        )
        with pytest.raises(GitServiceError, match="output_limit_exceeded"):
            _service(tmp_path / "allowed")._run_git(repo, ["status"], 1024)


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_missing_git_and_permission_errors_are_classified(
    tmp_path: Path, monkeypatch, error
) -> None:
    repo = _make_repo(tmp_path / "allowed" / "repo")

    def fail(*_args, **_kwargs):
        raise error()

    monkeypatch.setattr(
        "backend.infrastructure.git_service.subprocess.Popen", fail
    )
    expected = "git_unavailable" if error is FileNotFoundError else "git_access_denied"
    with pytest.raises(GitServiceError, match=expected):
        _service(tmp_path / "allowed").get_current_branch(repo)


def test_status_parser_handles_unusual_paths_and_renames() -> None:
    output = (
        b"1 .M N... 100644 100644 100644 abc abc path with spaces.txt\0"
        b"? newline\nname.txt\0"
        b"2 R. N... 100644 100644 100644 abc def R100 renamed.txt\0old name.txt\0"
    )
    entries = GitService._parse_status(output)
    assert entries[0].path == "path with spaces.txt"
    assert entries[1].path == "newline\nname.txt"
    assert entries[1].kind == "untracked"
    assert entries[2].path == "renamed.txt"
    assert entries[2].original_path == "old name.txt"


@pytest.mark.parametrize("output", [b"garbage\0", b"1 .M\0", b"? ../outside\0"])
def test_status_parser_rejects_malformed_or_escaping_paths(output: bytes) -> None:
    with pytest.raises(GitServiceError, match="invalid_git_output"):
        GitService._parse_status(output)


def test_history_and_diff_limits_are_validated(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path / "allowed" / "repo")
    service = _service(tmp_path / "allowed")
    with pytest.raises(GitServiceError):
        service.get_commit_history(repo, 101)
    with pytest.raises(GitServiceError):
        service.get_diff(repo, max_bytes=DIFF_OUTPUT_LIMIT + 1)

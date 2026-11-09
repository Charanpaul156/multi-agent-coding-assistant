"""Read-only Git workspace API tests using temporary repositories."""

from __future__ import annotations

import subprocess
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import get_git_workspace_use_cases
from backend.main import app
from backend.application.git_workspace_use_cases import GitWorkspaceUseCases
from backend.infrastructure.git_service import GitService
from rag.config import RagConfig


def _git(repo: Path, *args: str) -> bytes:
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        stdin=subprocess.DEVNULL,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=True,
    )
    return result.stdout


def _make_repo(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "API Test")
    _git(path, "config", "user.email", "api-test@example.invalid")
    (path / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "commit", "-qm", "api test commit")
    (path / "untracked.txt").write_text("untracked contents\n", encoding="utf-8")
    (path / "tracked.txt").write_text("changed contents\n", encoding="utf-8")
    return path


def _repository_snapshot(repo: Path):
    index = (repo / ".git" / "index").read_bytes()
    status = _git(repo, "status", "--porcelain=v2", "-z", "--untracked-files=all")
    files = {
        path.relative_to(repo).as_posix(): path.read_bytes()
        for path in repo.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(repo).parts
    }
    return index, status, files


@pytest.fixture
def api_client(tmp_path: Path):
    root = tmp_path / "allowed"
    repo = _make_repo(root / "repo")
    use_cases = GitWorkspaceUseCases(
        GitService(RagConfig(allowed_repository_roots=(str(root),)))
    )
    app.dependency_overrides[get_git_workspace_use_cases] = lambda: use_cases
    try:
        with TestClient(app) as client:
            yield client, repo, root
    finally:
        app.dependency_overrides.pop(get_git_workspace_use_cases, None)


@pytest.mark.parametrize(
    ("endpoint", "extra", "expected_key"),
    [
        ("/git-workspace/status", {}, "entries"),
        ("/git-workspace/branch", {}, "branch"),
        ("/git-workspace/diff", {"max_bytes": 4096}, "diff"),
        ("/git-workspace/history", {"max_count": 5}, "commits"),
    ],
)
def test_read_only_endpoints_serialize_and_preserve_repository(
    api_client, endpoint: str, extra: dict, expected_key: str
) -> None:
    client, repo, _root = api_client
    before = _repository_snapshot(repo)
    response = client.post(
        endpoint, json={"repository_path": str(repo), **extra}
    )
    after = _repository_snapshot(repo)

    assert response.status_code == 200, response.text
    assert expected_key in response.json()
    assert after == before
    if endpoint.endswith("/status"):
        payload = response.json()
        assert payload["is_clean"] is False
        assert {entry["path"] for entry in payload["entries"]} == {
            "tracked.txt",
            "untracked.txt",
        }
    elif endpoint.endswith("/diff"):
        assert "tracked.txt" in response.json()["diff"]
    elif endpoint.endswith("/history"):
        assert response.json()["commits"][0]["message"] == "api test commit"


def test_invalid_repository_error_is_generic_and_does_not_leak_paths(api_client) -> None:
    client, _repo, _root = api_client
    outside = str(Path(_root).parent / "outside-secret-path")
    response = client.post(
        "/git-workspace/status", json={"repository_path": outside}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Repository is unavailable or not permitted."
    assert outside not in response.text


@pytest.mark.parametrize(
    ("endpoint", "body"),
    [
        ("/git-workspace/history", {"max_count": 101}),
        ("/git-workspace/diff", {"max_bytes": 1048577}),
        ("/git-workspace/status", {"git_args": ["reset", "--hard"]}),
    ],
)
def test_api_rejects_out_of_bounds_and_arbitrary_command_fields(
    api_client, endpoint: str, body: dict
) -> None:
    client, repo, _root = api_client
    response = client.post(
        endpoint, json={"repository_path": str(repo), **body}
    )
    assert response.status_code == 422

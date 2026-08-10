"""Tests for RepositoryLoader and repository security/filtering."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag.config import RagConfig
from rag.file_loader import (
    RepositoryLoader,
    RepositoryNotAllowedError,
    RepositoryNotFoundError,
)


def _make_config(allowed_root: Path) -> RagConfig:
    return RagConfig(allowed_repository_roots=(str(allowed_root),))


def _write(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_loads_supported_python_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "app.py", "def foo():\n    return 1\n")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1
        assert files[0].repo_path == "app.py"
        assert files[0].language == "python"
        assert files[0].line_count == 2
        assert files[0].file_hash


def test_skips_unsupported_extension() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "app.py", "def foo():\n    return 1\n")
        _write(root, "notes.xyz", "ignored")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1
        assert all(f.repo_path != "notes.xyz" for f in files)


def test_skips_env_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "app.py", "def foo(): pass\n")
        _write(root, ".env", "SECRET=abc\n")
        _write(root, ".env.local", "SECRET=abc\n")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1
        assert files[0].repo_path == "app.py"


def test_skips_git_and_venv_dirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, ".git/config", "x")
        _write(root, ".venv/lib/py/site.py", "x")
        _write(root, "app.py", "def f(): pass\n")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1


def test_skips_node_modules_and_build() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "node_modules/pkg/index.js", "x")
        _write(root, "dist/bundle.js", "x")
        _write(root, "build/out.js", "x")
        _write(root, "src/index.js", "const x=1;\n")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1
        assert files[0].repo_path == "src/index.js"


def test_skips_binary_extensions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "app.py", "def f(): pass\n")
        (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1


def test_skips_private_key_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "app.py", "def f(): pass\n")
        _write(root, "id_rsa", "-----BEGIN RSA PRIVATE KEY-----\nAAAA\n")
        _write(root, "service-account.json", "{}")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1


def test_preserves_file_path_metadata() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "pkg/util/helper.py", "def helper(): pass\n")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1
        assert files[0].repo_path == "pkg/util/helper.py"
        assert files[0].repository == root.name


def test_missing_repository_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        loader = RepositoryLoader(_make_config(root))
        with pytest.raises(RepositoryNotFoundError):
            loader.load_repository(root / "does_not_exist")


def test_path_outside_allowed_root_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        outer = Path(tmp)
        allowed = outer / "allowed"
        allowed.mkdir()
        outside = outer / "outside"
        outside.mkdir()
        _write(outside, "app.py", "def f(): pass\n")
        loader = RepositoryLoader(_make_config(allowed))
        with pytest.raises(RepositoryNotAllowedError):
            loader.load_repository(outside)


def test_no_allowed_roots_disables_indexing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "app.py", "def f(): pass\n")
        loader = RepositoryLoader(RagConfig(allowed_repository_roots=()))
        with pytest.raises(RepositoryNotAllowedError):
            loader.load_repository(root)


def test_markdown_is_supported() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write(root, "README.md", "# Hi\n")
        _write(root, ".env", "X=1")
        loader = RepositoryLoader(_make_config(root))
        files = loader.load_repository(root)
        assert len(files) == 1
        assert files[0].repo_path == "README.md"


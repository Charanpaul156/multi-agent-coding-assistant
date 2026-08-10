"""Tests for the ChangeValidator security boundary.

These tests verify path-traversal rejection, protected-file rejection,
operation consistency, syntax validation, and hash verification. All tests
use temporary directories and never touch the real project.
"""

from __future__ import annotations

import hashlib

import pytest

from backend.domain.change_models import (
    ChangeOperation,
    ChangeSet,
    FileChange,
)
from backend.infrastructure.change_validation import ChangeValidator, ChangeValidationError
from rag.config import RagConfig


@pytest.fixture
def repo_root(tmp_path):
    """Create a temporary repository root and return its string path."""
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def config(repo_root):
    """RagConfig with the temp repo as the only allowed root."""
    return RagConfig(allowed_repository_roots=(str(repo_root),))


@pytest.fixture
def validator(config):
    return ChangeValidator(config=config)


def _create_change(file_path, operation=ChangeOperation.CREATE, content="x = 1\n"):
    return FileChange(
        file_path=file_path,
        operation=operation,
        new_content=content,
    )


def _validate_single(validator, change):
    report = validator.validate(ChangeSet(changes=[change]))
    return report.valid, report.results[0].messages


# ---------------------------------------------------------------------------
# Basic structural checks
# ---------------------------------------------------------------------------


def test_valid_create_pass(validator):
    valid, messages = _validate_single(validator, _create_change("new_file.py"))
    assert valid is True
    assert messages == []


def test_empty_change_set_invalid(validator):
    report = validator.validate(ChangeSet(changes=[]))
    assert report.valid is False


def test_non_changeset_raises(validator):
    with pytest.raises(ChangeValidationError):
        validator.validate("not a changeset")


def test_empty_file_path_invalid(validator):
    change = FileChange(
        file_path="  ",
        operation=ChangeOperation.CREATE,
        new_content="x = 1\n",
    )
    valid, messages = _validate_single(validator, change)
    assert valid is False


def test_empty_content_invalid(validator):
    change = FileChange(
        file_path="a.py",
        operation=ChangeOperation.CREATE,
        new_content="   ",
    )
    valid, messages = _validate_single(validator, change)
    assert valid is False


# ---------------------------------------------------------------------------
# Path traversal / security
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(validator):
    change = _create_change("../outside.py")
    valid, messages = _validate_single(validator, change)
    assert valid is False
    assert any("traversal" in m for m in messages)


def test_absolute_path_rejected(validator):
    change = _create_change("/etc/passwd")
    valid, messages = _validate_single(validator, change)
    assert valid is False
    assert any("absolute" in m for m in messages)


def test_backslash_rejected(validator):
    change = _create_change("..\\..\\secret.py")
    valid, messages = _validate_single(validator, change)
    assert valid is False


def test_nul_byte_rejected(validator):
    change = _create_change("a\x00b.py")
    valid, messages = _validate_single(validator, change)
    assert valid is False


def test_outside_root_rejected(validator, tmp_path):
    # A path that resolves outside the allowed root.
    outside = tmp_path / "other" / "file.py"
    outside.parent.mkdir()
    outside.write_text("x = 1\n")
    change = _create_change("other/file.py")
    valid, messages = _validate_single(validator, change)
    assert valid is False


# ---------------------------------------------------------------------------
# Protected files
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        ".env",
        ".env.local",
        ".env.production",
        "config/.env",
        "deploy/.env",
    ],
)
def test_env_files_rejected(validator, path):
    change = _create_change(path)
    valid, messages = _validate_single(validator, change)
    assert valid is False
    assert any("protected" in m for m in messages)


@pytest.mark.parametrize(
    "path",
    [
        ".git/config",
        "secrets.json",
        "credentials",
        "service-account.json",
        "token.txt",
        "deploy/id_rsa",
        "keys/private.pem",
    ],
)
def test_protected_patterns_rejected(validator, path):
    change = _create_change(path)
    valid, messages = _validate_single(validator, change)
    assert valid is False
    assert any("protected" in m for m in messages)


# ---------------------------------------------------------------------------
# Operation consistency
# ---------------------------------------------------------------------------


def test_create_existing_file_rejected(validator, repo_root):
    (repo_root / "exists.py").write_text("old\n")
    change = _create_change("exists.py", operation=ChangeOperation.CREATE)
    valid, messages = _validate_single(validator, change)
    assert valid is False
    assert any("existing" in m for m in messages)


def test_create_existing_dir_rejected(validator, repo_root):
    (repo_root / "somedir").mkdir()
    change = _create_change("somedir", operation=ChangeOperation.CREATE)
    valid, messages = _validate_single(validator, change)
    assert valid is False


def test_modify_missing_file_rejected(validator):
    change = _create_change("missing.py", operation=ChangeOperation.MODIFY)
    valid, messages = _validate_single(validator, change)
    assert valid is False
    assert any("non-existent" in m for m in messages)


def test_modify_existing_file_pass(validator, repo_root):
    (repo_root / "a.py").write_text("x = 1\n")
    change = _create_change("a.py", operation=ChangeOperation.MODIFY)
    valid, messages = _validate_single(validator, change)
    assert valid is True


# ---------------------------------------------------------------------------
# Hash verification
# ---------------------------------------------------------------------------


def test_hash_mismatch_rejected(validator, repo_root):
    (repo_root / "a.py").write_text("x = 1\n")
    change = FileChange(
        file_path="a.py",
        operation=ChangeOperation.MODIFY,
        new_content="x = 2\n",
        original_hash="wronghash",
    )
    valid, messages = _validate_single(validator, change)
    assert valid is False
    assert any("original_hash mismatch" in m for m in messages)


def test_hash_match_pass(validator, repo_root):
    (repo_root / "a.py").write_text("x = 1\n")
    correct_hash = hashlib.sha256((repo_root / "a.py").read_bytes()).hexdigest()
    change = FileChange(
        file_path="a.py",
        operation=ChangeOperation.MODIFY,
        new_content="x = 2\n",
        original_hash=correct_hash,
    )
    valid, messages = _validate_single(validator, change)
    assert valid is True


# ---------------------------------------------------------------------------
# Python syntax validation
# ---------------------------------------------------------------------------


def test_valid_python_syntax_pass(validator):
    change = _create_change("ok.py", content="def f():\n    return 1\n")
    valid, messages = _validate_single(validator, change)
    assert valid is True


def test_invalid_python_syntax_rejected(validator):
    change = _create_change("bad.py", content="def f(:\n    return 1\n")
    valid, messages = _validate_single(validator, change)
    assert valid is False
    assert any("syntax" in m for m in messages)


def test_non_python_syntax_not_checked(validator):
    change = _create_change("data.json", content="{ invalid json !!!")
    valid, messages = _validate_single(validator, change)
    # JSON isn't python so no python syntax check; content is non-empty so pass.
    assert valid is True


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def test_aggregate_invalid_if_any_invalid(validator):
    ok = _create_change("ok.py")
    bad = _create_change("../escape.py")
    report = validator.validate(ChangeSet(changes=[ok, bad]))
    assert report.valid is False
    assert len(report.results) == 2

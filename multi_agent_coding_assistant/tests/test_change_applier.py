"""Tests for the ChangeApplier: create, modify, dry-run, and rollback.

All tests use temporary directories and never touch the real project.
"""

from __future__ import annotations

import hashlib

import pytest

from backend.domain.change_models import (
    ChangeOperation,
    ChangeSet,
    FileChange,
)
from backend.infrastructure.change_applier import ChangeApplier, ChangeApplyError
from rag.config import RagConfig


@pytest.fixture
def repo_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def config(repo_root):
    return RagConfig(allowed_repository_roots=(str(repo_root),))


@pytest.fixture
def applier(config):
    return ChangeApplier(config=config)


def _modify_change(file_path, new_content, original_hash=None):
    return FileChange(
        file_path=file_path,
        operation=ChangeOperation.MODIFY,
        new_content=new_content,
        original_hash=original_hash,
    )


def _create_change(file_path, content):
    return FileChange(
        file_path=file_path,
        operation=ChangeOperation.CREATE,
        new_content=content,
    )


# ---------------------------------------------------------------------------
# Create
# ---------------------------------------------------------------------------


def test_create_file(applier, repo_root):
    result = applier.apply(
        ChangeSet(changes=[_create_change("new.py", "x = 1\n")])
    )
    assert result.success is True
    assert result.applied_files == ["new.py"]
    assert result.dry_run is False
    assert (repo_root / "new.py").read_text() == "x = 1\n"


def test_create_creates_parent_dirs(applier, repo_root):
    result = applier.apply(
        ChangeSet(changes=[_create_change("a/b/c.py", "x = 1\n")])
    )
    assert result.success is True
    assert (repo_root / "a" / "b" / "c.py").exists()


def test_create_existing_fails(applier, repo_root):
    (repo_root / "exists.py").write_text("old\n")
    result = applier.apply(
        ChangeSet(changes=[_create_change("exists.py", "new\n")])
    )
    assert result.success is False
    # Original content preserved.
    assert (repo_root / "exists.py").read_text() == "old\n"


# ---------------------------------------------------------------------------
# Modify
# ---------------------------------------------------------------------------


def test_modify_file(applier, repo_root):
    (repo_root / "a.py").write_text("x = 1\n")
    result = applier.apply(
        ChangeSet(changes=[_modify_change("a.py", "x = 2\n")])
    )
    assert result.success is True
    assert (repo_root / "a.py").read_text() == "x = 2\n"


def test_modify_verifies_hash_success(applier, repo_root):
    (repo_root / "a.py").write_text("x = 1\n")
    correct = hashlib.sha256((repo_root / "a.py").read_bytes()).hexdigest()
    result = applier.apply(
        ChangeSet(changes=[_modify_change("a.py", "x = 2\n", correct)])
    )
    assert result.success is True


def test_modify_hash_mismatch_aborts(applier, repo_root):
    (repo_root / "a.py").write_text("x = 1\n")
    # Simulate the file changing after generation: wrong hash supplied.
    result = applier.apply(
        ChangeSet(changes=[_modify_change("a.py", "x = 2\n", "wrong-hash")])
    )
    assert result.success is False
    assert "hash mismatch" in (result.error or "")
    # File unchanged.
    assert (repo_root / "a.py").read_text() == "x = 1\n"


def test_modify_missing_file_fails(applier):
    result = applier.apply(
        ChangeSet(changes=[_modify_change("missing.py", "x = 2\n")])
    )
    assert result.success is False


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_does_not_write(applier, repo_root):
    result = applier.apply(
        ChangeSet(changes=[_create_change("dry.py", "x = 1\n")]),
        dry_run=True,
    )
    assert result.success is True
    assert result.dry_run is True
    assert result.applied_files == ["dry.py"]
    assert not (repo_root / "dry.py").exists()


def test_dry_run_modify_does_not_change(applier, repo_root):
    (repo_root / "a.py").write_text("x = 1\n")
    result = applier.apply(
        ChangeSet(changes=[_modify_change("a.py", "x = 2\n")]),
        dry_run=True,
    )
    assert result.success is True
    assert (repo_root / "a.py").read_text() == "x = 1\n"


# ---------------------------------------------------------------------------
# Rollback / transaction
# ---------------------------------------------------------------------------


def test_rollback_restores_modified_file(applier, repo_root):
    (repo_root / "a.py").write_text("original\n")
    record = applier.apply(
        ChangeSet(changes=[_modify_change("a.py", "changed\n")])
    ).rollback_records
    assert len(record) == 1

    applier.rollback(record)
    assert (repo_root / "a.py").read_text() == "original\n"


def test_rollback_removes_created_file(applier, repo_root):
    record = applier.apply(
        ChangeSet(changes=[_create_change("new.py", "x = 1\n")])
    ).rollback_records
    assert record[0].was_created is True
    assert (repo_root / "new.py").exists()

    applier.rollback(record)
    assert not (repo_root / "new.py").exists()


def test_multi_file_partial_failure_rolls_back(applier, repo_root):
    # First change succeeds (create new.py); second change fails because
    # target file changed (hash mismatch). The whole transaction rolls back.
    (repo_root / "a.py").write_text("x = 1\n")
    create = _create_change("new.py", "x = 1\n")
    bad_modify = _modify_change("a.py", "x = 2\n", "wrong-hash")

    result = applier.apply(ChangeSet(changes=[create, bad_modify]))
    assert result.success is False
    # Created file must be rolled back.
    assert not (repo_root / "new.py").exists()
    # Modified file unchanged.
    assert (repo_root / "a.py").read_text() == "x = 1\n"


def test_rollback_multi_file(applier, repo_root):
    (repo_root / "a.py").write_text("orig_a\n")
    (repo_root / "b.py").write_text("orig_b\n")
    records = applier.apply(
        ChangeSet(
            changes=[
                _modify_change("a.py", "new_a\n"),
                _modify_change("b.py", "new_b\n"),
            ]
        )
    ).rollback_records
    assert len(records) == 2
    assert (repo_root / "a.py").read_text() == "new_a\n"
    assert (repo_root / "b.py").read_text() == "new_b\n"

    applier.rollback(records)
    assert (repo_root / "a.py").read_text() == "orig_a\n"
    assert (repo_root / "b.py").read_text() == "orig_b\n"


# ---------------------------------------------------------------------------
# Path safety in applier (defense in depth)
# ---------------------------------------------------------------------------


def test_applier_rejects_traversal(applier, tmp_path):
    result = applier.apply(
        ChangeSet(changes=[_create_change("../evil.py", "x = 1\n")])
    )
    assert result.success is False


def test_applier_rejects_absolute(applier):
    result = applier.apply(
        ChangeSet(changes=[_create_change("/tmp/evil.py", "x = 1\n")])
    )
    assert result.success is False


def test_applier_rejects_unsafe_backslash(applier):
    result = applier.apply(
        ChangeSet(changes=[_create_change("..\\..\\evil.py", "x = 1\n")])
    )
    assert result.success is False


def test_non_changeset_raises(applier):
    with pytest.raises(ChangeApplyError):
        applier.apply("not a changeset")

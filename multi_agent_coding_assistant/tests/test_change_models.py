"""Tests for the domain change models (FileChange, ChangeSet, etc.)."""

from __future__ import annotations

from backend.domain.change_models import (
    ApplicationResult,
    ChangeOperation,
    ChangeSet,
    ChangeValidationResult,
    DiffEntry,
    FileChange,
    RollbackRecord,
    ValidationReport,
)


def test_change_operation_values():
    assert ChangeOperation.CREATE.value == "create"
    assert ChangeOperation.MODIFY.value == "modify"


def test_change_operation_no_delete():
    assert not hasattr(ChangeOperation, "DELETE")


def test_file_change_create():
    change = FileChange(
        file_path="backend/auth/service.py",
        operation=ChangeOperation.CREATE,
        new_content="def add():\n    return 1\n",
        description="Create auth service",
    )
    assert change.file_path == "backend/auth/service.py"
    assert change.operation == ChangeOperation.CREATE
    assert change.original_content is None
    assert change.original_hash is None


def test_file_change_modify_with_original():
    change = FileChange(
        file_path="a.py",
        operation=ChangeOperation.MODIFY,
        new_content="x = 2\n",
        original_content="x = 1\n",
        original_hash="deadbeef",
        description="bump",
    )
    assert change.original_content == "x = 1\n"
    assert change.original_hash == "deadbeef"


def test_change_set_defaults():
    change_set = ChangeSet()
    assert change_set.changes == []
    assert change_set.summary == ""


def test_change_set_with_changes():
    change = FileChange(
        file_path="a.py",
        operation=ChangeOperation.CREATE,
        new_content="pass\n",
    )
    change_set = ChangeSet(changes=[change], summary="Add a.py")
    assert len(change_set.changes) == 1
    assert change_set.summary == "Add a.py"


def test_validation_result_and_report():
    r1 = ChangeValidationResult(True, "a.py", "create", [])
    r2 = ChangeValidationResult(False, "b.py", "modify", ["unsafe path"])
    report = ValidationReport(valid=False, results=[r1, r2])
    assert report.valid is False
    assert len(report.results) == 2
    assert report.results[0].valid is True
    assert "unsafe path" in report.results[1].messages


def test_diff_entry():
    entry = DiffEntry(file_path="a.py", operation="create", diff_text="+ x = 1")
    assert entry.file_path == "a.py"
    assert entry.operation == "create"


def test_rollback_record():
    record = RollbackRecord(
        file_path="/tmp/repo/a.py",
        original_content="old",
        original_hash="abc",
        was_created=False,
    )
    assert record.was_created is False
    assert record.original_content == "old"


def test_rollback_record_for_create():
    record = RollbackRecord(
        file_path="/tmp/repo/new.py",
        original_content=None,
        original_hash=None,
        was_created=True,
    )
    assert record.was_created is True


def test_application_result_defaults():
    result = ApplicationResult(success=True)
    assert result.applied_files == []
    assert result.rollback_records == []
    assert result.dry_run is False
    assert result.error is None

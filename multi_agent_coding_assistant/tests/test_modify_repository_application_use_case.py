"""Tests for the repository modification application use case."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.application.modify_repository_use_cases import (
    ModifyRepositoryRequest,
    ModifyRepositoryResult,
    ModifyRepositoryUseCase,
)
from backend.domain.change_models import (
    ApplicationResult,
    ChangeOperation,
    ChangeSet,
    ChangeValidationResult,
    DiffEntry,
    FileChange,
    ValidationReport,
    ApprovalStatus,
)


def _change_set() -> ChangeSet:
    return ChangeSet(
        changes=[
            FileChange(
                file_path="app.py",
                operation=ChangeOperation.MODIFY,
                new_content="print('new')\n",
                original_content="print('old')\n",
                description="update greeting",
            )
        ],
        summary="update app",
    )


def _valid_validation() -> ValidationReport:
    return ValidationReport(
        valid=True,
        results=[
            ChangeValidationResult(
                valid=True,
                file_path="app.py",
                operation="modify",
                messages=[],
            )
        ],
    )


def _invalid_validation() -> ValidationReport:
    return ValidationReport(
        valid=False,
        results=[
            ChangeValidationResult(
                valid=False,
                file_path="app.py",
                operation="modify",
                messages=["path outside allowed root"],
            )
        ],
    )


class FakeValidator:
    def __init__(self, validation_result: ValidationReport):
        self.validation_result = validation_result
        self.calls: list[ChangeSet] = []

    def validate(self, change_set: ChangeSet) -> ValidationReport:
        self.calls.append(change_set)
        return self.validation_result


class FakeDiffGenerator:
    def __init__(self):
        self.calls: list[list[FileChange]] = []

    def generate_many(self, changes):
        changes = list(changes)
        self.calls.append(changes)
        return [f"diff:{change.file_path}" for change in changes]


class FakeApplier:
    def __init__(self, result: ApplicationResult):
        self.result = result
        self.calls: list[tuple[ChangeSet, bool]] = []

    def apply(self, change_set: ChangeSet, *, dry_run: bool = False) -> ApplicationResult:
        self.calls.append((change_set, dry_run))
        return self.result


@dataclass
class FactorySpy:
    instance: object

    def __post_init__(self):
        self.configs = []

    def __call__(self, config):
        self.configs.append(config)
        return self.instance


def _request(repo_root: str, *, dry_run: bool = False, approval: ApprovalStatus | None = None) -> ModifyRepositoryRequest:
    appr = approval if approval is not None else (ApprovalStatus.PREVIEW if dry_run else ApprovalStatus.APPROVED)
    return ModifyRepositoryRequest(
        repository_root=repo_root,
        change_set=_change_set(),
        dry_run=dry_run,
        approval=appr,
    )


from rag.config import RagConfig


def _make_use_case(repo_root: str, validator: FakeValidator, applier: FakeApplier) -> ModifyRepositoryUseCase:
    from unittest.mock import MagicMock
    config = RagConfig(allowed_repository_roots=(repo_root,))
    return ModifyRepositoryUseCase(
        config=config,
        retriever_use_case=MagicMock(),
        plan_use_case=MagicMock(),
        coder_agent=MagicMock(),
        validator=validator,
        applier=applier,
    )


def test_execute_valid_dry_run(tmp_path):
    repo_root = str(tmp_path / "repo")
    validator = FakeValidator(_valid_validation())
    applier = FakeApplier(
        ApplicationResult(
            success=True,
            applied_files=[],
            rollback_records=[],
            dry_run=True,
        )
    )
    use_case = _make_use_case(repo_root, validator, applier)

    result = use_case.execute(_request(repo_root, dry_run=True))

    assert result.success is True
    assert result.dry_run is True
    assert result.validation_result == _valid_validation()
    assert result.applied_files == []
    assert len(result.diff) == 1
    assert result.diff[0].file_path == "app.py"


def test_execute_validation_failure_short_circuits(tmp_path):
    repo_root = str(tmp_path / "repo")
    validator = FakeValidator(_invalid_validation())
    applier = FakeApplier(
        ApplicationResult(success=True, applied_files=["app.py"], dry_run=False)
    )
    use_case = _make_use_case(repo_root, validator, applier)

    result = use_case.execute(_request(repo_root))

    assert result.success is False
    assert result.validation_result == _invalid_validation()
    assert result.applied_files == []
    assert "validation" in (result.error or "").lower()
    assert applier.calls == []


def test_execute_application_failure_preserves_result(tmp_path):
    repo_root = str(tmp_path / "repo")
    validator = FakeValidator(_valid_validation())
    applier = FakeApplier(
        ApplicationResult(
            success=False,
            applied_files=["app.py"],
            rollback_records=[],
            dry_run=False,
            error="apply boom",
        )
    )
    use_case = _make_use_case(repo_root, validator, applier)

    result = use_case.execute(_request(repo_root))

    assert result.success is False
    assert result.validation_result == _valid_validation()
    assert result.application_result is not None
    assert result.application_result.error == "apply boom"
    assert result.applied_files == ["app.py"]
    assert len(result.diff) == 1
    assert "apply boom" in (result.error or "")


def test_execute_rejects_invalid_request_type(tmp_path):
    repo_root = str(tmp_path / "repo")
    validator = FakeValidator(_valid_validation())
    applier = FakeApplier(ApplicationResult(success=True))
    use_case = _make_use_case(repo_root, validator, applier)

    with pytest.raises(TypeError):
        use_case.execute("not a request")  # type: ignore[arg-type]

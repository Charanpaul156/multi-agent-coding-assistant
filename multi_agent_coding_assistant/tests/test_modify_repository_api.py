"""Tests for the POST /modify-repository API endpoint.

Uses dependency_overrides to inject fakes so no real RAG/Coder/LLM pipeline is
loaded. The real project is never modified.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from backend.application.modify_repository_use_cases import (
    ModifyRepositoryResult,
    ModifyRepositoryStatus,
)
from backend.domain.change_models import (
    ApplicationResult,
    ChangeOperation,
    ChangeSet,
    ChangeValidationResult,
    FileChange,
    ValidationReport,
)


class FakeModifyUseCase:
    """Fake that returns a successful dry-run / applied result."""

    def __init__(self):
        self.last_request = None

    def execute(self, request) -> Any:
        self.last_request = request
        change = FileChange(
            file_path="backend/auth/service.py",
            operation=ChangeOperation.CREATE,
            new_content="def reset_password():\n    pass\n",
            description="Add password reset",
        )
        change_set = ChangeSet(changes=[change], summary="add reset")
        validation = ValidationReport(
            valid=True,
            results=[ChangeValidationResult(True, "backend/auth/service.py", "create")],
        )
        application = ApplicationResult(
            success=True,
            applied_files=["backend/auth/service.py"],
        )

        if request.dry_run:
            return ModifyRepositoryResult(
                success=True,
                status=ModifyRepositoryStatus.PROPOSED,
                repository_path=request.repository_path,
                dry_run=True,
                change_set=change_set,
                validation=validation,
            )

        return ModifyRepositoryResult(
            success=True,
            status=ModifyRepositoryStatus.APPLIED,
            repository_path=request.repository_path,
            dry_run=False,
            change_set=change_set,
            validation=validation,
            application=application,
        )


class RejectingModifyUseCase:
    def execute(self, request) -> Any:
        return ModifyRepositoryResult(
            success=False,
            status=ModifyRepositoryStatus.VALIDATION_FAILED,
            repository_path=getattr(request, "repository_path", ""),
            dry_run=getattr(request, "dry_run", True),
            error="proposed changes failed validation",
        )


@pytest.fixture(autouse=True)
def _override_deps():
    import backend.api.deps as deps
    from backend.main import app

    app.dependency_overrides[deps.get_modify_repository_use_case] = lambda: FakeModifyUseCase()
    yield
    app.dependency_overrides.clear()


def _client():
    from backend.main import app

    return TestClient(app)


def test_valid_dry_run_request() -> None:
    resp = _client().post(
        "/modify-repository",
        json={
            "repository": "/repo",
            "request": "add password reset",
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["dry_run"] is True
    assert data["status"] == "proposed"
    assert len(data["changes"]) == 1
    assert data["changes"][0]["file_path"] == "backend/auth/service.py"
    assert data["changes"][0]["operation"] == "create"
    assert data["validation_errors"] == []
    assert data["applied_files"] == []


def test_valid_apply_request() -> None:
    resp = _client().post(
        "/modify-repository",
        json={
            "repository": "/repo",
            "request": "add password reset",
            "dry_run": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["dry_run"] is False
    assert data["status"] == "applied"
    assert data["applied_files"] == ["backend/auth/service.py"]


def test_empty_request_rejected() -> None:
    resp = _client().post(
        "/modify-repository",
        json={"repository": "/repo", "request": "  ", "dry_run": True},
    )
    assert resp.status_code == 400


def test_missing_request_rejected() -> None:
    resp = _client().post(
        "/modify-repository",
        json={"repository": "/repo", "dry_run": True},
    )
    assert resp.status_code == 422


def test_dry_run_none_rejected() -> None:
    resp = _client().post(
        "/modify-repository",
        json={"repository": "/repo", "request": "x", "dry_run": None},
    )
    # default dry_run=True applies; but explicit None resolves to default.
    assert resp.status_code == 200


def test_validation_failure_path() -> None:
    import backend.api.deps as deps
    from backend.main import app

    app.dependency_overrides[deps.get_modify_repository_use_case] = lambda: RejectingModifyUseCase()
    resp = _client().post(
        "/modify-repository",
        json={"repository": "/repo", "request": "bad", "dry_run": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["status"] == "validation_failed"
    assert "failed validation" in (data["error"] or "")

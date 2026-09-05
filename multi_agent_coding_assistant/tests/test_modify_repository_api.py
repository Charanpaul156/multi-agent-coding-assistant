"""Tests for the POST /modify-repository API endpoint.

Uses FastAPI dependency overrides and temporary directories so the endpoint
can be exercised without any real RAG, LLM, or filesystem mutation.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agents.planner_agent import ImplementationPlan
from backend.application.modify_repository_use_cases import (
    ModifyRepositoryRequest,
    ModifyRepositoryResult,
    ModifyRepositoryStatus,
)
from backend.application.planning_use_cases import GeneratePlanResult
from backend.application.rag_use_cases import SearchRepositoryResult
from backend.domain.change_models import (
    ApplicationResult,
    ChangeOperation,
    ChangeSet,
    ChangeValidationResult,
    DiffEntry,
    FileChange,
    ValidationReport,
)
from rag.retriever import RetrievedChunk


def _plan() -> ImplementationPlan:
    return ImplementationPlan(
        problem_summary="add password reset",
        project_type="python-service",
        requirements=["update auth service"],
        modules=["backend/auth/service.py"],
        functions=["reset_password"],
        classes=[],
        external_libraries=[],
        database_needed=False,
        api_needed=[],
        algorithm="simple",
        edge_cases=[],
        estimated_complexity="low",
        future_improvements=[],
    )


def _context_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        content="def login():\n    pass\n",
        file_path="backend/auth/service.py",
        start_line=1,
        end_line=2,
        language="python",
        repository="repo",
        chunk_index=0,
        distance=0.1,
    )


def _change_set() -> ChangeSet:
    return ChangeSet(
        changes=[
            FileChange(
                file_path="backend/auth/service.py",
                operation=ChangeOperation.MODIFY,
                new_content="def reset_password():\n    return True\n",
                original_hash="abc123",
                description="Add password reset",
            )
        ],
        summary="update auth service",
    )


class FakeSearchUseCase:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        return SearchRepositoryResult(
            success=True,
            query=request.query,
            results=[_context_chunk()],
        )


class FakePlanUseCase:
    def __init__(self) -> None:
        self.calls = []

    def execute(self, request):
        self.calls.append(request)
        return GeneratePlanResult(plan=_plan())


class FakeCoderAgent:
    def __init__(self) -> None:
        self.calls = []

    def generate_changes(
        self,
        prompt,
        *,
        implementation_plan=None,
        retrieved_context=None,
        plan_summary=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "implementation_plan": implementation_plan,
                "retrieved_context": retrieved_context,
                "plan_summary": plan_summary,
            }
        )
        return _change_set()


class FakeModifyUseCase:
    def __init__(self, *, success: bool = True, application_success: bool = True):
        self.calls = []
        self.success = success
        self.application_success = application_success

    def execute(self, request: ModifyRepositoryRequest) -> ModifyRepositoryResult:
        self.calls.append(request)
        c_set = _change_set()
        validation = ValidationReport(
            valid=self.success,
            results=[
                ChangeValidationResult(
                    valid=self.success,
                    file_path=c_set.changes[0].file_path,
                    operation=c_set.changes[0].operation.value,
                    messages=[] if self.success else ["failed validation"],
                )
            ],
        )
        diffs = [
            DiffEntry(
                file_path=change.file_path,
                operation=change.operation.value,
                diff_text=f"diff:{change.file_path}",
            )
            for change in c_set.changes
        ]
        applied_files = (
            [change.file_path for change in c_set.changes]
            if self.application_success and not request.dry_run
            else []
        )
        application = ApplicationResult(
            success=self.application_success,
            applied_files=applied_files,
            rollback_records=[],
            dry_run=request.dry_run,
            error=None if self.application_success else "apply failed",
        )
        return ModifyRepositoryResult(
            success=self.success and self.application_success,
            status=ModifyRepositoryStatus.APPLIED if self.success and self.application_success else ModifyRepositoryStatus.VALIDATION_FAILED,
            repository_path=request.repository_path,
            dry_run=request.dry_run,
            change_set=c_set,
            validation=validation,
            diffs=diffs,
            application=application,
            error=None if self.success and self.application_success else (
                "failed validation" if not self.success else "apply failed"
            ),
        )


@pytest.fixture
def repo_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    return root


@pytest.fixture
def fakes(repo_root):
    search = FakeSearchUseCase()
    planner = FakePlanUseCase()
    coder = FakeCoderAgent()
    modifier = FakeModifyUseCase()

    import backend.api.deps as deps
    from backend.main import app

    app.dependency_overrides[deps.get_search_repository_use_case] = lambda: search
    app.dependency_overrides[deps.get_generate_plan_use_case] = lambda: planner
    app.dependency_overrides[deps.get_coder_agent] = lambda: coder
    app.dependency_overrides[deps.get_modify_repository_use_case] = lambda: modifier

    yield {
        "search": search,
        "planner": planner,
        "coder": coder,
        "modifier": modifier,
    }

    app.dependency_overrides.clear()


def _client():
    from backend.main import app

    return TestClient(app)


def test_modify_repository_dry_run_pipeline(repo_root, fakes) -> None:
    resp = _client().post(
        "/modify-repository",
        json={
            "repository": str(repo_root),
            "request": "add password reset",
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["dry_run"] is True
    assert body["proposed_changes"][0]["file_path"] == "backend/auth/service.py"
    assert body["proposed_changes"][0]["operation"] == "modify"
    assert body["diff"][0]["file_path"] == "backend/auth/service.py"
    assert body["validation_results"][0]["valid"] is True
    assert body["applied_files"] == []
    assert body["error"] is None

    assert len(fakes["modifier"].calls) == 1
    assert fakes["modifier"].calls[0].repository_path == str(repo_root)
    assert fakes["modifier"].calls[0].dry_run is True


def test_modify_repository_apply_pipeline(repo_root, fakes) -> None:
    resp = _client().post(
        "/modify-repository",
        json={
            "repository": str(repo_root),
            "request": "add password reset",
            "dry_run": False,
            "apply": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is True
    assert body["dry_run"] is False
    assert body["applied_files"] == ["backend/auth/service.py"]
    assert body["validation_results"][0]["messages"] == []
    assert fakes["modifier"].calls[0].dry_run is False


def test_modify_repository_requires_apply_intent(repo_root, fakes) -> None:
    resp = _client().post(
        "/modify-repository",
        json={
            "repository": str(repo_root),
            "request": "add password reset",
            "dry_run": False,
        },
    )
    assert resp.status_code == 400
    assert fakes["modifier"].calls == []


def test_modify_repository_validation_failure_returns_error(repo_root, fakes) -> None:
    import backend.api.deps as deps
    from backend.main import app

    failing = FakeModifyUseCase(success=False, application_success=True)
    app.dependency_overrides[deps.get_modify_repository_use_case] = lambda: failing

    resp = _client().post(
        "/modify-repository",
        json={
            "repository": str(repo_root),
            "request": "add password reset",
            "dry_run": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["success"] is False
    assert body["error"] == "failed validation"
    assert body["validation_results"][0]["valid"] is False


def test_modify_repository_empty_request_rejected(repo_root, fakes) -> None:
    resp = _client().post(
        "/modify-repository",
        json={
            "repository": str(repo_root),
            "request": "   ",
            "dry_run": True,
        },
    )
    assert resp.status_code == 400

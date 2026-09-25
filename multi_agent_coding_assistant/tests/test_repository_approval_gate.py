"""Focused tests for the safe human-approval gate for repository modifications.

Verifies:
1. Preview can be generated without writing files (filesystem unchanged).
2. Unapproved ChangeSet cannot be applied.
3. Approved ChangeSet can be applied.
4. Rejected ChangeSet is not applied.
5. Stale ChangeSet is rejected when repository files change concurrently.
6. Validation still runs before approval.
7. Protected paths remain rejected.
8. Existing repository modification behavior still works.
9. API approval flow works via POST /modify-repository, /approve, and /reject.
10. Regression: no approval -> ChangeApplier is never called.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from backend.api.deps import (
    get_approval_store,
    get_modify_repository_use_case,
)
from backend.application.modify_repository_use_cases import (
    ModifyRepositoryRequest,
    ModifyRepositoryResult,
    ModifyRepositoryStatus,
    ModifyRepositoryUseCase,
)
from backend.domain.change_models import (
    ApplicationResult,
    ApprovalStatus,
    ChangeOperation,
    ChangeSet,
    ChangeValidationResult,
    FileChange,
    ValidationReport,
)
from backend.infrastructure.approval_store import ApprovalStore
from backend.infrastructure.change_applier import ChangeApplier
from backend.infrastructure.change_validation import ChangeValidator
from backend.infrastructure.diff_generator import DiffGenerator
from backend.main import app
from rag.config import RagConfig


@pytest.fixture
def repo_dir(tmp_path: Path) -> Path:
    repo = tmp_path / "test_repo"
    repo.mkdir()
    (repo / "hello.py").write_text("print('hello v1')\n", encoding="utf-8")
    return repo


@pytest.fixture
def rag_config(repo_dir: Path) -> RagConfig:
    return RagConfig(allowed_repository_roots=(str(repo_dir),))


class SpyApplier:
    """Spy implementation of ChangeApplier that records calls."""

    def __init__(self, real_applier: ChangeApplier):
        self._real = real_applier
        self.calls: list[tuple[ChangeSet, bool]] = []

    def apply(self, change_set: ChangeSet, *, dry_run: bool = False) -> ApplicationResult:
        self.calls.append((change_set, dry_run))
        return self._real.apply(change_set, dry_run=dry_run)


def _build_use_case(
    repo_dir: Path,
    *,
    approval_store: ApprovalStore | None = None,
    spy_applier: SpyApplier | None = None,
) -> tuple[ModifyRepositoryUseCase, SpyApplier, ApprovalStore]:
    store = approval_store or ApprovalStore()
    config = RagConfig(allowed_repository_roots=(str(repo_dir),))
    validator = ChangeValidator(config=config)
    real_applier = ChangeApplier(config=config)
    applier = spy_applier or SpyApplier(real_applier)

    use_case = ModifyRepositoryUseCase(
        config=config,
        retriever_use_case=MagicMock(),
        plan_use_case=MagicMock(),
        coder_agent=MagicMock(),
        validator=validator,
        applier=applier,  # type: ignore[arg-type]
        approval_store=store,
    )
    return use_case, applier, store


# ============================================================================
# 1. Preview can be generated without writing files
# ============================================================================

def test_preview_can_be_generated_without_writing_files(repo_dir: Path):
    """Proposing a change generates diff & preview while leaving filesystem intact."""
    use_case, applier, _ = _build_use_case(repo_dir)

    initial_content = (repo_dir / "hello.py").read_text(encoding="utf-8")
    initial_files = list(repo_dir.iterdir())

    orig_hash = hashlib.sha256(initial_content.encode("utf-8")).hexdigest()
    cs = ChangeSet(
        changes=[
            FileChange(
                file_path="hello.py",
                operation=ChangeOperation.MODIFY,
                new_content="print('hello v2 - proposed')\n",
                original_content=initial_content,
                original_hash=orig_hash,
            )
        ]
    )

    req = ModifyRepositoryRequest(
        repository_root=str(repo_dir),
        change_set=cs,
        dry_run=True,
        approval=ApprovalStatus.PREVIEW,
    )
    result = use_case.execute(req)

    assert result.success is True
    assert result.approval_status == ApprovalStatus.PREVIEW
    assert result.approval_token is not None
    assert len(result.approval_token) > 20
    assert len(result.diff) == 1
    assert "hello v2 - proposed" in result.diff[0].diff_text

    # CRITICAL: Filesystem MUST remain unchanged
    assert (repo_dir / "hello.py").read_text(encoding="utf-8") == initial_content
    assert list(repo_dir.iterdir()) == initial_files
    # During dry_run preview, ChangeApplier is never called
    assert applier.calls == []


# ============================================================================
# 2. Regression: No approval -> ChangeApplier never called to mutate files
# ============================================================================

def test_regression_no_approval_change_applier_never_mutates_files(repo_dir: Path):
    """Executing an unapproved request (even if dry_run=False) is refused before mutation."""
    use_case, applier, _ = _build_use_case(repo_dir)

    initial_content = (repo_dir / "hello.py").read_text(encoding="utf-8")
    cs = ChangeSet(
        changes=[
            FileChange(
                file_path="hello.py",
                operation=ChangeOperation.MODIFY,
                new_content="print('valid code but unapproved')\n",
            )
        ]
    )

    req = ModifyRepositoryRequest(
        repository_root=str(repo_dir),
        change_set=cs,
        dry_run=False,
        approval=ApprovalStatus.PREVIEW,  # Not approved!
    )
    result = use_case.execute(req)

    assert result.success is False
    assert result.approval_status == ApprovalStatus.PREVIEW
    assert "approval required" in (result.error or "").lower()
    # Ensure ChangeApplier was NEVER called for mutation
    assert not any(dry_run is False for _, dry_run in applier.calls)
    # File content remains untouched
    assert (repo_dir / "hello.py").read_text(encoding="utf-8") == initial_content


# ============================================================================
# 3. Unapproved ChangeSet cannot be applied
# ============================================================================

def test_unapproved_changeset_cannot_be_applied(repo_dir: Path):
    """ChangeSet with REJECTED or PREVIEW status cannot be applied."""
    use_case, applier, _ = _build_use_case(repo_dir)

    for status in (ApprovalStatus.PREVIEW, ApprovalStatus.REJECTED):
        req = ModifyRepositoryRequest(
            repository_root=str(repo_dir),
            change_set=ChangeSet(
                changes=[
                    FileChange(
                        file_path="hello.py",
                        operation=ChangeOperation.MODIFY,
                        new_content="print('valid code')\n",
                    )
                ]
            ),
            dry_run=False,
            approval=status,
        )
        res = use_case.execute(req)
        assert res.success is False
        err = (res.error or "").lower()
        assert "approval required" in err or "rejected" in err


# ============================================================================
# 4. Approved ChangeSet can be applied
# ============================================================================

def test_approved_changeset_can_be_applied(repo_dir: Path):
    """A valid proposal can be approved via token and mutates the repository."""
    use_case, applier, _ = _build_use_case(repo_dir)

    initial_content = (repo_dir / "hello.py").read_text(encoding="utf-8")
    orig_hash = hashlib.sha256(initial_content.encode("utf-8")).hexdigest()
    cs = ChangeSet(
        changes=[
            FileChange(
                file_path="hello.py",
                operation=ChangeOperation.MODIFY,
                new_content="print('approved hello')\n",
                original_content=initial_content,
                original_hash=orig_hash,
            )
        ]
    )

    # Step 1: Preview / Propose
    req = ModifyRepositoryRequest(
        repository_root=str(repo_dir),
        change_set=cs,
        dry_run=True,
        approval=ApprovalStatus.PREVIEW,
    )
    preview_res = use_case.execute(req)
    token = preview_res.approval_token
    assert token is not None

    # Step 2: Approve
    approve_res = use_case.approve(token, str(repo_dir))
    assert approve_res.success is True
    assert approve_res.approval_status == ApprovalStatus.APPLIED
    assert "hello.py" in approve_res.applied_files

    # Step 3: Check on-disk mutation
    assert (repo_dir / "hello.py").read_text(encoding="utf-8") == "print('approved hello')\n"


# ============================================================================
# 5. Rejected ChangeSet is not applied
# ============================================================================

def test_rejected_changeset_is_not_applied(repo_dir: Path):
    """Rejecting a proposal marks it rejected and subsequent application fails."""
    use_case, _, _ = _build_use_case(repo_dir)

    initial_content = (repo_dir / "hello.py").read_text(encoding="utf-8")
    cs = ChangeSet(
        changes=[
            FileChange(
                file_path="hello.py",
                operation=ChangeOperation.MODIFY,
                new_content="print('rejected content')\n",
            )
        ]
    )

    preview_res = use_case.execute(
        ModifyRepositoryRequest(
            repository_root=str(repo_dir),
            change_set=cs,
            dry_run=True,
            approval=ApprovalStatus.PREVIEW,
        )
    )
    token = preview_res.approval_token
    assert token is not None

    # Explicit reject
    reject_res = use_case.reject(token, str(repo_dir))
    assert reject_res.success is False
    assert reject_res.approval_status == ApprovalStatus.REJECTED

    # Attempt to approve rejected ticket should fail
    approve_res = use_case.approve(token, str(repo_dir))
    assert approve_res.success is False
    assert "rejected" in (approve_res.error or "").lower()

    # Disk content unchanged
    assert (repo_dir / "hello.py").read_text(encoding="utf-8") == initial_content


# ============================================================================
# 6. Stale ChangeSet is rejected when repository files change concurrently
# ============================================================================

def test_stale_changeset_is_rejected_on_concurrent_modification(repo_dir: Path):
    """If file on disk changes after proposal is generated, approval is aborted."""
    use_case, _, _ = _build_use_case(repo_dir)

    initial_content = (repo_dir / "hello.py").read_text(encoding="utf-8")
    orig_hash = hashlib.sha256(initial_content.encode("utf-8")).hexdigest()
    cs = ChangeSet(
        changes=[
            FileChange(
                file_path="hello.py",
                operation=ChangeOperation.MODIFY,
                new_content="print('proposed modification')\n",
                original_content=initial_content,
                original_hash=orig_hash,
            )
        ]
    )

    preview = use_case.execute(
        ModifyRepositoryRequest(
            repository_root=str(repo_dir),
            change_set=cs,
            dry_run=True,
            approval=ApprovalStatus.PREVIEW,
        )
    )
    token = preview.approval_token
    assert token is not None

    # Simulate concurrent modification to hello.py before user clicks approve
    (repo_dir / "hello.py").write_text("print('concurrent external edit')\n", encoding="utf-8")

    # Approve should detect stale state and refuse to overwrite
    result = use_case.approve(token, str(repo_dir))
    assert result.success is False
    assert "stale" in (result.error or "").lower()
    # File content preserved from external edit
    assert (repo_dir / "hello.py").read_text(encoding="utf-8") == "print('concurrent external edit')\n"


# ============================================================================
# 7. Validation still runs before approval
# ============================================================================

def test_validation_still_runs_before_approval(repo_dir: Path):
    """Syntax or security validation errors prevent proposal approval."""
    use_case, _, _ = _build_use_case(repo_dir)

    cs = ChangeSet(
        changes=[
            FileChange(
                file_path="bad_syntax.py",
                operation=ChangeOperation.CREATE,
                new_content="def broken_syntax(:\n",
            )
        ]
    )

    res = use_case.execute(
        ModifyRepositoryRequest(
            repository_root=str(repo_dir),
            change_set=cs,
            dry_run=True,
            approval=ApprovalStatus.PREVIEW,
        )
    )
    assert res.success is False
    assert res.validation_result is not None
    assert res.validation_result.valid is False
    assert res.approval_token is None


# ============================================================================
# 8. Protected paths remain rejected
# ============================================================================

def test_protected_paths_remain_rejected(repo_dir: Path):
    """Paths outside allowed repository roots or protected files are blocked."""
    use_case, _, _ = _build_use_case(repo_dir)

    cs = ChangeSet(
        changes=[
            FileChange(
                file_path="../outside.py",
                operation=ChangeOperation.CREATE,
                new_content="pwned\n",
            )
        ]
    )

    res = use_case.execute(
        ModifyRepositoryRequest(
            repository_root=str(repo_dir),
            change_set=cs,
            dry_run=True,
            approval=ApprovalStatus.PREVIEW,
        )
    )
    assert res.success is False
    assert res.approval_token is None
    assert not (repo_dir.parent / "outside.py").exists()


# ============================================================================
# 9. Existing repository modification behavior still works
# ============================================================================

def test_existing_repository_modification_with_explicit_approval(repo_dir: Path):
    """When approval=APPROVED is explicitly given, ChangeApplier applies the changeset."""
    use_case, _, _ = _build_use_case(repo_dir)

    cs = ChangeSet(
        changes=[
            FileChange(
                file_path="hello.py",
                operation=ChangeOperation.MODIFY,
                new_content="print('direct approved execution')\n",
            )
        ]
    )

    req = ModifyRepositoryRequest(
        repository_root=str(repo_dir),
        change_set=cs,
        dry_run=False,
        approval=ApprovalStatus.APPROVED,
    )
    res = use_case.execute(req)
    assert res.success is True
    assert res.approval_status == ApprovalStatus.APPLIED
    assert (repo_dir / "hello.py").read_text(encoding="utf-8") == "print('direct approved execution')\n"


# ============================================================================
# 10. API Approval Flow Endpoints
# ============================================================================

def test_api_approval_flow_endpoints(repo_dir: Path):
    """Verify HTTP POST /modify-repository, /approve, and /reject."""
    store = ApprovalStore()
    config = RagConfig(allowed_repository_roots=(str(repo_dir),))
    use_case = ModifyRepositoryUseCase(
        config=config,
        retriever_use_case=MagicMock(),
        plan_use_case=MagicMock(),
        coder_agent=MagicMock(),
        validator=ChangeValidator(config=config),
        applier=ChangeApplier(config=config),
        approval_store=store,
    )

    app.dependency_overrides[get_modify_repository_use_case] = lambda: use_case
    app.dependency_overrides[get_approval_store] = lambda: store

    client = TestClient(app)

    try:
        # Step 1: Propose changes via POST /modify-repository
        initial_content = (repo_dir / "hello.py").read_text(encoding="utf-8")
        orig_hash = hashlib.sha256(initial_content.encode("utf-8")).hexdigest()
        propose_payload = {
            "repository": str(repo_dir),
            "request": "Update hello",
            "dry_run": True,
            "change_set": {
                "summary": "Update hello",
                "changes": [
                    {
                        "file_path": "hello.py",
                        "operation": "modify",
                        "new_content": "print('api approved v1')\n",
                        "original_content": initial_content,
                        "original_hash": orig_hash,
                    }
                ],
            },
        }
        res_propose = client.post("/modify-repository", json=propose_payload)
        assert res_propose.status_code == 200
        data_propose = res_propose.json()
        assert data_propose["success"] is True
        assert data_propose["approval_status"] == "preview"
        token = data_propose.get("approval_token")
        assert token is not None
        # Verify file is NOT touched
        assert (repo_dir / "hello.py").read_text(encoding="utf-8") == initial_content

        # Step 2: Approve changes via POST /modify-repository/approve
        res_approve = client.post(
            "/modify-repository/approve",
            json={"approval_token": token, "repository": str(repo_dir)},
        )
        assert res_approve.status_code == 200
        data_approve = res_approve.json()
        assert data_approve["success"] is True
        assert data_approve["approval_status"] == "applied"
        assert "hello.py" in data_approve["applied_files"]
        # Verify file IS modified
        assert (repo_dir / "hello.py").read_text(encoding="utf-8") == "print('api approved v1')\n"

        # Step 3: Second approval of same token should be rejected
        res_double_approve = client.post(
            "/modify-repository/approve",
            json={"approval_token": token, "repository": str(repo_dir)},
        )
        assert res_double_approve.status_code == 200
        data_double = res_double_approve.json()
        assert data_double["success"] is False

        # Step 4: Test rejection flow with fresh proposal
        res_propose2 = client.post(
            "/modify-repository",
            json={
                "repository": str(repo_dir),
                "request": "Second update",
                "dry_run": True,
                "change_set": {
                    "summary": "Second update",
                    "changes": [
                        {
                            "file_path": "hello.py",
                            "operation": "modify",
                            "new_content": "print('api rejected')\n",
                        }
                    ],
                },
            },
        )
        token2 = res_propose2.json().get("approval_token")
        assert token2 is not None

        res_reject = client.post(
            "/modify-repository/reject",
            json={"approval_token": token2, "repository": str(repo_dir)},
        )
        assert res_reject.status_code == 200
        data_reject = res_reject.json()
        assert data_reject["success"] is False
        assert data_reject["approval_status"] == "rejected"
        # Content unchanged
        assert (repo_dir / "hello.py").read_text(encoding="utf-8") == "print('api approved v1')\n"

    finally:
        app.dependency_overrides.clear()

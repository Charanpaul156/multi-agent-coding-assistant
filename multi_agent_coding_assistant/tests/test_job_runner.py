"""Tests for JobRunner and workflow/repository event emissions.

Verifies:
1. JobRunner lifecycle:
   - runner creates a job (status QUEUED)
   - execution transitions to RUNNING then COMPLETED
   - result is stored in the job
   - failed execution transitions to FAILED
   - error is stored in the job
   - correct stable job_id is used for all events
   - completion event is emitted
   - failure event is emitted
   - event stream completion is signaled on EventBus
2. Workflow event emissions:
   - workflow emits expected stage events when emitter is supplied
   - event ordering is deterministic
   - successful workflow emits workflow_completed
   - failed workflow emits workflow_failed
   - no emitter preserves existing behavior
   - safe metadata only (no secrets, raw prompts, chain of thought)
3. Modify repository event emissions:
   - preview emits modification_started, validation_complete, diff_generated, approval_required
   - approval and rejection emit approval_approved, approval_rejected, changes_applied
   - stale ChangeSet emits stale_change_detected and modification_failed
   - existing approval enforcement unchanged
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.application.event_emitter import RecordingEventEmitter
from backend.application.job_manager import JobManager
from backend.application.job_runner import JobRunner
from backend.application.modify_repository_use_cases import (
    ModifyRepositoryRequest,
    ModifyRepositoryStatus,
    ModifyRepositoryUseCase,
)
from backend.application.workflow_use_cases import (
    RunWorkflowUseCase,
    WorkflowRequest,
    WorkflowStatus,
)
from backend.domain.change_models import (
    ApprovalStatus,
    ChangeOperation,
    ChangeSet,
    FileChange,
)
from backend.domain.job_models import JobStatus, WorkflowEvent
from backend.infrastructure.approval_store import ApprovalStore
from backend.infrastructure.change_applier import ChangeApplier
from backend.infrastructure.change_validation import ChangeValidator
from backend.infrastructure.event_bus import EventBus
from rag.config import RagConfig


# ====================================================================== #
# 1. JobRunner Tests
# ====================================================================== #

class TestJobRunnerLifecycle:
    """Verifies JobRunner creation, execution, status transitions, and events."""

    def test_runner_creates_job_queued(self) -> None:
        manager = JobManager()
        bus = EventBus()
        runner = JobRunner(job_manager=manager, event_bus=bus)

        job = runner.create_job(metadata={"test": "meta"})
        assert job.status == JobStatus.QUEUED
        assert job.metadata == {"test": "meta"}
        assert manager.get_job(job.job_id) is not None

    def test_runner_executes_and_completes_job(self) -> None:
        manager = JobManager()
        emitter = RecordingEventEmitter()
        runner = JobRunner(job_manager=manager, event_emitter=emitter)

        def sample_task(job_id: str, scoped_emitter: RecordingEventEmitter | None) -> dict[str, str]:
            # Emitting an intermediate event during execution
            if scoped_emitter is not None:
                scoped_emitter.emit_event("step_start", {"step": "work"})
            return {"status": "ok", "output": "done"}

        job = runner.run_job(sample_task, metadata={"purpose": "unit_test"})

        assert job.status == JobStatus.COMPLETED
        assert job.result == {"status": "ok", "output": "done"}
        assert job.error is None

        # Check emitted events
        events = emitter.get_events()
        event_types = [e.event_type for e in events]
        assert "step_start" in event_types
        assert "job_completed" in event_types

        # Verify all events used the exact same stable job_id
        for event in events:
            assert event.job_id == job.job_id

    def test_runner_handles_failure_and_stores_error(self) -> None:
        manager = JobManager()
        emitter = RecordingEventEmitter()
        runner = JobRunner(job_manager=manager, event_emitter=emitter)

        def failing_task(job_id: str) -> None:
            raise ValueError("calculation exploded")

        job = runner.run_job(failing_task)

        assert job.status == JobStatus.FAILED
        assert job.error == "calculation exploded"
        assert job.result is None

        events = emitter.get_events()
        assert len(events) == 1
        assert events[0].event_type == "job_failed"
        assert events[0].job_id == job.job_id
        assert events[0].payload["error"] == "calculation exploded"

    def test_runner_raise_on_error_option(self) -> None:
        manager = JobManager()
        runner = JobRunner(job_manager=manager)

        def failing_task() -> None:
            raise RuntimeError("unhandled crash")

        with pytest.raises(RuntimeError, match="unhandled crash"):
            runner.run_job(failing_task, raise_on_error=True)

    @pytest.mark.anyio
    async def test_runner_signals_event_bus_completion_on_success_and_failure(self) -> None:
        manager = JobManager()
        bus = EventBus()
        runner = JobRunner(job_manager=manager, event_bus=bus)

        # 1. Success case: subscribe before run_job finishes
        job_success = runner.create_job()
        sub_success = bus.subscribe(job_success.job_id)
        runner.run_job(lambda: "success_result", job_id=job_success.job_id)

        ev_comp = await sub_success.get(timeout=1.0)
        assert ev_comp is not None
        assert ev_comp.event_type == "job_completed"
        ev_term = await sub_success.get(timeout=1.0)
        assert ev_term is None
        assert sub_success.is_closed is True

        # 2. Failure case
        job_fail = runner.create_job()
        sub_fail = bus.subscribe(job_fail.job_id)

        def bad_task():
            raise RuntimeError("failure")

        runner.run_job(bad_task, job_id=job_fail.job_id)
        ev_fail = await sub_fail.get(timeout=1.0)
        assert ev_fail is not None
        assert ev_fail.event_type == "job_failed"
        ev_term2 = await sub_fail.get(timeout=1.0)
        assert ev_term2 is None
        assert sub_fail.is_closed is True

    def test_runner_accepts_existing_queued_job_id(self) -> None:
        manager = JobManager()
        runner = JobRunner(job_manager=manager)

        pre_job = runner.create_job()
        assert pre_job.status == JobStatus.QUEUED

        finished_job = runner.run_job(lambda j_id: f"handled_{j_id}", job_id=pre_job.job_id)
        assert finished_job.job_id == pre_job.job_id
        assert finished_job.status == JobStatus.COMPLETED
        assert finished_job.result == f"handled_{pre_job.job_id}"


# ====================================================================== #
# 2. Workflow Event Emission Tests
# ====================================================================== #

class TestWorkflowEvents:
    """Verifies event emissions and safety in RunWorkflowUseCase."""

    @pytest.fixture
    def mock_workflow_use_case(self) -> RunWorkflowUseCase:
        # Mock use case dependencies
        fake_plan = MagicMock()
        fake_plan.execute.return_value = MagicMock(
            plan=MagicMock(problem_summary="Calc", modules=["calc"], functions=["add"])
        )

        fake_code = MagicMock()
        fake_code.execute.return_value = MagicMock(generated_code="def add(a, b): return a + b\n")

        fake_test_gen = MagicMock()
        fake_test_gen.execute.return_value = MagicMock(
            report=MagicMock(generated_test_code="def test_add(): assert add(1, 2) == 3\n")
        )

        fake_exec = MagicMock()
        fake_exec.execute.return_value = MagicMock(
            success=True,
            exit_code=0,
            status="success",
            output="tests passed",
            stderr="",
        )

        fake_test_exec = MagicMock()
        mock_test_res = MagicMock()
        mock_test_res.success = True
        mock_test_res.exit_code = 0
        mock_test_res.status = "passed"
        mock_test_res.passed = 1
        mock_test_res.failed = 0
        mock_test_res.failed_tests = 0
        mock_test_res.total_tests = 1
        mock_test_res.passed_tests = 1
        mock_test_res.error_tests = 0
        mock_test_res.summary = "1 passed"
        mock_test_res.stdout = "1 passed"
        mock_test_res.stderr = ""
        fake_test_exec.execute.return_value = mock_test_res

        fake_review = MagicMock()
        fake_review.execute.return_value = MagicMock(
            report=MagicMock(overall_score=95, logic_issues=[], security_concerns=[])
        )

        fake_doc = MagicMock()
        fake_doc.execute.return_value = MagicMock(
            report=MagicMock(markdown_documentation="# Calculator Docs")
        )

        return RunWorkflowUseCase(
            plan_use_case=fake_plan,
            coder_use_case=fake_code,
            test_generation_use_case=fake_test_gen,
            execute_use_case=fake_exec,
            test_execution_use_case=fake_test_exec,
            review_use_case=fake_review,
            documentation_use_case=fake_doc,
        )

    def test_workflow_emits_ordered_stage_events(self, mock_workflow_use_case: RunWorkflowUseCase) -> None:
        emitter = RecordingEventEmitter()
        job_id = "test-workflow-job-123"

        request = WorkflowRequest(
            prompt="Create an add function",
            job_id=job_id,
            event_emitter=emitter,
        )

        result = mock_workflow_use_case.execute(request)

        assert result.workflow_status == WorkflowStatus.COMPLETED
        assert result.success is True

        events = emitter.get_events()
        assert len(events) > 0

        # Check stable job_id
        for event in events:
            assert event.job_id == job_id

        step_sequence = [
            f"{e.event_type}: {e.payload['step']}" if "step" in e.payload else e.event_type
            for e in events
        ]

        expected_sequence = [
            "workflow_started",
            "step_start: planner",
            "step_complete: planner",
            "step_start: coder",
            "step_complete: coder",
            "iteration_progress",
            "step_start: test_generator",
            "step_complete: test_generator",
            "step_start: test_executor",
            "test_result",
            "step_start: reviewer",
            "review_result",
            "step_start: documentation",
            "step_complete: documentation",
            "workflow_completed",
        ]

        # Verify subsequence ordering
        seq_idx = 0
        for expected in expected_sequence:
            assert expected in step_sequence[seq_idx:], f"Expected event '{expected}' not found in remaining sequence"
            seq_idx = step_sequence.index(expected, seq_idx) + 1

    def test_failed_workflow_emits_workflow_failed(self) -> None:
        fake_plan = MagicMock()
        fake_plan.execute.side_effect = RuntimeError("Planner crash")

        uc = RunWorkflowUseCase(
            plan_use_case=fake_plan,
            coder_use_case=MagicMock(),
            test_generation_use_case=MagicMock(),
            execute_use_case=MagicMock(),
            test_execution_use_case=MagicMock(),
            review_use_case=MagicMock(),
        )

        emitter = RecordingEventEmitter()
        job_id = "fail-job-999"
        request = WorkflowRequest(
            prompt="Make app",
            job_id=job_id,
            event_emitter=emitter,
        )

        result = uc.execute(request)
        assert result.workflow_status == WorkflowStatus.PLANNING_FAILED

        events = emitter.get_events()
        event_types = [e.event_type for e in events]
        assert "workflow_started" in event_types
        assert "workflow_failed" in event_types

        failed_event = next(e for e in events if e.event_type == "workflow_failed")
        assert failed_event.payload["status"] == "planning_failed"

    def test_no_emitter_preserves_behavior(self, mock_workflow_use_case: RunWorkflowUseCase) -> None:
        # Standard synchronous execution without job_id or emitter
        request = WorkflowRequest(prompt="Create add function")
        result = mock_workflow_use_case.execute(request)
        assert result.workflow_status == WorkflowStatus.COMPLETED
        assert result.success is True
        assert result.generated_code is not None

    def test_workflow_payload_safety(self, mock_workflow_use_case: RunWorkflowUseCase) -> None:
        emitter = RecordingEventEmitter()
        request = WorkflowRequest(
            prompt="Confidential prompt with key=sk-12345 secret",
            job_id="safety-job",
            event_emitter=emitter,
        )
        mock_workflow_use_case.execute(request)

        events = emitter.get_events()
        for event in events:
            payload_str = str(event.payload)
            # Ensure no API keys or raw secret prompt strings leak into event payloads
            assert "sk-12345" not in payload_str
            assert "Confidential prompt" not in payload_str


# ====================================================================== #
# 3. Modify Repository Event Emission Tests
# ====================================================================== #

class TestModifyRepositoryEvents:
    """Verifies event emissions and safety in ModifyRepositoryUseCase."""

    @pytest.fixture
    def repo_setup(self, tmp_path: Path) -> tuple[Path, RagConfig, ApprovalStore, ModifyRepositoryUseCase]:
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / "app.py").write_text("def run(): pass\n", encoding="utf-8")

        config = RagConfig(allowed_repository_roots=(str(repo_dir),))
        store = ApprovalStore()
        validator = ChangeValidator(config=config)
        applier = ChangeApplier(config=config)

        uc = ModifyRepositoryUseCase(
            config=config,
            retriever_use_case=MagicMock(),
            plan_use_case=MagicMock(),
            coder_agent=MagicMock(),
            validator=validator,
            applier=applier,
            approval_store=store,
        )
        return repo_dir, config, store, uc

    def test_preview_emits_expected_events(
        self, repo_setup: tuple[Path, RagConfig, ApprovalStore, ModifyRepositoryUseCase]
    ) -> None:
        repo_dir, _, store, uc = repo_setup
        emitter = RecordingEventEmitter()
        job_id = "preview-job-1"

        change_set = ChangeSet(
            changes=[
                FileChange(
                    file_path="app.py",
                    operation=ChangeOperation.MODIFY,
                    new_content="def run(): return 42\n",
                )
            ]
        )

        request = ModifyRepositoryRequest(
            repository_path=str(repo_dir),
            dry_run=True,
            change_set=change_set,
            job_id=job_id,
            event_emitter=emitter,
        )

        result = uc.execute(request)
        assert result.status == ModifyRepositoryStatus.PROPOSED
        assert result.approval_token is not None

        events = emitter.get_events()
        assert len(events) > 0
        for event in events:
            assert event.job_id == job_id

        types = [e.event_type for e in events]
        assert "modification_started" in types
        assert "validation_complete" in types
        assert "diff_generated" in types
        assert "approval_required" in types

    def test_approve_and_reject_emit_events(
        self, repo_setup: tuple[Path, RagConfig, ApprovalStore, ModifyRepositoryUseCase]
    ) -> None:
        repo_dir, _, store, uc = repo_setup

        change_set = ChangeSet(
            changes=[
                FileChange(
                    file_path="app.py",
                    operation=ChangeOperation.MODIFY,
                    new_content="def run(): return 99\n",
                )
            ]
        )

        # 1. Preview to get ticket
        preview_res = uc.execute(
            ModifyRepositoryRequest(
                repository_path=str(repo_dir),
                dry_run=True,
                change_set=change_set,
            )
        )
        token = preview_res.approval_token
        assert token is not None

        # 2. Reject flow
        emitter_reject = RecordingEventEmitter()
        reject_res = uc.reject(token, repository_path=str(repo_dir), job_id="reject-job", event_emitter=emitter_reject)
        assert reject_res.status == ModifyRepositoryStatus.REJECTED

        reject_types = [e.event_type for e in emitter_reject.get_events()]
        assert "approval_rejected" in reject_types

        # 3. Create another ticket for approve flow
        preview_res2 = uc.execute(
            ModifyRepositoryRequest(
                repository_path=str(repo_dir),
                dry_run=True,
                change_set=change_set,
            )
        )
        token2 = preview_res2.approval_token

        emitter_approve = RecordingEventEmitter()
        approve_res = uc.approve(token2, repository_path=str(repo_dir), job_id="approve-job", event_emitter=emitter_approve)
        assert approve_res.status == ModifyRepositoryStatus.APPLIED
        assert approve_res.success is True

        approve_types = [e.event_type for e in emitter_approve.get_events()]
        assert "approval_approved" in approve_types
        assert "changes_applied" in approve_types

    def test_stale_changes_emit_stale_and_failure_events(
        self, repo_setup: tuple[Path, RagConfig, ApprovalStore, ModifyRepositoryUseCase]
    ) -> None:
        repo_dir, _, store, uc = repo_setup

        change_set = ChangeSet(
            changes=[
                FileChange(
                    file_path="app.py",
                    operation=ChangeOperation.MODIFY,
                    new_content="def run(): return 'fresh'\n",
                )
            ]
        )

        # Generate preview ticket
        preview_res = uc.execute(
            ModifyRepositoryRequest(
                repository_path=str(repo_dir),
                dry_run=True,
                change_set=change_set,
            )
        )
        token = preview_res.approval_token
        assert token is not None

        # Mutate the file on disk behind the scenes to trigger stale check
        (repo_dir / "app.py").write_text("def run(): return 'concurrent change'\n", encoding="utf-8")

        emitter = RecordingEventEmitter()
        apply_res = uc.approve(token, repository_path=str(repo_dir), job_id="stale-job", event_emitter=emitter)
        assert apply_res.status == ModifyRepositoryStatus.APPLICATION_FAILED
        assert apply_res.success is False

        events = emitter.get_events()
        types = [e.event_type for e in events]
        assert "stale_change_detected" in types
        assert "modification_failed" in types

    def test_approval_required_event_does_not_expose_token(
        self, repo_setup: tuple[Path, RagConfig, ApprovalStore, ModifyRepositoryUseCase]
    ) -> None:
        repo_dir, _, store, uc = repo_setup
        emitter = RecordingEventEmitter()
        job_id = "safe-preview-job"

        change_set = ChangeSet(
            changes=[
                FileChange(
                    file_path="app.py",
                    operation=ChangeOperation.MODIFY,
                    new_content="def run(): return 'guarded'\n",
                )
            ]
        )

        # 1. Preview execution
        result = uc.execute(
            ModifyRepositoryRequest(
                repository_path=str(repo_dir),
                dry_run=True,
                change_set=change_set,
                job_id=job_id,
                event_emitter=emitter,
            )
        )
        assert result.status == ModifyRepositoryStatus.PROPOSED
        # Actual approval token is returned through the normal response object:
        assert result.approval_token is not None
        actual_token = result.approval_token

        # 2. Verify approval_required event was emitted
        events = emitter.get_events()
        req_events = [e for e in events if e.event_type == "approval_required"]
        assert len(req_events) >= 1

        for ev in events:
            # Token string itself must never appear in any payload
            assert actual_token not in str(ev.payload)
            # Sensitive bearer/secret key names must not exist in any payload
            for forbidden_key in ("token", "approval_token", "ticket_token", "secret", "credential"):
                assert forbidden_key not in ev.payload

        # 3. Payload metadata contains only safe flags
        approval_event = req_events[0]
        assert approval_event.payload.get("requires_approval") is True
        assert approval_event.payload.get("approval_status") == "preview"

        # 4. Prove the approval flow still works using the token returned through normal response
        emitter_approve = RecordingEventEmitter()
        approve_result = uc.approve(
            actual_token,
            repository_path=str(repo_dir),
            job_id="safe-approve-job",
            event_emitter=emitter_approve,
        )
        assert approve_result.success is True
        assert approve_result.status == ModifyRepositoryStatus.APPLIED

        # Verify approve events also do NOT expose the token
        for ev in emitter_approve.get_events():
            assert actual_token not in str(ev.payload)
            for forbidden_key in ("token", "approval_token", "ticket_token", "secret", "credential"):
                assert forbidden_key not in ev.payload

    def test_modify_repo_payload_safety(
        self, repo_setup: tuple[Path, RagConfig, ApprovalStore, ModifyRepositoryUseCase]
    ) -> None:
        repo_dir, _, store, uc = repo_setup
        emitter = RecordingEventEmitter()

        change_set = ChangeSet(
            changes=[
                FileChange(
                    file_path="app.py",
                    operation=ChangeOperation.MODIFY,
                    new_content="SECRET_API_KEY = 'super_secret_12345'\n",
                )
            ]
        )

        res = uc.execute(
            ModifyRepositoryRequest(
                repository_path=str(repo_dir),
                dry_run=True,
                change_set=change_set,
                job_id="secret-job",
                event_emitter=emitter,
            )
        )

        for event in emitter.get_events():
            payload_str = str(event.payload)
            # Payloads should not contain full new_content or secret values
            assert "super_secret_12345" not in payload_str
            if res.approval_token:
                assert res.approval_token not in payload_str
